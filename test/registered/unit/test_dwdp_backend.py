import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_BACKENDS_PATH = (
    Path(__file__).parents[3] / "python/sglang/srt/layers/moe/dwdp/backends.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "dwdp_backends_under_test", _BACKENDS_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
get_dwdp_backend = _MODULE.get_dwdp_backend


class TestDwdpBackendSelection(unittest.TestCase):
    def test_invalid_backend_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported DWDP VMM backend"):
            get_dwdp_backend("typo")

    def test_cotensor_backend_is_selected_without_native_fallback(self):
        module_name = "sglang.srt.layers.moe.dwdp.cotensor_backend"
        candidate = types.ModuleType(module_name)
        candidate.CoTensorDWDPTransport = type("CoTensorDWDPTransport", (), {})
        candidate.CoTensorWeightBuffer = type("CoTensorWeightBuffer", (), {})

        with patch.dict(sys.modules, {module_name: candidate}):
            transport, weight_buffer = get_dwdp_backend("cotensor")

        self.assertIs(transport, candidate.CoTensorDWDPTransport)
        self.assertIs(weight_buffer, candidate.CoTensorWeightBuffer)

    def test_missing_cotensor_adapter_propagates_import_error(self):
        module_name = "sglang.srt.layers.moe.dwdp.cotensor_backend"
        with patch.dict(sys.modules, {module_name: None}):
            with self.assertRaises(ModuleNotFoundError):
                get_dwdp_backend("cotensor")


if __name__ == "__main__":
    unittest.main()
