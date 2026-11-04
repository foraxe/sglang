import importlib.util
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def load_adapter():
    path = Path(__file__).parents[4] / "python/sglang/srt/engram_share/rdma/provider.py"
    spec = importlib.util.spec_from_file_location("test_remote_engram_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RemoteEngramValidationTest(CustomTestCase):
    def setUp(self):
        self.module = load_adapter()

    def test_checkpoint_shapes_and_dtypes(self):
        table = self.module.RemoteEngramTable(4, 256, 1, "/model")
        weight = torch.empty((0, 256), dtype=torch.float8_e4m3fn)
        scale = torch.empty((0, 8), dtype=torch.float8_e8m0fnu)
        table.check_weight(weight, torch.empty((4, 256), dtype=weight.dtype), weight)
        table.check_weight(scale, torch.empty((4, 8), dtype=scale.dtype), weight)
        self.assertEqual(table.loaded, {"weight", "scale"})
        with self.assertRaises(ValueError):
            table.check_weight(
                weight, torch.empty((3, 256), dtype=weight.dtype), weight
            )
        with self.assertRaises(ValueError):
            table.check_weight(scale, torch.empty((4, 8)), weight)

    def test_requires_both_checkpoint_components(self):
        table = self.module.RemoteEngramTable(4, 256, 1, "/model")
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            table.finish_load(0, 4)

    def test_rejects_wrong_remote_table_size(self):
        table = self.module.RemoteEngramTable(4, 256, 1, "/model")
        table.loaded = {"weight", "scale"}
        self.module._runtime = mock.Mock(
            storage_layout="row264", metadata={"lengths": [1]}
        )
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            table.finish_load(0, 4)

    def test_unsupported_layout_and_uninitialized_prefetch_rejected(self):
        with self.assertRaises(ValueError):
            self.module.RemoteEngramTable(4, 128, 1, "/model")
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            self.module.prepare_forward(True)

    def test_rejects_unsupported_cp_at_startup(self):
        from sglang.srt.arg_groups.engram_config import (
            EngramPlacement,
            EngramReplication,
        )
        from sglang.srt.layers.engram import EngramEmbedding

        for cp in (1, 4, 8):
            self.module.validate_cp_size(cp)
        with self.assertRaisesRegex(ValueError, "CP sizes"):
            self.module.validate_cp_size(2)
        config = SimpleNamespace(
            placement=EngramPlacement.CPU,
            replication=EngramReplication.NONE,
            lookup=None,
        )
        parallel = SimpleNamespace(tp_size=1, tp_rank=0, attn_cp_size=2)
        with (
            mock.patch.dict(os.environ, {"SGLANG_EXPERIMENTAL_REMOTE_ENGRAM": "1"}),
            mock.patch("sglang.srt.layers.engram.get_parallel", return_value=parallel),
            mock.patch(
                "sglang.srt.layers.engram.resolve_engram_config", return_value=config
            ),
            self.assertRaisesRegex(ValueError, "CP sizes"),
        ):
            EngramEmbedding(4, 256, 1, (0, 4))

    def test_safetensors_loader_does_not_materialize_remote_table(self):
        from safetensors.torch import save_file

        from sglang.srt.model_loader.weight_utils import (
            buffered_multi_thread_safetensors_weights_iterator,
            safetensors_weights_iterator,
        )

        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.safetensors")
            save_file(
                {
                    "layers.1.engram.embed.weight": torch.zeros(
                        (4, 256), dtype=torch.float8_e4m3fn
                    ),
                    "layers.1.engram.embed.scale": torch.zeros(
                        (4, 8), dtype=torch.float8_e8m0fnu
                    ),
                    "other.weight": torch.ones((2, 2)),
                },
                path,
            )
            native = dict(safetensors_weights_iterator([path]))
            self.assertEqual(native["layers.1.engram.embed.weight"].device.type, "cpu")
            for iterator in (
                safetensors_weights_iterator([path], remote_engram=True),
                buffered_multi_thread_safetensors_weights_iterator(
                    [path], max_workers=1, remote_engram=True
                ),
            ):
                weights = dict(iterator)
                self.assertEqual(
                    weights["layers.1.engram.embed.weight"].device.type, "meta"
                )
                self.assertEqual(
                    weights["layers.1.engram.embed.scale"].device.type, "meta"
                )
                self.assertEqual(weights["other.weight"].device.type, "cpu")
                self.assertTrue(
                    torch.equal(weights["other.weight"], torch.ones((2, 2)))
                )
            for iterator in (
                safetensors_weights_iterator(
                    [path], disable_mmap=True, remote_engram=True
                ),
                buffered_multi_thread_safetensors_weights_iterator(
                    [path], max_workers=1, disable_mmap=True, remote_engram=True
                ),
            ):
                with self.assertRaisesRegex(ValueError, "mmap safetensors"):
                    list(iterator)

    def test_instanttensor_rejected_before_loading(self):
        from sglang.srt.configs.load_config import LoadFormat
        from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader

        loader = object.__new__(DefaultModelLoader)
        loader.load_config = SimpleNamespace(
            load_format=LoadFormat.INSTANTTENSOR, model_loader_extra_config={}
        )
        with (
            mock.patch.dict(os.environ, {"SGLANG_EXPERIMENTAL_REMOTE_ENGRAM": "1"}),
            mock.patch.object(loader, "_prepare_weights") as prepare,
            self.assertRaisesRegex(ValueError, "mmap safetensors"),
        ):
            loader._get_weights_iterator(mock.Mock())
        prepare.assert_not_called()
        with (
            mock.patch.dict(os.environ, {"SGLANG_EXPERIMENTAL_REMOTE_ENGRAM": "1"}),
            self.assertRaisesRegex(ValueError, "mmap safetensors"),
        ):
            get_model_loader(loader.load_config)

    def test_online_update_paths_reject_before_modifying_weights(self):
        from sglang.srt.arg_groups.engram_config import (
            EngramPlacement,
            EngramReplication,
        )
        from sglang.srt.layers.engram import EngramEmbedding
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            WeightUpdater,
        )

        config = SimpleNamespace(
            placement=EngramPlacement.CPU,
            replication=EngramReplication.NONE,
            lookup=None,
        )
        parallel = SimpleNamespace(tp_size=1, tp_rank=0, attn_cp_size=1)
        with (
            mock.patch.dict(os.environ, {"SGLANG_EXPERIMENTAL_REMOTE_ENGRAM": "1"}),
            mock.patch("sglang.srt.layers.engram.get_parallel", return_value=parallel),
            mock.patch(
                "sglang.srt.layers.engram.resolve_engram_config", return_value=config
            ),
            mock.patch(
                "sglang.srt.layers.engram.get_model",
                return_value=SimpleNamespace(model_path="/model"),
            ),
        ):
            model = torch.nn.Sequential(
                EngramEmbedding(4, 256, 1, (0, 4)), torch.nn.Linear(2, 2)
            )
        model._derived_weight_cache_error = self.module.weight_update_error()
        before = model[1].weight.detach().clone()
        updater = SimpleNamespace(
            get_model=lambda: model, _assert_weight_cache_inactive=lambda _: None
        )
        for result in (
            WeightUpdater.update_weights_from_disk(updater, "/other", "auto"),
            WeightUpdater.update_weights_from_tensor(updater, []),
            WeightUpdater.load_weights_from_distributed(updater, []),
            WeightUpdater.update_weights_from_ipc(updater, mock.Mock()),
        ):
            self.assertFalse(result[0])
            self.assertIn("remote Engram", result[1])
        self.assertTrue(torch.equal(before, model[1].weight))

    def test_prefetch_lookup_rejects_capture_before_submitting_read(self):
        table = self.module.RemoteEngramTable(4, 256, 1, "/model")
        table.runtime = mock.Mock(prefetch_enabled=True)
        indices = torch.tensor([0], dtype=torch.int64)
        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(
                RuntimeError, "unsupported during CUDA graph capture"
            ):
                table.lookup(indices, 0, 4, lambda ids: torch.empty((ids.numel(), 256)))
        table.runtime.fetch.assert_not_called()

    def test_plan_is_primed_only_after_both_layers_and_deferred_read_after_reduce(self):
        runtime = mock.Mock()
        runtime.storage_layout = "row264"
        runtime.metadata = {"lengths": [4 * 264, 4 * 264]}
        runtime.ranges = {}
        runtime.prefetch_enabled = True
        runtime.defer_layer14 = True
        self.module._runtime = runtime
        one = self.module.RemoteEngramTable(4, 256, 1, "/model")
        two = self.module.RemoteEngramTable(4, 256, 14, "/model")
        one.loaded = two.loaded = {"weight", "scale"}
        with (
            mock.patch.dict(os.environ, {"ENGRAM_SHARED_READ_PLAN": "1"}),
            mock.patch("torch.cuda.stream", return_value=nullcontext()),
        ):
            one.finish_load(0, 2)
            runtime.context.prime.assert_not_called()
            two.finish_load(2, 2)
        runtime.context.prime.assert_called_once()
        self.assertEqual(runtime.ranges, {1: (0, 2), 14: (2, 4)})
        one.after_reduction()
        runtime.issue_prefetch_layer.assert_called_once_with(14)


if __name__ == "__main__":
    unittest.main()
