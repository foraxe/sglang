"""Fail-closed selection of the DWDP VMM lifecycle implementation."""

from __future__ import annotations

import importlib
from typing import Any, Tuple, Type


def get_dwdp_backend(name: str) -> Tuple[Type[Any], Type[Any]]:
    """Return ``(transport_class, weight_buffer_class)`` for *name*.

    Imports are lazy so the default native path has no coTensor dependency. The
    candidate path intentionally propagates import/setup errors: an A/B run must
    never silently execute the native implementation under a coTensor label.
    """

    if name == "native":
        transport = importlib.import_module(
            "sglang.srt.layers.moe.dwdp.transport"
        ).DWDPTransport
        weight_buffer = importlib.import_module(
            "sglang.srt.layers.moe.dwdp.weight_buffer"
        ).WeightBuffer
        return transport, weight_buffer

    if name == "cotensor":
        candidate = importlib.import_module(
            "sglang.srt.layers.moe.dwdp.cotensor_backend"
        )
        return candidate.CoTensorDWDPTransport, candidate.CoTensorWeightBuffer

    raise ValueError(
        f"Unsupported DWDP VMM backend {name!r}; expected 'native' or 'cotensor'"
    )
