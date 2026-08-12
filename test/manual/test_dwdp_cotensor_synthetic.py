"""Two-rank real DWDP lifecycle E1 without model/kernel dependencies.

Run with::

    torchrun --standalone --nproc-per-node=2 \
      test/manual/test_dwdp_cotensor_synthetic.py --backend native

The harness uses the production DWDP transport, composite weight buffer, and
double-buffer prefetch manager. It changes only the selected VMM backend.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.cuda_vmm_utils import get_device_granularity
from sglang.srt.layers.moe.dwdp.backends import get_dwdp_backend
from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    build_layer_weight_specs,
)
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _hbm_used(device: int) -> int:
    free, total = torch.cuda.mem_get_info(device)
    return total - free


def _pattern(global_expert: int, layer: int, weight: int) -> float:
    return float(100 * layer + 10 * weight + global_expert + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "cotensor"), required=True)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"synthetic E1 requires exactly 2 ranks, got {world_size}")

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu_group = dist.new_group(backend="gloo")
    group = SimpleNamespace(device_group=dist.group.WORLD, cpu_group=cpu_group)

    fd_before = _fd_count()
    hbm_before = _hbm_used(rank)
    granularity = get_device_granularity(rank)
    elements_per_expert = (
        granularity // torch.empty((), dtype=torch.float32).element_size()
    )
    layout = DwdpExpertLayout(4, world_size, rank)

    local_params = {}
    for layer_idx in range(2):
        for weight_idx, name in enumerate(("w13_weight", "w2_weight")):
            values = [
                torch.full(
                    (elements_per_expert,),
                    _pattern(global_expert, layer_idx, weight_idx),
                    dtype=torch.float32,
                    device=rank,
                )
                for global_expert in range(
                    layout.local_expert_start, layout.local_expert_end
                )
            ]
            local_params[(layer_idx, name)] = torch.stack(values)

    specs = build_layer_weight_specs(local_params, layout.num_routed_experts)
    transport_cls, buffer_cls = get_dwdp_backend(args.backend)
    transport = transport_cls.create(specs, local_params, group, layout, rank)
    weight_buffer = buffer_cls.create(
        specs,
        transport.handle_set,
        layout.local_expert_start,
        layout.local_expert_end,
        world_size,
        rank,
    )
    manager = DWDPWeightManager(
        weight_buffer,
        transport.peer_views,
        layout.peer_ranges,
        [0, 1],
        ["w13_weight", "w2_weight"],
        rank,
        world_size,
        transport,
    )

    manager.prefetch_first_layers()
    for layer_idx in range(2):
        manager.wait_prefetch(layer_idx)
    torch.cuda.synchronize(rank)

    # Measure the unchanged production copy-stream/event protocol after one
    # warmup. The 200 layer-prefetch operations alternate both buffer slots.
    repetitions = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter()
    for _ in range(repetitions):
        for layer_idx in range(2):
            manager.prefetch_layer(layer_idx)
            manager.wait_prefetch(layer_idx)
    end.record()
    end.synchronize()
    prefetch_gpu_ms = start.elapsed_time(end)
    prefetch_wall_ms = (time.perf_counter() - wall_start) * 1000

    mismatches = 0
    hashes = {}
    for layer_idx in range(2):
        for weight_idx, name in enumerate(("w13_weight", "w2_weight")):
            tensor = weight_buffer.get_full_tensor(layer_idx, name)
            observed = tensor[:, 0].cpu().tolist()
            expected = [
                _pattern(global_expert, layer_idx, weight_idx)
                for global_expert in range(4)
            ]
            mismatches += sum(a != b for a, b in zip(observed, expected))
            hashes[f"{layer_idx}:{name}"] = float(tensor.double().sum().item())

    dist.barrier()
    manager.release()
    del manager, weight_buffer, transport, local_params
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(rank)
    dist.barrier()

    result = {
        "backend": args.backend,
        "rank": rank,
        "classification": "PASS" if mismatches == 0 else "FAIL",
        "mismatches": mismatches,
        "hashes": hashes,
        "fd_delta": _fd_count() - fd_before,
        "hbm_delta_bytes": _hbm_used(rank) - hbm_before,
        "granularity": granularity,
        "prefetch_ops": repetitions * 2,
        "prefetch_gpu_ms": prefetch_gpu_ms,
        "prefetch_wall_ms": prefetch_wall_ms,
    }
    print("DWDP_SYNTHETIC_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    if mismatches:
        raise RuntimeError(f"DWDP synthetic E1 mismatches={mismatches}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
