"""Validate packaged remote Engram QP16 prefetch against checkpoint rows.

Start the experimental owner first and set the ENGRAM_SHARED_* configuration
shown in remote_engram_proposal.md. This test needs one assigned CUDA GPU,
access to the same checkpoint, and the matching legacy native library.
"""

import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from sglang.srt.engram_share.rdma import provider as remote_engram


def main():
    model = Path(os.environ["ENGRAM_SHARED_MODEL"])
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    config = json.loads((model / "config.json").read_text())
    config = config.get("text_config", config)
    layers = (1, 14)
    tables = []
    reader = None
    try:
        for layer, count in zip(layers, config["engram_num_embeddings"]):
            table = remote_engram.RemoteEngramTable(int(count), 256, layer, str(model))
            weight = torch.empty((0, 256), dtype=torch.float8_e4m3fn)
            scale = torch.empty((0, 8), dtype=torch.float8_e8m0fnu)
            table.check_weight(
                weight,
                SimpleNamespace(shape=(int(count), 256), dtype=weight.dtype),
                weight,
            )
            table.check_weight(
                scale, SimpleNamespace(shape=(int(count), 8), dtype=scale.dtype), weight
            )
            table.finish_load(0, int(count))
            tables.append(table)
            reader = table.runtime
            if (
                layer == 1
                and os.environ.get("ENGRAM_SMOKE_GATHER_NATIVE_CONTROL") == "1"
            ):
                if not hasattr(reader.context, "threshold"):
                    raise RuntimeError("native-control mode requires the gather plan")
                # Keep both endpoint sets and execution leases in both arms.
                # Set this before the second layer primes either cached graph.
                reader.context.threshold = reader.max_queries + 1
        if not reader.prefetch_enabled or not hasattr(reader.context, "graph_sizes"):
            raise RuntimeError("enable prefetch and the prepared READ plan")
        barrier = os.environ.get("ENGRAM_SMOKE_BARRIER_DIR")
        if barrier:
            barrier_path = Path(barrier)
            rank = os.environ["ENGRAM_SMOKE_BARRIER_RANK"]
            (barrier_path / f"reader-{rank}.ready").touch(exist_ok=False)
            deadline = time.monotonic() + 600
            while not (barrier_path / "release").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("multi-reader test barrier timed out")
                time.sleep(1)
        values = [0, 1, 1234, 8191]
        base = torch.tensor(values, dtype=torch.int64, device="cuda")
        tokens = int(os.environ.get("ENGRAM_SMOKE_TOKENS", "256"))
        if tokens < 4 or tokens % 4:
            raise ValueError("ENGRAM_SMOKE_TOKENS must be a positive multiple of four")
        full = base[torch.arange(tokens * 2 * 16, device="cuda").remainder(4)].view(
            tokens, 2, 16
        )
        remote_engram.prepare_forward(True)
        remote_engram.prefetch_full_hash(full, tokens // 4, 4)
        ordered = torch.cat([full[rank::4] for rank in range(4)], dim=0)
        references = []
        for layer_index, layer in enumerate(layers):
            count = int(config["engram_num_embeddings"][layer_index])
            ids = ordered[:, layer_index, :].contiguous().view(-1)
            output = tables[layer_index].lookup(
                ids,
                0,
                count,
                lambda query: torch.empty(
                    (query.numel(), 256), dtype=torch.bfloat16, device="cuda"
                ),
            )
            torch.cuda.synchronize()
            key_w = f"layers.{layer}.engram.embed.weight"
            key_s = f"layers.{layer}.engram.embed.scale"
            with safe_open(
                str(model / index[key_w]), framework="pt", device="cpu"
            ) as f:
                weights = torch.cat(
                    [f.get_slice(key_w)[i : i + 1] for i in values]
                ).float()
                scales = torch.cat(
                    [f.get_slice(key_s)[i : i + 1] for i in values]
                ).view(torch.uint8)
            expected = (
                weights
                * torch.exp2(scales.int().float() - 127).repeat_interleave(32, 1)
                + 0.0
            ).to(torch.bfloat16)
            references.append(expected)
            ids_cpu = ids.cpu()
            actual = output.cpu()
            for i, value in enumerate(values):
                selected = actual[ids_cpu == value].contiguous().view(torch.int16)
                reference = (
                    expected[i]
                    .expand(selected.shape[0], -1)
                    .contiguous()
                    .view(torch.int16)
                )
                assert torch.equal(selected, reference), (layer, value)
            print(f"PASS layer={layer} selections={ids.numel()}", flush=True)
        # Non-CP health requests use a single lane of the same prepared plan.
        remote_engram.prepare_forward(False)
        for layer_index, layer in enumerate(layers):
            count = int(config["engram_num_embeddings"][layer_index])
            actual = tables[layer_index].lookup(
                base,
                0,
                count,
                lambda query: torch.empty(
                    (query.numel(), 256), dtype=torch.bfloat16, device="cuda"
                ),
            )
            torch.cuda.synchronize()
            assert torch.equal(
                actual.cpu().view(torch.int16),
                references[layer_index].view(torch.int16),
            ), layer
            print(f"PASS noncp layer={layer} selections={base.numel()}", flush=True)
        if reader.context.graph_replays < 2:
            raise RuntimeError("expected a graph replay for each layer")
        print(f"PASS graph_replays={reader.context.graph_replays}", flush=True)
        rounds = int(os.environ.get("ENGRAM_SMOKE_TIMING_ROUNDS", "0"))
        if rounds:
            # Exercise distinct rows through the same provider/prefetch path.
            row_ids = torch.arange(tokens * 16, dtype=torch.int64, device="cuda")
            full_unique = torch.stack((row_ids.view(tokens, 16),) * 2, dim=1)
            ordered_unique = torch.cat([full_unique[rank::4] for rank in range(4)])
            expected_layers = []
            for layer_index, layer in enumerate(layers):
                key_w = f"layers.{layer}.engram.embed.weight"
                key_s = f"layers.{layer}.engram.embed.scale"
                with safe_open(
                    str(model / index[key_w]), framework="pt", device="cpu"
                ) as f:
                    weights = f.get_slice(key_w)[: tokens * 16].float()
                    scales = f.get_slice(key_s)[: tokens * 16].view(torch.uint8)
                expected = (
                    weights
                    * torch.exp2(scales.int().float() - 127).repeat_interleave(32, 1)
                ).to(torch.bfloat16)
                ids_cpu = ordered_unique[:, layer_index, :].reshape(-1).cpu()
                expected_layers.append(expected[ids_cpu].contiguous().view(torch.int16))
            samples = []
            for iteration in range(rounds + 3):
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                wall = time.perf_counter()
                start.record()
                remote_engram.prepare_forward(True)
                remote_engram.prefetch_full_hash(full_unique, tokens // 4, 4)
                results = []
                for layer_index, layer in enumerate(layers):
                    ids = ordered_unique[:, layer_index, :].contiguous().view(-1)
                    results.append(
                        tables[layer_index].lookup(
                            ids,
                            0,
                            int(config["engram_num_embeddings"][layer_index]),
                            lambda query: torch.empty(
                                (query.numel(), 256),
                                dtype=torch.bfloat16,
                                device="cuda",
                            ),
                        )
                    )
                end.record()
                end.synchronize()
                wall_ms = (time.perf_counter() - wall) * 1000
                gpu_ms = start.elapsed_time(end)
                for layer_index, (actual, expected) in enumerate(
                    zip(results, expected_layers)
                ):
                    bits = actual.cpu().contiguous().view(torch.int16)
                    if not torch.equal(bits, expected):
                        mismatch = (bits != expected).nonzero()
                        row, col = mismatch[0].tolist()
                        raise AssertionError(
                            f"layer={layers[layer_index]} iteration={iteration} "
                            f"query={row} col={col} "
                            f"id={int(ordered_unique[:, layer_index, :].reshape(-1)[row])} "
                            f"got={int(bits[row, col])} expected={int(expected[row, col])} "
                            f"mismatches={mismatch.shape[0]}"
                        )
                if iteration >= 3:
                    samples.append({"gpu_ms": gpu_ms, "wall_ms": wall_ms})
            print(
                "TIMING "
                + json.dumps(
                    {
                        "rows_per_layer": tokens * 16,
                        "rounds": rounds,
                        "gpu_p50_ms": statistics.median(x["gpu_ms"] for x in samples),
                        "wall_p50_ms": statistics.median(x["wall_ms"] for x in samples),
                        "samples": samples,
                    }
                ),
                flush=True,
            )
    finally:
        if reader is not None:
            reader.close()
    print("PASS reader closed", flush=True)


if __name__ == "__main__":
    main()
