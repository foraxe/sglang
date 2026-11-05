"""Validate isolated decode-graph GET alongside CP-prefetch GET.

Use a small indexed checkpoint and a running research owner. Configure the same
environment as engram_remote_smoke.py plus ENGRAM_SHARED_DECODE_GRAPH=1.
"""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.engram_share.rdma.provider import RemoteEngramTable, prepare_forward


def main():
    model = Path(os.environ["ENGRAM_SHARED_MODEL"])
    config = json.loads((model / "config.json").read_text())
    config = config.get("text_config", config)
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tables = []
    try:
        for layer, rows in zip((1, 14), config["engram_num_embeddings"]):
            table = RemoteEngramTable(rows, 256, layer, str(model))
            table.loaded = {"weight", "scale"}
            table.finish_load(0, rows)
            tables.append(table)
        runtime = tables[0].runtime
        assert runtime.graph_context is not None
        reference = []
        for layer in (1, 14):
            weight = f"layers.{layer}.engram.embed.weight"
            scale = f"layers.{layer}.engram.embed.scale"
            with safe_open(
                str(model / index[weight]), framework="pt", device="cpu"
            ) as f:
                w = f.get_tensor(weight).float()
            with safe_open(
                str(model / index[scale]), framework="pt", device="cpu"
            ) as f:
                s = f.get_tensor(scale).view(torch.uint8)
            reference.append(
                (
                    w * torch.exp2(s.int().float() - 127).repeat_interleave(32, 1) + 0.0
                ).to(torch.bfloat16)
            )
        prepare_forward(False)
        stream = torch.cuda.Stream()
        graphs = []
        for rows in (32, 64):
            graph_ids = torch.zeros(rows, dtype=torch.int64, device="cuda")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_out = [
                    table.lookup(
                        graph_ids,
                        0,
                        table.num_embeddings,
                        lambda ids: torch.empty(
                            (ids.numel(), 256), dtype=torch.bfloat16, device="cuda"
                        ),
                    )
                    for table in tables
                ]
            graphs.append((graph, graph_ids, graph_out))
        for step in range(16):
            prepare_forward(True)
            full = (
                (
                    torch.arange(256 * 2 * 16, device="cuda", dtype=torch.int64)
                    + step * 31
                )
                % min(table.num_embeddings for table in tables)
            ).view(256, 2, 16)
            runtime.begin_prefetch(full, local_rows=64, cp_size=4)
            graph, graph_ids, graph_out = graphs[step % len(graphs)]
            current = (
                torch.arange(graph_ids.numel(), dtype=torch.int64) + step * 37
            ) % min(table.num_embeddings for table in tables)
            graph_ids.copy_(current.to("cuda"))
            graph.replay()
            prefetch_ids = [runtime.prefetch_ids[layer] for layer in (1, 14)]
            prefetch_out = [
                table.lookup(
                    ids,
                    0,
                    table.num_embeddings,
                    lambda x: torch.empty(
                        (x.numel(), 256), dtype=torch.bfloat16, device="cuda"
                    ),
                )
                for table, ids in zip(tables, prefetch_ids)
            ]
            torch.cuda.synchronize()
            for layer_idx in range(2):
                assert torch.equal(
                    graph_out[layer_idx].cpu().view(torch.int16),
                    reference[layer_idx][current].view(torch.int16),
                )
                assert torch.equal(
                    prefetch_out[layer_idx].cpu().view(torch.int16),
                    reference[layer_idx][prefetch_ids[layer_idx].cpu()].view(
                        torch.int16
                    ),
                )
        print(
            "PASS 16 dynamic replays across two graph buckets with overlapping 16-lane CP prefetch",
            flush=True,
        )
        for graph, _, _ in graphs:
            graph.reset()
    finally:
        if tables and tables[0].runtime is not None:
            tables[0].runtime.close()
    print("PASS reader closed", flush=True)


if __name__ == "__main__":
    main()
