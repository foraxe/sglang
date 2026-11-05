"""Check local GPU lookup and a second-process remote lookup use one owner."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.engram_share.rdma.local import LocalTable


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
            table = LocalTable(rows, 256, layer, str(model))
            table.loaded = {"weight", "scale"}
            table.finish_load(0, rows)
            tables.append(table)
        runtime = tables[0].runtime
        assert runtime is tables[1].runtime
        owner = runtime.lease.metadata["local_backing"]
        assert os.fstat(runtime.fd).st_ino == int(owner["inode"])
        # Keep the local lease and mapping alive while a separate remote reader
        # attaches to, reads, and releases this same owner generation.
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("engram_remote_smoke.py"))],
            check=True,
        )
        ids_cpu = torch.tensor([0, 1, 1234, 8191], dtype=torch.int64)
        ids = ids_cpu.to("cuda")
        for layer, table in zip((1, 14), tables):
            result = table.lookup(
                ids,
                lambda x: torch.empty(
                    (x.numel(), 256), dtype=torch.bfloat16, device="cuda"
                ),
            )
            torch.cuda.synchronize()
            weight = f"layers.{layer}.engram.embed.weight"
            scale = f"layers.{layer}.engram.embed.scale"
            with safe_open(
                str(model / index[weight]), framework="pt", device="cpu"
            ) as f:
                w = torch.cat(
                    [f.get_slice(weight)[i : i + 1] for i in ids_cpu.tolist()]
                ).float()
            with safe_open(
                str(model / index[scale]), framework="pt", device="cpu"
            ) as f:
                s = torch.cat(
                    [f.get_slice(scale)[i : i + 1] for i in ids_cpu.tolist()]
                ).view(torch.uint8)
            expected = (
                w * torch.exp2(s.int().float() - 127).repeat_interleave(32, 1) + 0.0
            ).to(torch.bfloat16)
            assert torch.equal(
                result.cpu().view(torch.int16), expected.view(torch.int16)
            )
            print(f"PASS local layer={layer} rows={ids.numel()}", flush=True)
        graph_ids = ids.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            graph_out = [
                table.lookup(
                    graph_ids,
                    lambda x: torch.empty(
                        (x.numel(), 256), dtype=torch.bfloat16, device="cuda"
                    ),
                )
                for table in tables
            ]
        for _ in range(4):
            graph_ids.copy_(ids)
            graph.replay()
            torch.cuda.synchronize()
            for layer_idx, table in enumerate(tables):
                eager = table.lookup(
                    ids,
                    lambda x: torch.empty(
                        (x.numel(), 256), dtype=torch.bfloat16, device="cuda"
                    ),
                )
                assert torch.equal(graph_out[layer_idx], eager)
        graph.reset()
        print("PASS local CUDA graph replay", flush=True)
        print("PASS local and remote readers used the same owner backing", flush=True)
    finally:
        if tables and tables[0].runtime is not None:
            tables[0].runtime.close()
    print("PASS local reader closed", flush=True)


if __name__ == "__main__":
    main()
