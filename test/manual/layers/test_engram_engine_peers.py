"""Run with CUDA_VISIBLE_DEVICES=0,2 on two GPUs in one FABRIC domain."""

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
from cuda.bindings import driver as cu

from sglang.srt.arg_groups.engram_config import EngramConfig
from sglang.srt.layers.engram import EngramEmbedding
from sglang.srt.models.qwen4_exp import Qwen4ExpPeerEmbedding
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils.cuda_vmm_utils import check_drv


def worker(device, store, transport, placement):
    torch.cuda.set_device(device)
    rows, dim = 16, 256
    full = (torch.arange(rows * 2 * dim, device="cuda").view(-1, dim) % 97 - 48).to(
        torch.float8_e4m3fn
    )
    scales = torch.full((rows * 2, 8), 127, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    config = EngramConfig.from_dict(
        dict(
            lookup="peer_read",
            placement=placement,
            peer_group_size=2,
            peer_group_store=store,
            peer_group_transport=transport,
            peer_group_max_tokens=32,
        )
    )
    assert not torch.distributed.is_initialized()
    with (
        get_context().override_server_args(
            model_path="/test/checkpoint",
            revision="test",
            tp_size=1,
            engram_config=config,
        ),
        get_parallel().override(tp_rank=0),
    ):
        table = EngramEmbedding(rows * 2, dim, 1, (0, rows, rows * 2))
        group = table.pool.group
        assert table.row_start == group.rank_in_group * rows
        assert table.weight.shape == (rows, dim)
        table.weight.weight_loader(table.weight, full.cpu())
        table.scale.weight_loader(table.scale, scales)
        table.finish_load("layer.1")

    if placement == "cpu":
        assert table.weight.device.type == "cpu"
    else:
        handle = check_drv(
            cu.cuMemRetainAllocationHandle(table.weight.data_ptr()),
            "retain shard allocation",
        )
        try:
            prop = check_drv(
                cu.cuMemGetAllocationPropertiesFromHandle(handle),
                "query Engram shard backing",
            )
            location = cu.CUmemLocationType
            assert prop.location.type == (
                location.CU_MEM_LOCATION_TYPE_HOST_NUMA
                if placement == "egm"
                else location.CU_MEM_LOCATION_TYPE_DEVICE
            )
        finally:
            check_drv(cu.cuMemRelease(handle), "release retained shard handle")

    ids = torch.tensor([[0, 15, 16, 31]], device="cuda")
    assert torch.equal(table(ids), full.to(torch.bfloat16)[ids])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = table(ids)
    # The ranks can service different queries without a lookup collective.
    ids.copy_(torch.tensor([[31, 16, 0, 16 - device]], device="cuda"))
    graph.replay()
    assert torch.equal(output, full.to(torch.bfloat16)[ids])
    torch.cuda.synchronize()
    if transport == "rdma":
        compiled = torch.compile(table, fullgraph=True)
        assert torch.equal(compiled(ids), full.to(torch.bfloat16)[ids])

    embedding = SimpleNamespace(
        org_vocab_size=rows * 2,
        embedding_dim=160,
        weight_scale=torch.ones(1, device="cuda", dtype=torch.bfloat16),
    )
    with get_context().override_server_args(
        model_path="/test/checkpoint", revision="test"
    ):
        ple = Qwen4ExpPeerEmbedding(embedding, config, 4, "test.ple")
    ple_full = full[:, :160].contiguous()
    ple.load_rows(ple_full[:17].cpu(), 0, 17)
    ple.load_rows(ple_full[17:].cpu(), 17, rows * 2)
    ple.publish()
    assert torch.equal(ple(ids), ple_full.to(torch.bfloat16)[ids])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = ple(ids)
    ids.copy_(ids.flip(1))
    graph.replay()
    assert torch.equal(output, ple_full.to(torch.bfloat16)[ids])
    torch.cuda.synchronize()
    if transport == "rdma":
        compiled = torch.compile(ple, fullgraph=True)
        assert torch.equal(compiled(ids), ple_full.to(torch.bfloat16)[ids])

    try:
        group.group.exchange(("mismatched identity", group.rank_in_group), None)
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("expected checkpoint identity mismatch")
    print(f"PASS peer {group.rank_in_group}: {transport}/{placement}, Engram+PLE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("nvlink", "rdma"), default="nvlink")
    parser.add_argument("--placement", choices=("gpu", "egm", "cpu"), default="gpu")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        mp.spawn(
            worker,
            args=(str(Path(temp) / "peers"), args.transport, args.placement),
            nprocs=2,
        )
