# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/distributed/engram_pool at 36b40f524c (Apache-2.0).
"""Peer HBM or EGM over NVLink: every GPU maps each shard (cuMem FABRIC), so
a fetch is one kernel that loads each row from its owner's memory."""

import torch

from sglang.srt.engram_share.pool.memory import CuMem, device_view, gpu_numa_node
from sglang.srt.engram_share.pool.rows import SHARD_BASES, locate
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_items"])
def _gather_kernel(
    ids,
    meta,
    out,
    num_items,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Load `[..., TABLES, K]` rows from their owners into `[TABLES, ..., K]`
    rows, one 8-byte word per lane; dead ids give zeros."""
    word = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    item = word // WORDS
    valid = item < num_items
    index, table, owner, local, live = locate(ids, meta, item, valid, TABLES, K)
    shards = tl.load(table + SHARD_BASES, mask=live, other=0)
    base = tl.load(shards.to(tl.pointer_type(tl.int64)) + owner, mask=live, other=0)
    column = word - item * WORDS
    data = tl.load(
        base.to(tl.pointer_type(tl.uint64)) + local * WORDS + column, mask=live, other=0
    )
    place = index * (num_items // TABLES) + item // (TABLES * K) * K + item % K
    tl.store(
        out.to(tl.pointer_type(tl.uint64)) + place * WORDS + column, data, mask=valid
    )


class NvlinkTransport:
    """`group` provides rank_in_group and all_gather(bytes)."""

    def __init__(self, group, host_shards: bool) -> None:
        self.group = group
        self.host_numa = gpu_numa_node() if host_shards else None
        if self.host_numa is not None and self.host_numa < 0:
            raise RuntimeError("NVLink Engram EGM shards require a CUDA host NUMA node")

    def shard(self, num_bytes: int, name: str) -> tuple[torch.Tensor, list, list]:
        """Allocate this GPU's shard and map every GPU's. Returns the shard, its
        `meta` arrays (shard addresses) and what keeps them alive. Collective."""
        memory = CuMem(num_bytes, fabric=True, host_numa=self.host_numa)
        exported = self.group.all_gather(memory.export_fabric())
        addresses = []
        for rank, handle in enumerate(exported):
            if rank == self.group.rank_in_group:
                addresses.append(memory.map())
                continue
            imported = memory.import_fabric(handle)
            addresses.append(memory.map(imported, read_only=self.host_numa is not None))
            memory.release(imported)
        own = addresses[self.group.rank_in_group]
        bases = torch.tensor(addresses, dtype=torch.int64, device="cuda")
        return device_view(own, num_bytes, memory), [bases.data_ptr(), 0, 0], [bases]

    def start(self, meta: torch.Tensor, ids: torch.Tensor, fetch) -> None:
        if n := ids.numel():
            out, words = fetch.rows, fetch.row_stride // 8
            _gather_kernel[(triton.cdiv(n * words, 1024),)](
                ids,
                meta,
                out,
                n,
                TABLES=meta.shape[0],
                K=ids.shape[-1],
                WORDS=words,
                BLOCK=1024,
            )

    @staticmethod
    def wait(fetch: None, table: int, out: torch.Tensor) -> None:
        """Rows are in place once `start` returns: SMs moved them."""

    @staticmethod
    def landed(fetch: None, table: int) -> None:
        """Rows are in place: see `wait`."""

    def close(self) -> None:
        pass
