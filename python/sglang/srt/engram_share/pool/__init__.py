# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/distributed/engram_pool at 36b40f524c (Apache-2.0).
"""Read-only tables split by rows over every GPU of a peer group.

The pool owns sharding, rendezvous and transfers; models own row formats.
Load each table's local shard, call ready, then lookup raw rows.

Transports: `nvlink` maps every GPU's HBM or EGM shard (cuMem FABRIC) and loads rows
directly; with `rdma`, the owners' NICs gather rows from host or HBM shards.
"""

import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from sglang.srt.engram_share.pool.rows import landed_row
from sglang.srt.utils.custom_op import register_custom_op

from .utils import cdiv, round_up

if TYPE_CHECKING:
    from sglang.srt.engram_share.pool.nvlink import NvlinkTransport
    from sglang.srt.engram_share.pool.rdma import RdmaTransport

logger = logging.getLogger(__name__)

__all__ = [
    "EngramFetch",
    "EngramPool",
    "EngramTable",
    "Landed",
    "get_peer_pool",
    "landed_row",
]

# Rows move 8 bytes at a time.
_ROW_ALIGN = 8


@dataclass(eq=False)
class EngramTable:
    """One table's rows, split evenly over the group's GPUs in rank order.
    `shard` is this GPU's rows from `row_start` (host memory for host shards)."""

    num_rows: int
    row_bytes: int
    row_stride: int
    row_start: int
    shard: torch.Tensor
    _meta: torch.Tensor
    _keep: list = field(default_factory=list)
    _loaded: bool = False
    _token: int = -1


class Landed(NamedTuple):
    """`landed_row`'s view of one table of a fetch."""

    rows: torch.Tensor
    slots: torch.Tensor
    ready: torch.Tensor
    placed: bool


class EngramFetch:
    """A started fetch: `wait(i)` places table i's rows, `landed(i)` finds them."""

    def __init__(
        self,
        pool: "EngramPool",
        shape: tuple[int, ...],
        out: torch.Tensor | None = None,
    ) -> None:
        self.pool, self.shape, self.row_stride = pool, shape, shape[-1]
        self._out = out
        self.state: Any = None

    @cached_property
    def rows(self) -> torch.Tensor:
        """Placed rows, `[L, ..., K, row_stride]` uint8."""
        if self._out is not None:
            return self._out
        return self.pool._output(math.prod(self.shape)).view(self.shape)

    def wait(self, index: int) -> torch.Tensor:
        """Table `index`'s rows; the current stream waits for them."""
        self.pool.transport.wait(self.state, index, self.rows[index])
        return self.rows[index]

    def landed(self, index: int) -> Landed:
        """Table `index`'s rows where they land, for a kernel that reads them
        with `landed_row` instead of `wait`. Launches nothing."""
        got = self.pool.transport.landed(self.state, index)
        if got is None:
            rows = self.rows[index]
            return Landed(rows, rows, rows, True)
        rows, slots, ready = got
        return Landed(rows, slots, ready, False)


class EngramPool:
    """Tables shared by a peer group, and the buffers fetches land in.

    Every GPU of the group creates the same pool and tables in the same order
    (each call is collective). A fetch's rows stay valid until the next starts.

    Args:
        group: The peer group splitting the tables.
        transport: "nvlink" or "rdma".
        placement: "gpu", "egm" (NVLink), or "cpu" (RDMA).
        max_items: Most rows of one table one fetch gathers on this GPU.
        max_row_bytes: Largest row of any table.
        max_tables: Most tables one fetch reads.
        rdma_inflight: Most prefill-sized gathers each NIC keeps in flight
            (RDMA; 0: no limit), so they delay others' decode fetches little.

    """

    def __init__(
        self,
        group: Any,
        transport: str,
        placement: str,
        max_items: int,
        max_row_bytes: int,
        max_tables: int = 1,
        rdma_inflight: int = 8,
        fetch_mode: str = "sync",
    ) -> None:
        if min(max_items, max_row_bytes, max_tables) <= 0:
            raise ValueError("Pool capacities must be positive")
        if (
            placement not in ("gpu", "egm", "cpu")
            or (placement == "egm" and transport != "nvlink")
            or (placement == "cpu" and transport != "rdma")
        ):
            raise ValueError(
                f"Unsupported transport/placement: {transport}/{placement}"
            )
        host_shards = placement != "gpu"
        group = _GroupAdapter(group)
        if transport not in ("nvlink", "rdma"):
            raise ValueError(f"Unknown Engram pool transport {transport!r}")
        self._check_same(group, ("pool", transport, host_shards))
        self.group, self.fetch_mode = group, fetch_mode
        self._transport_name = transport
        self._tokens: list[int] = []
        self.max_row_bytes = round_up(max_row_bytes, _ROW_ALIGN)
        self.max_items, self.max_tables = max_items, max_tables
        self.transport: NvlinkTransport | RdmaTransport
        if transport == "nvlink":
            from sglang.srt.engram_share.pool import nvlink

            self.transport = nvlink.NvlinkTransport(group, host_shards)
        else:
            from sglang.srt.engram_share.pool import rdma

            self.transport = rdma.RdmaTransport(
                group,
                host_shards,
                max_items,
                self.max_row_bytes,
                max_tables,
                rdma_inflight,
            )
        if host_shards and transport == "nvlink":
            self._where = "EGM"
        else:
            self._where = "host memory" if host_shards else "HBM"
        self._buffer: torch.Tensor | None = None
        self._metas: dict[tuple[int, ...], torch.Tensor] = {}

    @staticmethod
    def _check_same(group: Any, identity: tuple) -> None:
        """Fail startup on every GPU unless all of them passed `identity`."""
        mine = json.dumps(identity)
        everyone = [v.decode() for v in group.all_gather(mine.encode())]
        if differ := [r for r, theirs in enumerate(everyone) if theirs != mine]:
            raise RuntimeError(
                f"Engram pool GPUs {differ} disagree with GPU "
                f"{group.rank_in_group}: {[everyone[r] for r in differ]} vs {identity}"
            )

    def table(self, num_rows: int, row_bytes: int, name: str = "") -> EngramTable:
        """Create a table of `num_rows` rows of `row_bytes` bytes. Collective;
        `name` identifies it across the group."""
        if num_rows <= 0 or row_bytes <= 0:
            raise ValueError("Table dimensions must be positive")
        row_stride = round_up(row_bytes, _ROW_ALIGN)
        if row_stride > self.max_row_bytes:
            raise ValueError(f"Rows of {row_bytes} bytes exceed the pool's rows")
        self._check_same(self.group, ("table", name, num_rows, row_bytes))
        rank, world = self.group.rank_in_group, self.group.world_size
        rows_per_shard = cdiv(num_rows, world)
        row_start = min(rank * rows_per_shard, num_rows)
        shard_rows = min(rows_per_shard, num_rows - row_start)
        num_bytes = max(rows_per_shard * row_stride, _ROW_ALIGN)
        local, arrays, keep = self.transport.shard(num_bytes, name)
        meta = torch.tensor([rows_per_shard, num_rows, *arrays], dtype=torch.int64)
        shard = local[: shard_rows * row_stride].view(shard_rows, row_stride)
        logger.info(
            "Engram pool table %r: rows [%d, %d) of %d on this GPU, %.2f GiB in %s",
            name,
            row_start,
            row_start + shard_rows,
            num_rows,
            num_bytes / 2**30,
            self._where,
        )
        table = EngramTable(
            num_rows, row_bytes, row_stride, row_start, shard[:, :row_bytes], meta, keep
        )
        if self._transport_name == "rdma":
            table._token = next(_table_tokens)
            _tables[table._token] = (self, table)
            self._tokens.append(table._token)
        return table

    def ready(self, table: EngramTable) -> None:
        """Serve `table` once every GPU has loaded its shard. Peers read it at
        any time from then on, so it cannot be reloaded."""
        if table._loaded:
            raise RuntimeError(
                "Engram tables split across engines (peer_group_size) cannot "
                "reload weights; restart the peer group instead"
            )
        torch.accelerator.synchronize()
        self.group.barrier()
        table._loaded = True

    def _meta(self, tables: list[EngramTable]) -> torch.Tensor:
        key = tuple(id(t) for t in tables)
        if (meta := self._metas.get(key)) is None:
            if len({t.row_stride for t in tables}) != 1:
                raise ValueError("One fetch's tables must have equal row sizes")
            meta = torch.stack([t._meta for t in tables]).cuda()
            self._metas[key] = meta
        return meta

    def _output(self, numel: int) -> torch.Tensor:
        """Placed rows, allocated once a fetch places any."""
        if self._buffer is None:
            numel_max = self.max_items * self.max_tables * self.max_row_bytes
            self._buffer = torch.empty(numel_max, dtype=torch.uint8, device="cuda")
        return self._buffer[:numel]

    def start(
        self,
        tables: list[EngramTable],
        ids: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> EngramFetch:
        """Start fetching `ids` (`[..., L, K]`: rows of table l at `[..., l, :]`)
        on the current stream; ids < 0 or past a table give zero rows."""
        if (
            not tables
            or len(tables) > self.max_tables
            or ids.numel() > self.max_items * len(tables)
        ):
            raise ValueError("Fetch exceeds pool capacity or has no tables")
        if any(not table._loaded for table in tables):
            raise RuntimeError("Engram table is not ready")
        if ids.shape[-2] != len(tables):
            raise ValueError(
                f"ids {tuple(ids.shape)} do not match {len(tables)} tables"
            )
        meta = self._meta(tables)
        shape = (len(tables), *ids.shape[:-2], ids.shape[-1], tables[0].row_stride)
        if out is not None and (
            out.shape != shape
            or out.dtype != torch.uint8
            or out.device != ids.device
            or not out.is_contiguous()
        ):
            raise ValueError(f"Engram pool output must be contiguous uint8 of {shape}")
        fetch = EngramFetch(self, shape, out)
        fetch.state = self.transport.start(meta, ids.contiguous(), fetch)
        return fetch

    def lookup(
        self,
        tables: list[EngramTable],
        ids: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rows `[L, ..., K, row_stride]` (uint8) of `ids`, as `start` takes."""
        fetch = self.start(tables, ids, out=out)
        for index in range(len(tables)):
            fetch.wait(index)
        return fetch.rows

    def lookup_rows(self, table: EngramTable, ids: torch.Tensor) -> torch.Tensor:
        """Look up one table, returning padded uint8 rows shaped like ids."""
        if ids.ndim == 0:
            raise ValueError("ids must have at least one dimension")
        if self._transport_name == "rdma":
            out = ids.new_empty((*ids.shape, table.row_stride), dtype=torch.uint8)
            _rdma_lookup_rows(ids, out, table._token)
            return out
        return self.lookup([table], ids.unsqueeze(-2))[0]

    def close(self) -> None:
        self.transport.close()
        for token in self._tokens:
            _tables.pop(token, None)
        self._tokens.clear()


@cache
def get_peer_pool(
    group,
    transport: str,
    placement: str,
    max_items: int,
    max_row_bytes: int,
    max_tables: int = 1,
) -> EngramPool:
    return EngramPool(group, transport, placement, max_items, max_row_bytes, max_tables)


class _GroupAdapter:
    """Bridge the startup group's JSON exchange to the byte rendezvous."""

    def __init__(self, group):
        self.group = group
        self.rank_in_group = group.rank_in_group
        self.world_size = group.world_size

    def all_gather(self, value: bytes) -> list[bytes]:
        return [
            bytes.fromhex(item)
            for item in self.group.exchange("pool_exchange", value.hex())
        ]

    def barrier(self):
        self.all_gather(b"")


# Tokens are process-local and fixed before model compilation.  The mutable
# output makes the operation visible to Dynamo without tracing NIC state.
_table_tokens = itertools.count()
_tables: dict[int, tuple[EngramPool, EngramTable]] = {}


@register_custom_op(mutates_args=["out"])
def _rdma_lookup_rows(ids: torch.Tensor, out: torch.Tensor, token: int) -> None:
    pool, table = _tables[token]
    pool.lookup([table], ids.unsqueeze(-2), out=out.unsqueeze(0))
