"""Private hybrid adapter for the frozen ReadPlan; requires gather_graph C ABI."""

import base64
import ctypes as C
import json
import threading

import torch
from sglang.srt.engram_share.rdma import dedup_rows
from sglang.srt.engram_share.rdma.read_plan import ReadPlan


def _api(path):
    lib = C.CDLL(path)
    lib.gg_last_error.restype = C.c_char_p
    u64, u32, ptr = C.c_uint64, C.c_uint, C.c_void_p
    signatures = {
        "gg_owner_create": (
            [
                C.c_int,
                C.c_char_p,
                C.c_int,
                u32,
                C.POINTER(u64),
                C.POINTER(u64),
                u32,
                u32,
            ],
            ptr,
        ),
        "gg_owner_close": ([ptr], C.c_int),
        "gg_reader_create": ([C.c_int, C.c_char_p, C.c_int, u32, u64, u64, u32], ptr),
        "gg_reader_connect": ([ptr, C.c_char_p, u32], C.c_int),
        "gg_fetch": (
            [ptr, u64, u32, u64, u32, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64],
            C.c_int,
        ),
        "gg_record_consumed": ([ptr, u64], C.c_int),
        "gg_reader_check": ([ptr], C.c_int),
        "gg_reader_close": ([ptr], C.c_int),
    }
    for name, (args, result) in signatures.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, result
    return lib


def _check(lib, result):
    if result:
        raise RuntimeError((lib.gg_last_error() or b"gather_graph failure").decode())


def unpack_metadata(blob):
    meta = json.loads(blob)
    if meta.get("protocol") != "gather_graph_v1":
        raise ValueError("unexpected gather metadata protocol")
    return base64.b64decode(meta["native_md"]), meta


class GatherSessionFactory:
    """Allocate one single-reader endpoint per lease, over the same backing."""

    def __init__(self, library, gpu, device, gid, host, first_port, capacity, tables):
        if capacity < 1 or first_port < 1 or first_port + capacity - 1 > 65535:
            raise ValueError("invalid gather endpoint range")
        self.lib, self.gpu, self.device, self.gid = _api(library), gpu, device, gid
        self.host, self.first_port, self.capacity = host, first_port, capacity
        self.tables = tables
        self.lock = threading.Lock()
        self.occupied = set()

    def __call__(self):
        with self.lock:
            port = next(
                (
                    p
                    for p in range(self.first_port, self.first_port + self.capacity)
                    if p not in self.occupied
                ),
                None,
            )
            if port is None:
                raise RuntimeError("no gather endpoints available")
            self.occupied.add(port)
        addresses = (C.c_uint64 * len(self.tables))(
            *(t.data_ptr() for t in self.tables)
        )
        lengths = (C.c_uint64 * len(self.tables))(*(t.numel() for t in self.tables))
        handle = self.lib.gg_owner_create(
            self.gpu,
            self.device.encode(),
            self.gid,
            16,
            addresses,
            lengths,
            len(self.tables),
            port,
        )
        if not handle:
            # Keep this slot quarantined if setup failed after allocating resources.
            raise RuntimeError(
                (self.lib.gg_last_error() or b"owner create failed").decode()
            )
        factory = self

        class Session:
            metadata = {
                "protocol": "gather_graph_v1",
                "bulk_host": factory.host,
                "bulk_port": port,
            }

            def close(self):
                _check(factory.lib, factory.lib.gg_owner_close(handle))
                with factory.lock:
                    factory.occupied.remove(port)

        return Session()


class GatherReadPlan(ReadPlan):
    def __init__(
        self,
        library,
        bulk_library,
        gpu,
        port,
        lanes,
        device,
        gid,
        max_queries=196608,
        graph_sizes=(49152,),
        threshold=32768,
    ):
        if lanes != 16 or threshold < 4096:
            raise ValueError("gather requires 16 lanes and threshold >= 4096")
        super().__init__(library, gpu, port, lanes, max_queries, graph_sizes)
        self.lib, self.device, self.gid = _api(bulk_library), device, gid
        self.threshold, self.bulk = threshold, None
        self.native_used, self.bulk_used = set(), False

    def register(self, tensors, host):
        super().register(tensors, host)
        self.bulk = self.lib.gg_reader_create(
            self.gpu,
            self.device.encode(),
            self.gid,
            16,
            self.arena.data_ptr(),
            self.arena.numel(),
            self.maximum,
        )
        if not self.bulk:
            raise RuntimeError(
                (self.lib.gg_last_error() or b"reader create failed").decode()
            )

    def connect_raw(self, blob, addresses, lengths, remote_gpu):
        native, meta = unpack_metadata(blob)
        super().connect_raw(native, addresses, lengths, remote_gpu)
        _check(
            self.lib,
            self.lib.gg_reader_connect(
                self.bulk, meta["bulk_host"].encode(), meta["bulk_port"]
            ),
        )
        self.source_rows = [int(x) // 264 for x in lengths]
        self.native_stats = [
            torch.zeros((16, 8), dtype=torch.uint64, device="cuda") for _ in lengths
        ]
        self.bulk_stats = [
            torch.zeros((16, 8), dtype=torch.uint64, device="cuda") for _ in lengths
        ]
        for w in self.workspace:
            w["dense"] = torch.empty(self.maximum, dtype=torch.int64, device="cuda")
            w["inverse"] = torch.empty_like(w["dense"])

    def _enqueue(self, li, q, ids, payload, dump, window, lo, hi, dedup):
        if ids.numel() < self.threshold:
            return super()._enqueue(li, q, ids, payload, dump, window, lo, hi, dedup)
        w = self.workspace[li]
        n = ids.numel()
        dedup_rows.prepare(ids, lo, hi, w["scratch"], w["reps"][:n], w["work"][:n])
        _check(
            self.lib,
            self.lib.gg_fetch(
                self.bulk,
                ids.data_ptr(),
                n,
                payload,
                li,
                lo,
                hi,
                self.source_rows[li],
                w["reps"].data_ptr(),
                w["work"].data_ptr(),
                w["scratch"][-1:].data_ptr(),
                w["dense"].data_ptr(),
                w["inverse"].data_ptr(),
                self.stats.data_ptr(),
                torch.cuda.current_stream(self.gpu).cuda_stream,
            ),
        )

    def fetch(
        self,
        ids,
        count,
        payload_offset,
        scale_offset,
        dump_offset,
        window_rows,
        stats,
        weight_index=0,
        scale_index=0,
        row_lo=0,
        row_hi=2**63 - 1,
        row264=False,
    ):
        if ids.dtype != torch.int64 or not ids.is_cuda or not ids.is_contiguous():
            raise ValueError("invalid IDs: expected contiguous CUDA int64")
        li = weight_index
        if count >= self.threshold:
            self.bulk_used = True
        else:
            self.native_used.update(range(self.lanes if count >= 4096 else 1))
        self.stats = (
            self.bulk_stats[li] if count >= self.threshold else self.native_stats[li]
        )
        args = (
            ids,
            count,
            payload_offset,
            scale_offset,
            dump_offset,
            window_rows,
            stats,
            weight_index,
            scale_index,
            row_lo,
            row_hi,
            row264,
        )
        # External graph replays can bypass submission-time stream checks.
        # Keep the existing outer-CP-capture guard until graph ownership is wired.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("outer capture of gathered Engram is not qualified")
        result = super().fetch(*args)
        if count >= self.threshold:
            self.last_map[li] = self.workspace[li]["dense"][:count]
        return result

    def record_consumed(self):
        stream = torch.cuda.current_stream(self.gpu).cuda_stream
        if self.bulk_used:
            _check(self.lib, self.lib.gg_record_consumed(self.bulk, stream))
        for i in sorted(self.native_used):
            self.contexts[i].record_consumed()

    def close(self):
        torch.cuda.synchronize()
        if self.bulk:
            # Do not destroy graph references when a device or worker error
            # requires the entire reader to remain quarantined.
            _check(self.lib, self.lib.gg_reader_check(self.bulk))
        self.plans.clear()
        if self.bulk:
            _check(self.lib, self.lib.gg_reader_close(self.bulk))
            self.bulk = None
        super().close()
