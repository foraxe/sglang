"""Experimental same-node consumption of the owner's row264 backing."""

import json
import mmap
import os
import signal
from pathlib import Path

import torch

from sglang.srt.engram_share.rdma.owner_control import OwnerClient

_runtime = None
_quarantined = []


def close_runtime():
    """Close the local reader after the scheduler has stopped launching work."""
    if _runtime is not None:
        _runtime.close()


class LocalRuntime:
    def __init__(self, model_path):
        from sglang.srt.engram_share.rdma.owner import inspect_checkpoint

        checkpoint = inspect_checkpoint(Path(model_path))
        ready = json.loads(Path(os.environ["ENGRAM_SHARED_READY"]).read_text())
        if checkpoint["checkpoint_identity"] != ready["checkpoint_id"]:
            raise ValueError("shared owner/serving checkpoint identity mismatch")
        self.lease = OwnerClient(
            ready["host"], ready["port"], ready["checkpoint_id"], ready["generation"]
        ).attach()
        self.fd = None
        self.mapping = None
        self.bytes = None
        self.registered = False
        self.closed = False
        self.peer_retirement = False
        self._peers_bound = False
        try:
            metadata = self.lease.metadata
            local = metadata.get("local_backing")
            self.peer_retirement = bool(metadata.get("peer_retirement"))
            if self.peer_retirement:
                self.lease.bind_peers([])
                self._peers_bound = True
            if metadata.get("storage_layout") != "row264" or not local:
                raise ValueError("owner did not publish a local row264 backing")
            if int(local["pid_namespace"]) != os.stat("/proc/self/ns/pid").st_ino:
                raise ValueError("local Engram owner must share the PID namespace")
            pid, fd = int(local["pid"]), int(local["fd"])
            ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            if ticks != local["start_ticks"]:
                raise ValueError("shared Engram owner process changed")
            self.fd = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR)
            stat = os.fstat(self.fd)
            if stat.st_ino != int(local["inode"]) or stat.st_size != int(
                local["bytes"]
            ):
                raise ValueError("shared Engram backing identity mismatch")
            self.mapping = mmap.mmap(self.fd, stat.st_size, flags=mmap.MAP_SHARED)
            self.bytes = torch.frombuffer(self.mapping, dtype=torch.uint8)
            err = torch.cuda.cudart().cudaHostRegister(
                self.bytes.data_ptr(), stat.st_size, 0
            )
            if int(err) != 0:
                raise RuntimeError(f"local Engram cudaHostRegister failed: {err}")
            self.registered = True
            self.tables = {
                int(table["layer"]): table for table in metadata["table_specs"]
            }
            if set(self.tables) != {1, 14}:
                raise ValueError("unexpected local Engram layers")
            end = 0
            for layer in (1, 14):
                table = self.tables[layer]
                rows, offset, nbytes = map(
                    int, (table["rows"], table["offset"], table["nbytes"])
                )
                if (
                    rows <= 0
                    or offset != end
                    or nbytes != rows * 264
                    or offset + nbytes > stat.st_size
                ):
                    raise ValueError("invalid local Engram table layout")
                end = offset + nbytes
            if end != stat.st_size:
                raise ValueError("local Engram backing has unexpected size")
            if os.environ.get("ENGRAM_SHARED_CONTROLLED_DRAIN") == "1":
                if signal.getsignal(signal.SIGUSR2) != signal.SIG_DFL:
                    raise RuntimeError(
                        "refuse to overwrite an existing SIGUSR2 handler"
                    )
                signal.signal(signal.SIGUSR2, self._controlled_drain)
        except BaseException:
            try:
                if self.peer_retirement and not self._peers_bound:
                    self.lease.bind_peers([])
                self.close()
            except BaseException:
                _quarantined.append(self)
            raise

    def close(self):
        if self.closed:
            return
        if self.registered:
            torch.cuda.synchronize()
            err = torch.cuda.cudart().cudaHostUnregister(self.bytes.data_ptr())
            if int(err) != 0:
                raise RuntimeError(f"local Engram cudaHostUnregister failed: {err}")
            self.registered = False
        self.bytes = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.lease.release()
        self.closed = True

    def _controlled_drain(self, *_):
        self.close()
        print(
            json.dumps({"event": "local-engram-reader-drained", "pid": os.getpid()}),
            flush=True,
        )


class LocalTable:
    def __init__(self, num_embeddings, dim, layer_id, model_path):
        if dim != 256 or layer_id not in (1, 14):
            raise ValueError("local shared Engram supports 256-wide layers 1 and 14")
        self.num_embeddings, self.dim, self.layer_id = num_embeddings, dim, layer_id
        self.model_path = model_path
        self.loaded = set()
        self.runtime = None

    def check_weight(self, param, loaded_weight, weight):
        kind = "weight" if param is weight else "scale"
        width = self.dim if kind == "weight" else self.dim // 32
        if tuple(loaded_weight.shape) != (self.num_embeddings, width):
            raise ValueError("local Engram checkpoint tensor shape mismatch")
        if loaded_weight.dtype != param.dtype:
            raise ValueError("local Engram checkpoint tensor dtype mismatch")
        self.loaded.add(kind)

    def finish_load(self, row_start, rows):
        global _runtime
        if self.loaded != {"weight", "scale"}:
            raise RuntimeError("local Engram checkpoint parameters not loaded")
        if _runtime is None:
            _runtime = LocalRuntime(self.model_path)
        self.runtime = _runtime
        table = self.runtime.tables[self.layer_id]
        if (
            int(table["rows"]) != self.num_embeddings
            or int(table["nbytes"]) != self.num_embeddings * 264
        ):
            raise ValueError("local Engram table size mismatch")

    def lookup(self, indices, new_rows):
        if self.runtime is None or self.runtime.closed:
            raise RuntimeError("local Engram backing is unavailable")
        from sglang.srt.engram_share.rdma.row264_kernels import gather_row264

        output = new_rows(indices)
        ids = indices.reshape(-1).contiguous()
        offset = int(self.runtime.tables[self.layer_id]["offset"])
        gather_row264(
            self.runtime.bytes.data_ptr() + offset,
            ids,
            output.view(-1, self.dim),
            0,
            self.num_embeddings,
        )
        return output
