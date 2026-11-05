"""Check shared memfd pages for local GPU lookup and GPUNETIO READ on H20.

Run in one PID namespace with two visible CUDA GPUs and the ABI-3 research
shim. This small probe does not start a model service.
"""

import json
import mmap
import os
import subprocess
import sys

import torch

from sglang.srt.engram_share.rdma.engram_transport import Context
from sglang.srt.engram_share.rdma.row264_kernels import gather_row264


def child(pid, fd, nbytes):
    torch.cuda.set_device(1)
    with open(f"/proc/{pid}/fd/{fd}", "r+b", buffering=0) as stream:
        mm = mmap.mmap(stream.fileno(), nbytes, flags=mmap.MAP_SHARED)
        data = torch.frombuffer(mm, dtype=torch.uint8)
        err = torch.cuda.cudart().cudaHostRegister(data.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"child cudaHostRegister failed: {err}")
        try:
            ids = torch.tensor([3, 0, 2, 1], device="cuda", dtype=torch.int64)
            out = torch.empty((4, 256), dtype=torch.bfloat16, device="cuda")
            gather_row264(data.data_ptr(), ids, out, 0, 4)
            torch.cuda.synchronize()
            expected = (
                torch.tensor([4, 1, 3, 2], dtype=torch.bfloat16)
                .view(4, 1)
                .expand(4, 256)
            )
            assert torch.equal(out.cpu(), expected)
            print(
                json.dumps(
                    {"local_gpu": "PASS", "inode": os.fstat(stream.fileno()).st_ino}
                )
            )
        finally:
            torch.cuda.synchronize()
            err = torch.cuda.cudart().cudaHostUnregister(data.data_ptr())
            if int(err) != 0:
                raise RuntimeError(f"child cudaHostUnregister failed: {err}")
            del data
            mm.close()


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "child":
        child(*map(int, sys.argv[2:]))
        return
    torch.cuda.set_device(0)
    nbytes = 4 * 264
    fd = os.memfd_create("sglang-engram-shared-probe", 0)
    os.ftruncate(fd, nbytes)
    mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED)
    data = torch.frombuffer(mm, dtype=torch.uint8)
    rows = data.view(4, 264)
    for row in range(4):
        value = torch.full((256,), row + 1, dtype=torch.float32).to(torch.float8_e4m3fn)
        rows[row, :256].copy_(value.view(torch.uint8))
        rows[row, 256:].fill_(127)
    err = torch.cuda.cudart().cudaHostRegister(data.data_ptr(), nbytes, 0)
    if int(err) != 0:
        raise RuntimeError(f"owner cudaHostRegister failed: {err}")
    owner = reader = None
    try:
        assert data.is_pinned(), "registered memfd must be accepted as pinned Host"
        owner = Context(
            os.environ["ENGRAM_SHARED_LIBRARY"], 0, int(os.environ["OWNER_PORT"])
        )
        owner.register([data], host=True)
        child_out = subprocess.check_output(
            [sys.executable, __file__, "child", str(os.getpid()), str(fd), str(nbytes)],
            text=True,
        )
        result = json.loads(child_out.strip().splitlines()[-1])
        assert result == {"local_gpu": "PASS", "inode": os.fstat(fd).st_ino}
        reader = Context(
            os.environ["ENGRAM_SHARED_LIBRARY"], 0, int(os.environ["READER_PORT"])
        )
        arena = torch.empty(4096, dtype=torch.uint8, device="cuda")
        reader.register([arena], host=False)
        reader.connect_raw(
            owner.metadata(), list(owner.addresses), list(owner.lengths), 0
        )
        ids = torch.tensor([3, 0, 2, 1], device="cuda", dtype=torch.int64)
        stats = torch.zeros(8, dtype=torch.uint64, device="cuda")
        reader.fetch_row264(ids, 64, 2048, 0, stats, 0, 4)
        reader.record_consumed()
        torch.cuda.synchronize()
        assert int(stats.cpu()[0]) == 0
        expected = torch.stack([rows[i] for i in (3, 0, 2, 1)])
        assert torch.equal(arena[64 : 64 + nbytes].cpu().view(4, 264), expected)
        print("PASS same memfd inode: local GPU gather and GPUNETIO READ", flush=True)
    finally:
        if reader is not None:
            reader.close()
        if owner is not None:
            owner.close()
        err = torch.cuda.cudart().cudaHostUnregister(data.data_ptr())
        if int(err) != 0:
            raise RuntimeError(f"owner cudaHostUnregister failed: {err}")
        del rows, data
        mm.close()
        os.close(fd)


if __name__ == "__main__":
    main()
