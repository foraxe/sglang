"""CPU-only regression tests for POSIX-FD exchange timeout cleanup."""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
import threading
import types
from pathlib import Path

import pytest


def _load_cuda_vmm_utils(monkeypatch: pytest.MonkeyPatch):
    """Load the module without importing SGLang's optional runtime stack."""
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    utils = types.ModuleType("sglang.srt.utils")
    utils.log_info_on_rank0 = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)

    module_path = Path(__file__).parents[3] / "python/sglang/srt/cuda_vmm_utils.py"
    spec = importlib.util.spec_from_file_location(
        "_test_cuda_vmm_utils_fd_timeout", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fd_count() -> int:
    fd_dir = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    return len(os.listdir(fd_dir))


def test_exchange_posix_fds_joins_late_receiver_and_closes_fd(
    monkeypatch: pytest.MonkeyPatch,
):
    cuda_vmm_utils = _load_cuda_vmm_utils(monkeypatch)
    monkeypatch.setattr(cuda_vmm_utils, "_FD_SEND_TIMEOUT_S", 0.02)
    # macOS does not implement AF_UNIX/SOCK_SEQPACKET. A single SCM_RIGHTS
    # record over SOCK_STREAM exercises the same ownership and timeout path.
    monkeypatch.setattr(cuda_vmm_utils.socket, "SOCK_SEQPACKET", socket.SOCK_STREAM)

    baseline_fds = _fd_count()
    source_read, source_write = os.pipe()
    peer_path = f"/tmp/sgl_fd_peer_{os.getpid()}_{threading.get_ident()}.sock"
    peer_server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    peer_server.bind(peer_path)
    peer_server.listen(1)

    receiver_entered = threading.Event()
    release_late_receive = threading.Event()
    received_fd: list[int] = []
    receiver_thread: list[threading.Thread] = []
    peer_errors: list[BaseException] = []
    original_recv_fd = cuda_vmm_utils._recv_fd

    def delayed_recv_fd(conn):
        packet = original_recv_fd(conn)
        assert packet is not None
        received_fd.append(packet[2])
        receiver_thread.append(threading.current_thread())
        receiver_entered.set()
        assert release_late_receive.wait(2)
        return packet

    monkeypatch.setattr(cuda_vmm_utils, "_recv_fd", delayed_recv_fd)

    peer_worker: list[threading.Thread] = []

    def all_gather_object(paths, local_path, group=None):
        paths[:] = [local_path, peer_path]

        def act_as_peer():
            try:
                outgoing, _ = peer_server.accept()
                with outgoing:
                    with socket.socket(
                        socket.AF_UNIX, socket.SOCK_SEQPACKET
                    ) as incoming:
                        incoming.connect(local_path)
                        cuda_vmm_utils._send_fd(incoming, source_read, 1, 0)
                        assert receiver_entered.wait(2)
                        threading.Timer(0.08, release_late_receive.set).start()
            except BaseException as error:
                peer_errors.append(error)
                release_late_receive.set()

        worker = threading.Thread(target=act_as_peer, name="fd-timeout-peer")
        worker.start()
        peer_worker.append(worker)

    monkeypatch.setattr(cuda_vmm_utils.dist, "all_gather_object", all_gather_object)

    try:
        with pytest.raises(
            RuntimeError, match="timed out waiting for POSIX fd exchange"
        ):
            cuda_vmm_utils.exchange_posix_fds(
                object(),
                rank=0,
                world_size=2,
                local_fds=[source_read],
                peer_base_counts=[1, 1],
            )
        peer_worker[0].join(timeout=2)
        assert not peer_worker[0].is_alive()
        assert not peer_errors
        assert len(receiver_thread) == 1
        assert not receiver_thread[0].is_alive()
        assert len(received_fd) == 1
        with pytest.raises(OSError):
            os.fstat(received_fd[0])
    finally:
        release_late_receive.set()
        peer_server.close()
        os.unlink(peer_path)
        os.close(source_read)
        os.close(source_write)

    assert _fd_count() == baseline_fds
