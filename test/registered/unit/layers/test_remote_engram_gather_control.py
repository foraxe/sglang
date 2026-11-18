import threading
import unittest
import socket
from unittest import mock

from sglang.srt.engram_share.rdma import owner_control
from sglang.srt.engram_share.rdma.owner_control import (
    ControlError,
    OwnerClient,
    OwnerServer,
)


class Session:
    def __init__(self, number, fail=False):
        self.metadata = {"port": 30000 + number}
        self.fail = fail
        self.closed = False

    def close(self):
        if self.fail:
            raise RuntimeError("uncertain teardown")
        self.closed = True


class GatherControlTests(unittest.TestCase):
    def test_factory_reuses_table_backing_but_not_endpoints(self):
        from sglang.srt.engram_share.rdma import gather

        class Table:
            def __init__(self, address, size):
                self.address, self.size = address, size

            def data_ptr(self):
                return self.address

            def numel(self):
                return self.size

        class Library:
            def __init__(self):
                self.calls, self.closed = [], []

            def gg_owner_create(
                self, gpu, dev, gid, qps, addresses, lengths, count, port
            ):
                self.calls.append((list(addresses), list(lengths), count, port))
                return port

            def gg_owner_close(self, handle):
                self.closed.append(handle)
                return 0

        lib = Library()
        with mock.patch.object(gather, "_api", return_value=lib):
            factory = gather.GatherSessionFactory(
                "unused",
                0,
                "device",
                3,
                "host",
                31000,
                2,
                [Table(1000, 2640), Table(5000, 5280)],
            )
            first, second = factory(), factory()
            self.assertNotEqual(
                first.metadata["bulk_port"], second.metadata["bulk_port"]
            )
            self.assertEqual(lib.calls[0][:3], lib.calls[1][:3])
            self.assertEqual(lib.calls[0][0], [1000, 5000])
            first.close()
            self.assertEqual(lib.closed, [31000])
            replacement = factory()
            self.assertEqual(replacement.metadata["bulk_port"], 31000)
            second.close()
            replacement.close()

    def test_lost_response_reuses_session_and_enforces_capacity(self):
        sessions = []

        def factory():
            session = Session(len(sessions))
            sessions.append(session)
            return session

        with OwnerServer(
            {"checkpoint_id": "test"}, gather_factory=factory, max_gather_sessions=1
        ).start() as server:
            client = OwnerClient(*server.address, "test", server.generation)
            first, second = client.attach(), client.attach()
            original_send = owner_control._send
            dropped = False

            def drop_response(sock, payload):
                nonlocal dropped
                if isinstance(payload, dict) and "gather" in payload and not dropped:
                    dropped = True
                    sock.shutdown(socket.SHUT_RDWR)
                    raise ConnectionResetError("lost response")
                return original_send(sock, payload)

            with mock.patch.object(owner_control, "_send", side_effect=drop_response):
                metadata = first.prepare_gather()
            self.assertTrue(dropped)
            self.assertEqual(metadata, sessions[0].metadata)
            self.assertEqual(len(sessions), 1)
            with self.assertRaisesRegex(ControlError, "capacity"):
                second.prepare_gather()
            first.release()
            second.prepare_gather()
            second.release()

    def test_independent_leases_and_idempotent_prepare(self):
        sessions = []

        def factory():
            session = Session(len(sessions))
            sessions.append(session)
            return session

        with OwnerServer(
            {"checkpoint_id": "test"}, gather_factory=factory, max_gather_sessions=2
        ).start() as server:
            client = OwnerClient(*server.address, "test", server.generation)
            one, two = client.attach(), client.attach()
            first = one.prepare_gather()
            self.assertEqual(one.prepare_gather(), first)
            self.assertNotEqual(two.prepare_gather(), first)
            self.assertEqual(len(sessions), 2)
            one.release()
            self.assertTrue(sessions[0].closed)
            self.assertFalse(sessions[1].closed)
            two.release()
            self.assertTrue(sessions[1].closed)

    def test_failed_close_keeps_only_affected_lease(self):
        sessions = []

        def factory():
            session = Session(len(sessions), fail=not sessions)
            sessions.append(session)
            return session

        server = OwnerServer(
            {"checkpoint_id": "test"}, gather_factory=factory, max_gather_sessions=2
        ).start()
        try:
            client = OwnerClient(*server.address, "test", server.generation)
            one, two = client.attach(), client.attach()
            one.prepare_gather()
            two.prepare_gather()
            with self.assertRaises(ControlError):
                one.release()
            two.release()
            self.assertEqual(client.status()["leases"], 1)
            self.assertFalse(sessions[0].closed)
            with self.assertRaises(ControlError):
                one.release()
        finally:
            server._server.shutdown()
            server._server.server_close()
            server._thread.join()

    def test_prepare_release_race_retains_live_session(self):
        started, proceed = threading.Event(), threading.Event()
        session = Session(0)

        def factory():
            started.set()
            if not proceed.wait(5):
                raise RuntimeError("test timed out")
            return session

        with OwnerServer(
            {"checkpoint_id": "test"}, gather_factory=factory, max_gather_sessions=1
        ).start() as server:
            client = OwnerClient(*server.address, "test", server.generation)
            lease = client.attach()
            errors = []

            def prepare():
                try:
                    lease.prepare_gather()
                except Exception as exc:
                    errors.append(exc)

            worker = threading.Thread(target=prepare)
            worker.start()
            self.assertTrue(started.wait(2))
            # A separate request may wait, but must not release during creation.
            closer = threading.Thread(target=lease.release)
            closer.start()
            self.assertEqual(client.status()["leases"], 1)
            proceed.set()
            worker.join(5)
            closer.join(5)
            self.assertFalse(worker.is_alive())
            self.assertFalse(closer.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
