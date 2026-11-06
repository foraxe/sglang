import socket
import threading
import unittest
from unittest import mock

from sglang.srt.engram_share.rdma import owner_control

from sglang.srt.engram_share.rdma.owner_control import (
    ControlError,
    OwnerClient,
    OwnerServer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class PeerRetireTests(CustomTestCase):
    def test_lost_replies_retry_without_duplicate_lease_or_retirement(self):
        retired = []
        with OwnerServer(
            {"checkpoint_id": "test"}, retire_peer=retired.append
        ).start() as server:
            client = OwnerClient(*server.address, "test", server.generation)
            original_send = owner_control._send
            main_thread = threading.current_thread()

            def lose_one_response():
                dropped = False

                def send(sock, payload):
                    nonlocal dropped
                    if threading.current_thread() is not main_thread and not dropped:
                        dropped = True
                        sock.shutdown(socket.SHUT_RDWR)
                        raise ConnectionResetError("lost reply after commit")
                    return original_send(sock, payload)

                return send

            with mock.patch.object(
                owner_control, "_send", side_effect=lose_one_response()
            ):
                lease = client.attach()
            self.assertEqual(client.status()["leases"], 1)
            peer = "shared-engram-" + "a" * 32
            with mock.patch.object(
                owner_control, "_send", side_effect=lose_one_response()
            ):
                lease.bind_peers([peer])
            with mock.patch.object(
                owner_control, "_send", side_effect=lose_one_response()
            ):
                lease.release()
            self.assertEqual(retired, [peer])
            self.assertEqual(client.status()["leases"], 0)

    def test_slow_retirement_does_not_block_other_leases(self):
        started, proceed = threading.Event(), threading.Event()
        calls, failures = [], []

        def retire(peer):
            calls.append(peer)
            started.set()
            if not proceed.wait(5):
                raise RuntimeError("test timed out")

        with OwnerServer(
            {"checkpoint_id": "test"}, retire_peer=retire
        ).start() as server:
            client = OwnerClient(*server.address, "test", server.generation, timeout=2)
            first = client.attach()
            first.bind_peers(["shared-engram-" + "b" * 32])

            def release():
                try:
                    first.release()
                except BaseException as exc:
                    failures.append(exc)

            worker = threading.Thread(target=release)
            worker.start()
            try:
                self.assertTrue(started.wait(2))
                other = client.attach()
                other.bind_peers([])
                other.release()
                self.assertEqual(client.status()["leases"], 1)
            finally:
                proceed.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(len(calls), 1)
            self.assertEqual(client.status()["leases"], 0)

    def test_generation_ledger_bounds_and_prevents_peer_reuse(self):
        with mock.patch.object(owner_control, "MAX_GENERATION_ATTACHES", 2):
            with OwnerServer(
                {"checkpoint_id": "test"}, retire_peer=lambda _: None
            ).start() as server:
                client = OwnerClient(*server.address, "test", server.generation)
                first = client.attach()
                peer = "shared-engram-" + "c" * 32
                first.bind_peers([peer])
                first.release()
                second = client.attach()
                with self.assertRaisesRegex(ControlError, "already bound"):
                    second.bind_peers([peer])
                second.bind_peers([])
                second.release()
                with self.assertRaisesRegex(ControlError, "attach limit"):
                    client.attach()

    def test_repeated_attach_after_drain_and_released_attach(self):
        with OwnerServer({"checkpoint_id": "test"}).start() as server:
            client = OwnerClient(*server.address, "test", server.generation)
            request_id = "request-id-1234567890"
            original = client._request(
                "attach", checkpoint_id="test", request_id=request_id
            )
            client.begin_drain()
            replay = client._request(
                "attach", checkpoint_id="test", request_id=request_id
            )
            self.assertEqual(original["lease"], replay["lease"])
            client._request("release", lease=original["lease"])
            with self.assertRaisesRegex(ControlError, "already released"):
                client._request("attach", checkpoint_id="test", request_id=request_id)

    def test_duplicate_release_does_not_repeat_backend_retirement(self):
        started, proceed = threading.Event(), threading.Event()
        calls, errors = [], []

        def retire(peer):
            calls.append(peer)
            started.set()
            if not proceed.wait(5):
                raise RuntimeError("test timed out")

        with OwnerServer(
            {"checkpoint_id": "test"}, retire_peer=retire
        ).start() as server:
            client = OwnerClient(*server.address, "test", server.generation, timeout=5)
            lease = client.attach()
            lease.bind_peers(["shared-engram-" + "d" * 32])

            def release():
                try:
                    client._request("release", lease=lease.token)
                except BaseException as exc:
                    errors.append(exc)

            first = threading.Thread(target=release)
            second = threading.Thread(target=release)
            first.start()
            try:
                self.assertTrue(started.wait(2))
                second.start()
            finally:
                proceed.set()
                first.join(5)
                if second.ident is not None:
                    second.join(5)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(calls), 1)
            self.assertEqual(client.status()["leases"], 0)

    def test_sixteen_prefetch_peers_and_one_graph_peer(self):
        retired = []
        with OwnerServer(
            {"checkpoint_id": "test"}, retire_peer=retired.append
        ).start() as server:
            client = OwnerClient(*server.address, "test", server.generation)
            lease = client.attach()
            peers = ["shared-engram-" + f"{i:032x}" for i in range(17)]
            lease.bind_peers(peers)
            lease.release()
            self.assertEqual(retired, peers)
            self.assertEqual(client.status()["leases"], 0)

    def test_bound_release_and_independent_readers(self):
        retired = []
        with OwnerServer(
            {"checkpoint_id": "test"}, retire_peer=retired.append
        ).start() as server:
            c = OwnerClient(*server.address, "test", server.generation)
            a, b = c.attach(), c.attach()
            pa = "shared-engram-" + ("a" * 32)
            pb = "shared-engram-" + ("b" * 32)
            a.bind_peers([pa])
            b.bind_peers([pb])
            a.release()
            self.assertEqual(retired, [pa])
            self.assertEqual(c.status()["leases"], 1)
            b.release()
            self.assertEqual(retired, [pa, pb])

    def test_failure_keeps_lease_and_retry_is_safe(self):
        calls = []

        def retire(peer):
            calls.append(peer)
            if len(calls) == 2:
                raise RuntimeError("synthetic failure")

        with OwnerServer(
            {"checkpoint_id": "test"}, retire_peer=retire
        ).start() as server:
            c = OwnerClient(*server.address, "test", server.generation)
            a = c.attach()
            peers = ["shared-engram-" + x * 32 for x in "ab"]
            a.bind_peers(peers)
            with self.assertRaisesRegex(ControlError, "lease retained"):
                a.release()
            self.assertEqual(c.status()["leases"], 1)
            a.release()
            self.assertEqual(c.status()["leases"], 0)
            self.assertEqual(calls, peers + peers)

    def test_unbound_or_duplicate_is_rejected(self):
        with OwnerServer(
            {"checkpoint_id": "test"}, retire_peer=lambda _: None
        ).start() as server:
            c = OwnerClient(*server.address, "test", server.generation)
            a, b = c.attach(), c.attach()
            with self.assertRaisesRegex(ControlError, "must bind"):
                a.release()
            p = "shared-engram-" + ("a" * 32)
            a.bind_peers([p])
            with self.assertRaisesRegex(ControlError, "already bound"):
                b.bind_peers([p])
            b.bind_peers([])
            a.release()
            b.release()


if __name__ == "__main__":
    unittest.main()
