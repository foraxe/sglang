import unittest

from sglang.srt.engram_share.rdma.owner_control import (
    ControlError,
    OwnerClient,
    OwnerServer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class PeerRetireTests(CustomTestCase):
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
