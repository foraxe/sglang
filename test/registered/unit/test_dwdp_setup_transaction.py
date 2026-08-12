import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


_MANAGER_PATH = (
    Path(__file__).parents[3] / "python/sglang/srt/layers/moe/dwdp/dwdp_manager.py"
)


class _Layout:
    def __init__(self, num_routed_experts, dwdp_size, dwdp_rank):
        self.num_routed_experts = num_routed_experts
        self.dwdp_size = dwdp_size
        self.dwdp_rank = dwdp_rank
        self.num_experts_per_worker = num_routed_experts // dwdp_size
        self.num_prefetch_experts = self.num_experts_per_worker
        self.local_expert_start = dwdp_rank * self.num_experts_per_worker
        self.local_expert_end = self.local_expert_start + self.num_experts_per_worker
        self.peer_ranges = [(0, 2), (2, 4)]


class _FusedMoE:
    pass


def _load_manager_module():
    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.layers": types.ModuleType("sglang.srt.layers"),
        "sglang.srt.layers.moe": types.ModuleType("sglang.srt.layers.moe"),
        "sglang.srt.layers.moe.dwdp": types.ModuleType("sglang.srt.layers.moe.dwdp"),
    }
    layout = types.ModuleType("sglang.srt.layers.moe.dwdp.layout")
    layout.DwdpExpertLayout = _Layout
    layout.build_layer_weight_specs = lambda local_params, num_routed: {}
    layout.lookup_owner = lambda expert, ranges: 0
    modules[layout.__name__] = layout

    backends = types.ModuleType("sglang.srt.layers.moe.dwdp.backends")
    backends.get_dwdp_backend = lambda backend: (None, None)
    modules[backends.__name__] = backends

    weight_manager = types.ModuleType("sglang.srt.layers.moe.dwdp.weight_manager")
    weight_manager.DWDPWeightManager = object
    modules[weight_manager.__name__] = weight_manager

    layer = types.ModuleType("sglang.srt.layers.moe.fused_moe_triton.layer")
    layer.FusedMoE = _FusedMoE
    modules[layer.__name__] = layer

    runtime = types.ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: None
    modules[runtime.__name__] = runtime

    spec = importlib.util.spec_from_file_location(
        "dwdp_manager_transaction_under_test", _MANAGER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


_MODULE = _load_manager_module()
DwdpManager = _MODULE.DwdpManager


class _Group:
    device_group = object()


class _Parallel:
    tp_group = _Group()


_EVENTS = []


class _Transport:
    instances = []
    fail_create = False
    fail_commit = False
    fail_release_once = False

    def __init__(self):
        self.handle_set = object()
        self.peer_views = {}
        self.release_count = 0
        self.commit_count = 0
        self.__class__.instances.append(self)

    @classmethod
    def create(cls, **kwargs):
        transport = cls()
        if cls.fail_create:
            # create() owns the object until it returns it to setup().
            transport.release()
            raise RuntimeError("injected transport creation")
        return transport

    def commit(self):
        self.commit_count += 1
        if self.fail_commit:
            raise RuntimeError("injected transport commit")

    def release(self):
        self.release_count += 1
        _EVENTS.append("transport-release")
        if self.fail_release_once:
            self.__class__.fail_release_once = False
            raise RuntimeError("injected transport release")


class _Buffer:
    instances = []
    fail_create = False
    fail_release_once = False

    def __init__(self):
        self.release_count = 0
        self.__class__.instances.append(self)

    @classmethod
    def create(cls, **kwargs):
        if cls.fail_create:
            raise RuntimeError("injected weight buffer creation")
        return cls()

    def get_full_tensor(self, layer_idx, name):
        value = 100 + layer_idx * 10 + (name == "w2_weight")
        return torch.full((4, 2), value, dtype=torch.int64)

    def weight_names(self, layer_idx):
        return ("w13_weight", "w2_weight")

    def release(self):
        self.release_count += 1
        _EVENTS.append("buffer-release")
        if self.fail_release_once:
            self.__class__.fail_release_once = False
            raise RuntimeError("injected weight buffer release")


class _WeightManager:
    instances = []
    fail_create = False

    def __init__(self, weight_buffer, transport, peer_views, **kwargs):
        if self.fail_create:
            raise RuntimeError("injected manager creation")
        self._weight_buffer = weight_buffer
        self._transport = transport
        self.peer_views = peer_views
        self.release_count = 0
        self.__class__.instances.append(self)

    def release(self):
        self.release_count += 1
        errors = []
        if self._weight_buffer is not None:
            try:
                self._weight_buffer.release()
            except BaseException as error:
                errors.append(error)
            else:
                self._weight_buffer = None
        if self._transport is not None:
            try:
                self._transport.release()
            except BaseException as error:
                errors.append(error)
            else:
                self._transport = None
        if self._weight_buffer is None and self._transport is None:
            self.peer_views.clear()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise RuntimeError("multiple fake resource releases failed") from errors[0]


class _Experts(_FusedMoE):
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.num_global_routed_experts = 4
        self.w13_weight = torch.full((2, 2), layer_idx + 1, dtype=torch.int64)
        self.w2_weight = torch.full((2, 2), layer_idx + 11, dtype=torch.int64)
        self.scale = torch.full((2, 1), layer_idx + 21, dtype=torch.int64)
        self.moe_ep_size = 2
        self.moe_ep_rank = 1
        self._num_local_routed = 2
        self.num_local_experts = 2
        self.runner_num_local_experts = 2
        self.dispatcher_moe_ep_size = 2
        self.dispatcher_moe_ep_rank = 1
        self._restore = None
        self.fail_bind = False
        self.fail_validate = False

    def bind_full_expert_weights(self, weights):
        self._restore = {
            "w13_weight": self.w13_weight,
            "w2_weight": self.w2_weight,
            "moe_ep_size": self.moe_ep_size,
            "moe_ep_rank": self.moe_ep_rank,
            "_num_local_routed": self._num_local_routed,
            "num_local_experts": self.num_local_experts,
            "runner_num_local_experts": self.runner_num_local_experts,
            "dispatcher_moe_ep_size": self.dispatcher_moe_ep_size,
            "dispatcher_moe_ep_rank": self.dispatcher_moe_ep_rank,
        }
        self.moe_ep_size = 1
        self.moe_ep_rank = 0
        self._num_local_routed = 4
        self.num_local_experts = 4
        self.runner_num_local_experts = 4
        self.dispatcher_moe_ep_size = 1
        self.dispatcher_moe_ep_rank = 0
        self.w13_weight = weights["w13_weight"]
        self.w2_weight = weights["w2_weight"]
        if self.fail_bind:
            raise RuntimeError("injected mid-bind")

    def unbind_full_expert_weights(self, *, restore=False):
        if self._restore is None:
            return
        if restore:
            for name, value in self._restore.items():
                setattr(self, name, value)
        else:
            self.w13_weight = torch.empty(0, dtype=self.w13_weight.dtype)
            self.w2_weight = torch.empty(0, dtype=self.w2_weight.dtype)
        self._restore = None

    def validate_full_expert_weights_commit(self):
        if self.fail_validate:
            raise RuntimeError("injected validation")
        if self._restore is None:
            raise RuntimeError("not rollback-capable")

    def commit_full_expert_weights(self):
        self._restore["w13_weight"] = None
        self._restore["w2_weight"] = None

    def named_per_expert_tensors(self, local_experts):
        return [("scale", self.scale)]

    def replace_expert_tensor(self, name, tensor):
        setattr(self, name, tensor)


class _Layer:
    def __init__(self, experts):
        self.experts = experts

    def modules(self):
        return [self, self.experts]


class _Model:
    def __init__(self):
        self.layers = [_Layer(_Experts(0)), _Layer(_Experts(1))]


class TestDwdpSetupTransaction(unittest.TestCase):
    def setUp(self):
        _Transport.instances.clear()
        _Buffer.instances.clear()
        _WeightManager.instances.clear()
        _EVENTS.clear()
        _Transport.fail_create = False
        _Transport.fail_commit = False
        _Transport.fail_release_once = False
        _Buffer.fail_create = False
        _Buffer.fail_release_once = False
        _WeightManager.fail_create = False
        self.model = _Model()
        self.experts = [layer.experts for layer in self.model.layers]
        self.original = [self._snapshot(experts) for experts in self.experts]
        self.manager = DwdpManager.__new__(DwdpManager)
        self.manager.dwdp_size = 2
        self.manager.dwdp_rank = 0
        self.manager.device_id = 0
        self.manager.vmm_backend = "fake"
        self.manager.layout = None
        self.manager._weight_manager = None
        self.manager._moe_layer_indices = []
        self.manager._moe_layers = []
        self.manager._setup_complete = False
        self.manager._fill_edge_bytes = lambda *args: None
        self.allgather_call = 0
        self.fail_allgather_call = None

    @staticmethod
    def _snapshot(experts):
        return {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in {
                "w13_weight": experts.w13_weight,
                "w2_weight": experts.w2_weight,
                "scale": experts.scale,
                "moe_ep_size": experts.moe_ep_size,
                "moe_ep_rank": experts.moe_ep_rank,
                "_num_local_routed": experts._num_local_routed,
                "num_local_experts": experts.num_local_experts,
                "runner_num_local_experts": experts.runner_num_local_experts,
                "dispatcher_moe_ep_size": experts.dispatcher_moe_ep_size,
                "dispatcher_moe_ep_rank": experts.dispatcher_moe_ep_rank,
            }.items()
        }

    def _assert_model_restored(self):
        for experts, original in zip(self.experts, self.original):
            for name, expected in original.items():
                actual = getattr(experts, name)
                if isinstance(expected, torch.Tensor):
                    self.assertTrue(torch.equal(actual, expected), name)
                else:
                    self.assertEqual(actual, expected, name)

    def _all_gather(self, shards, data, group):
        self.allgather_call += 1
        if self.allgather_call == self.fail_allgather_call:
            raise RuntimeError("injected side allgather")
        for rank, shard in enumerate(shards):
            shard.copy_(data + rank * 1000)

    def _run_setup(self):
        with (
            patch.object(_MODULE, "get_parallel", return_value=_Parallel()),
            patch.object(
                _MODULE,
                "get_dwdp_backend",
                return_value=(_Transport, _Buffer),
            ),
            patch.object(_MODULE, "DWDPWeightManager", _WeightManager),
            patch.object(_MODULE.dist, "all_gather", side_effect=self._all_gather),
            patch.object(
                _MODULE.torch.cuda,
                "synchronize",
                side_effect=lambda *args: _EVENTS.append("cuda-synchronize"),
            ),
        ):
            self.manager.setup(self.model)

    def _assert_failed_attempt_released_once(self):
        for resource in (
            _Transport.instances[0:1]
            + _Buffer.instances[0:1]
            + _WeightManager.instances[0:1]
        ):
            self.assertEqual(resource.release_count, 1, type(resource).__name__)

    def _fail_then_retry(self, message, after_failure=None):
        with self.assertRaisesRegex(RuntimeError, message):
            self._run_setup()
        self._assert_model_restored()
        self._assert_failed_attempt_released_once()
        if after_failure is not None:
            after_failure()

        _Transport.fail_create = False
        _Buffer.fail_create = False
        _Buffer.fail_release_once = False
        _WeightManager.fail_create = False
        _Transport.fail_commit = False
        self.fail_allgather_call = None
        for experts in self.experts:
            experts.fail_bind = False
            experts.fail_validate = False
        self.allgather_call = 0
        self._run_setup()
        self.assertTrue(self.manager._setup_complete)
        self.assertIsNotNone(self.manager._weight_manager)

    def test_weight_buffer_creation_failure_rolls_back_and_retries(self):
        _Buffer.fail_create = True
        self._fail_then_retry("weight buffer creation")

    def test_transport_creation_failure_releases_owned_object_and_retries(self):
        _Transport.fail_create = True
        self._fail_then_retry("transport creation")

    def test_edge_fill_failure_rolls_back_and_retries(self):
        original = self.manager._fill_edge_bytes
        calls = 0

        def fail_once(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected edge fill")
            return None

        self.manager._fill_edge_bytes = fail_once
        self._fail_then_retry(
            "edge fill",
            after_failure=lambda: self.assertEqual(
                _EVENTS[:3],
                ["cuda-synchronize", "buffer-release", "transport-release"],
            ),
        )
        self.manager._fill_edge_bytes = original

    def test_manager_creation_failure_rolls_back_and_retries(self):
        _WeightManager.fail_create = True
        self._fail_then_retry("manager creation")

    def test_mid_bind_failure_restores_weights_ep_and_retries(self):
        self.experts[1].fail_bind = True
        self._fail_then_retry("mid-bind")

    def test_side_allgather_failure_is_atomic_and_retries(self):
        self.fail_allgather_call = 2
        self._fail_then_retry("side allgather")

    def test_transport_commit_failure_restores_all_state_and_retries(self):
        _Transport.fail_commit = True
        self._fail_then_retry("transport commit")

    def test_buffer_release_error_still_releases_transport_once(self):
        _Transport.fail_commit = True
        _Buffer.fail_release_once = True
        with self.assertRaisesRegex(RuntimeError, "weight buffer release"):
            self._run_setup()
        self._assert_model_restored()
        self.assertEqual(_Buffer.instances[0].release_count, 1)
        self.assertEqual(_Transport.instances[0].release_count, 1)
        self.assertEqual(_WeightManager.instances[0].release_count, 1)

    def test_cleanup_retries_only_failed_buffer_release(self):
        self._run_setup()
        weight_manager = self.manager._weight_manager
        buffer = weight_manager._weight_buffer
        transport = weight_manager._transport
        _Buffer.fail_release_once = True

        with (
            patch.object(_MODULE.torch.cuda, "synchronize"),
            self.assertRaisesRegex(RuntimeError, "weight buffer release"),
        ):
            self.manager.cleanup()

        self.assertIs(self.manager._weight_manager, weight_manager)
        self.assertIs(weight_manager._weight_buffer, buffer)
        self.assertIsNone(weight_manager._transport)
        self.assertEqual(buffer.release_count, 1)
        self.assertEqual(transport.release_count, 1)

        with patch.object(_MODULE.torch.cuda, "synchronize"):
            self.manager.cleanup()
        self.assertIsNone(self.manager._weight_manager)
        self.assertIsNone(weight_manager._weight_buffer)
        self.assertIsNone(weight_manager._transport)
        self.assertEqual(buffer.release_count, 2)
        self.assertEqual(transport.release_count, 1)
        self.assertEqual(weight_manager.peer_views, {})

    def test_cleanup_retries_only_failed_transport_release(self):
        self._run_setup()
        weight_manager = self.manager._weight_manager
        buffer = weight_manager._weight_buffer
        transport = weight_manager._transport
        _Transport.fail_release_once = True

        with (
            patch.object(_MODULE.torch.cuda, "synchronize"),
            self.assertRaisesRegex(RuntimeError, "transport release"),
        ):
            self.manager.cleanup()

        self.assertIs(self.manager._weight_manager, weight_manager)
        self.assertIsNone(weight_manager._weight_buffer)
        self.assertIs(weight_manager._transport, transport)
        self.assertEqual(buffer.release_count, 1)
        self.assertEqual(transport.release_count, 1)

        with patch.object(_MODULE.torch.cuda, "synchronize"):
            self.manager.cleanup()
        self.assertIsNone(self.manager._weight_manager)
        self.assertIsNone(weight_manager._weight_buffer)
        self.assertIsNone(weight_manager._transport)
        self.assertEqual(buffer.release_count, 1)
        self.assertEqual(transport.release_count, 2)
        self.assertEqual(weight_manager.peer_views, {})

    def test_validation_midloop_restores_all_state_and_retries(self):
        self.experts[1].fail_validate = True
        self._fail_then_retry("validation")


if __name__ == "__main__":
    unittest.main()
