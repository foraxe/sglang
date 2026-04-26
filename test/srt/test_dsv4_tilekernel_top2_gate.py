import importlib.util

import pytest
import torch

from sglang.srt.layers.moe import topk as topk_mod
from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    select_experts,
    tilekernel_dsv4_top2_sum_gate,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("tile_kernels") is None,
    reason="requires CUDA and tile_kernels",
)


def _canonicalize(topk_weights, topk_ids):
    sorted_ids, order = topk_ids.sort(dim=1)
    sorted_weights = torch.gather(topk_weights, 1, order)
    return sorted_weights, sorted_ids


def _reference_dsv4_top2_gate(
    router_logits,
    correction_bias,
    topk,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
):
    scores = torch.nn.functional.softplus(router_logits).sqrt()
    _, topk_ids = torch.topk(
        scores + correction_bias.unsqueeze(0),
        k=topk,
        dim=-1,
        sorted=False,
    )
    topk_weights = scores.gather(1, topk_ids)

    topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights / topk_weights_sum
    if apply_routed_scaling_factor_on_output:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.float(), topk_ids.int()


@pytest.mark.parametrize("num_tokens", [1, 32, 256, 4096])
def test_tilekernel_dsv4_top2_gate_matches_sglang_contract(
    monkeypatch, num_tokens
):
    monkeypatch.setattr(topk_mod, "get_moe_expert_parallel_rank", lambda: 0)
    monkeypatch.setattr(topk_mod, "get_moe_expert_parallel_world_size", lambda: 1)
    monkeypatch.setattr(topk_mod, "get_moe_tensor_parallel_rank", lambda: 0)
    monkeypatch.setattr(topk_mod, "get_moe_tensor_parallel_world_size", lambda: 1)

    torch.manual_seed(20260426 + num_tokens)
    router_logits = torch.randn((num_tokens, 256), dtype=torch.float32, device="cuda")
    correction_bias = torch.randn((256,), dtype=torch.float32, device="cuda")

    topk_weights, topk_ids = tilekernel_dsv4_top2_sum_gate(
        router_logits=router_logits,
        correction_bias=correction_bias,
        num_topk=6,
        num_topk_groups=8,
        num_groups=8,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        apply_routed_scaling_factor_on_output=False,
    )
    ref_weights, ref_ids = _reference_dsv4_top2_gate(
        router_logits,
        correction_bias,
        6,
        1.5,
        False,
    )

    assert topk_ids.dtype == torch.int32
    assert topk_weights.dtype == torch.float32
    topk_weights, topk_ids = _canonicalize(topk_weights, topk_ids)
    ref_weights, ref_ids = _canonicalize(ref_weights, ref_ids)
    assert torch.equal(topk_ids, ref_ids)
    assert torch.allclose(topk_weights, ref_weights)


def test_select_experts_tilekernel_dsv4_path_matches_sglang_contract(monkeypatch):
    monkeypatch.setattr(topk_mod, "get_moe_expert_parallel_rank", lambda: 0)
    monkeypatch.setattr(topk_mod, "get_moe_expert_parallel_world_size", lambda: 1)
    monkeypatch.setattr(topk_mod, "get_moe_tensor_parallel_rank", lambda: 0)
    monkeypatch.setattr(topk_mod, "get_moe_tensor_parallel_world_size", lambda: 1)

    num_tokens = 32
    torch.manual_seed(20260426)
    hidden_states = torch.randn((num_tokens, 7168), dtype=torch.bfloat16, device="cuda")
    router_logits = torch.randn((num_tokens, 256), dtype=torch.float32, device="cuda")
    correction_bias = torch.randn((256,), dtype=torch.float32, device="cuda")
    topk_config = TopKConfig(
        top_k=6,
        use_grouped_topk=False,
        topk_group=8,
        num_expert_group=8,
        renormalize=True,
        num_fused_shared_experts=0,
        correction_bias=correction_bias,
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=False,
        scoring_func="sqrtsoftplus",
    )

    with envs.SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE.override(True):
        output = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=topk_config,
        )

    ref_weights, ref_ids = _reference_dsv4_top2_gate(
        router_logits,
        correction_bias,
        6,
        1.5,
        False,
    )

    topk_weights, topk_ids = _canonicalize(output.topk_weights, output.topk_ids)
    ref_weights, ref_ids = _canonicalize(ref_weights, ref_ids)
    assert torch.equal(topk_ids, ref_ids)
    assert torch.allclose(topk_weights, ref_weights)
