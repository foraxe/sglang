# DSv4 TileKernels Router Findings

## Scope

This worktree is an opt-in experiment for using TileKernels' DSv4 `top2_sum_gate` inside SGLang's non-hash DSv4 MoE routing path.

The hook is intentionally restricted to the DSv4 FP4 non-fused shared-expert contract:

- `router_logits.shape[1] == 256`
- `num_topk == 6`
- `num_groups == 8`
- `num_topk_groups == 8`
- `scoring_func == "sqrtsoftplus"`
- `correction_bias is not None`
- `num_fused_shared_experts == 0`

Reason: DSv4 2604B FP4 disables shared expert fusion in SGLang because shared and routed experts need different clamping. TileKernels' fused-shared convention is not the same as SGLang's fused-shared top-k contract, so this patch does not wire that path.

## Implementation Notes

- New env flag: `SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE=1`.
- SGLang expects `StandardTopKOutput(topk_weights, topk_ids, router_logits)`.
- TileKernels returns `(topk_ids, topk_weights)`.
- TileKernels applies `routed_scaling_factor` internally; SGLang's current non-fused path expects unscaled top-k weights before the MoE runner, so the wrapper divides the routed weights by the scale unless `apply_routed_scaling_factor_on_output=True`.
- The patched `select_experts` path now maps logical expert IDs through `topk_ids_logical_to_physical(...)`, matching the existing JIT path behavior when expert-location dispatch is active.

## Bugs Found

The first SGLang integration patch only tested the helper directly. A live request and an independent review found that `select_experts` fell through into `biased_topk_impl(...)` after the TileKernels branch, causing:

```text
UnboundLocalError: cannot access local variable 'biased_topk_impl' where it is not associated with a value
```

Fix: keep the old JIT path under an `else` branch and add a test that calls `select_experts(...)` with `SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE=True`.

## Dependency Boundary

The SGLang image `lmsysorg/sglang:deepseek-v4-grace-blackwell` contains a TileLang version that is too old for current TileKernels. Installing `tilelang==0.1.9` allows TileKernels' DSv4 gate tests to run, but SGLang's TileLang MHC kernels then fail with:

```text
TypeError: gemm() got an unexpected keyword argument 'wg_wait'
```

The live smoke therefore disables SGLang's TileLang MHC pre/post kernels:

```bash
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
```

This means the router experiment is validated as functional, but dependency compatibility remains unresolved for production-style integration.

## Remaining Risks

- Not PR-ready as a production feature because TileKernels is not a normal SGLang dependency in this image.
- Full model TP=4 and throughput benchmarks are not validated for this branch.
- HashTopK layers are not changed; the hook only applies to non-hash DSv4 routing layers.
- The live smoke used `SGLANG_APPLY_CONFIG_BACKUP=small` and four layers to keep the test bounded.

