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

The first live smoke therefore disabled SGLang's TileLang MHC pre/post kernels:

```bash
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
```

The follow-up fix is SGLang-side: inspect `T.gemm` at import time and pass `wg_wait=0` only when the installed TileLang accepts it. With that guard, `tilelang==0.1.9`, TileKernels router, and SGLang MHC pre/post can run together in the four-layer DSv4 smoke.

Rejected alternative: making current TileKernels source compatible with `tilelang==0.1.8`. That version still exposes `T.gemm(..., wg_wait=...)`, but the current TileKernels `top2_sum_gate` source relies on newer TileLang `T.shfl_sync(value, lane)` behavior and correctness broke after a local compatibility attempt. The local TileKernels branch was reverted to its known-good state.

## A/B Result

A bounded A/B run used:

- TP=1
- four DSv4 layers
- `tilelang==0.1.9`
- MHC pre/post enabled
- baseline SGLang top-k vs `SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE=1`

Warm 4-token request:

- baseline: HTTP 200, `warm_elapsed_s=132.33`
- TileKernels: HTTP 200, `warm_elapsed_s=138.33`

Long `4096+2` request:

- baseline: manually terminated after staying connected but not reaching scheduler admission / model forward
- TileKernels: same behavior

Conclusion: the current evidence says the TileKernels router path can run with the MHC compatibility guard, but it does not yet show a performance win. The long-input stall is shared by both variants in this bounded setup, so it is not caused by the TileKernels router hook.

## Remaining Risks

- Not PR-ready as a production feature because TileKernels is not a normal SGLang dependency in this image and no speedup has been proven.
- Full model TP=4 and throughput benchmarks are not validated for this branch.
- HashTopK layers are not changed; the hook only applies to non-hash DSv4 routing layers.
- The live smoke used `SGLANG_APPLY_CONFIG_BACKUP=small` and four layers to keep the test bounded.
- Long-input `4096+2` requests hang before scheduler admission for both baseline and TileKernels in the current TP=1 four-layer setup.

## Fused-Shared Tail-Kernel Shape

The DSv4 fused-shared router problem shape is:

- `router_logits.shape == (N, 256)`
- `correction_bias.shape == (256,)`
- routed expert top-k: `6`
- fused shared expert count: `1`
- total output top-k: `7`
- groups: `num_groups == 8`, `num_topk_groups == 8`
- scoring: `sqrtsoftplus`
- shared expert logical id: `256`

SGLang's current fused-shared contract appends shared experts in the final top-k columns, not mixed into the routed top-k ranking. The routed expert ids occupy columns `0..5`; the shared expert occupies column `6`. Before final normalization, the shared weight is the routed selected-weight sum divided by `routed_scaling_factor`. Because fused-shared normalization divides all outputs by the routed selected-weight sum, the final shared column is `1 / routed_scaling_factor` when `apply_routed_scaling_factor_on_output=False`.

The new opt-in experiment is:

```bash
export SGLANG_OPT_USE_TILEKERNEL_DSV4_FUSED_SHARED_TOP2_GATE=1
```

Implementation:

- Uses TileKernels only for the routed-only `top2_sum_gate` computation with routed `topk=6`.
- Appends the shared expert tail column in SGLang's wrapper so the output matches SGLang's fused-shared contract exactly.
- Keeps the path restricted to one fused shared expert and DSv4 routed `topk=6`.
- Applies logical-to-physical expert mapping only to routed columns. The appended shared column remains the shared expert id, which avoids indexing a routed-only EPLB dispatch table with id `256`.

This is a contract-validation step, not a full production routing replacement. The live DSv4 2604B FP4 path observed so far uses `num_fused_shared_experts=0`, so this branch is validated as a CUDA unit/integration path rather than a full `/generate` path.
