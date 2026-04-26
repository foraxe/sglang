# Tail Review: DSv4 TileKernels Fused Shared Top2 Gate

Scope reviewed: current diff in `python/sglang/srt/environ.py`,
`python/sglang/srt/layers/moe/topk.py`, and
`test/srt/test_dsv4_tilekernel_top2_gate.py`.

Result: the new `tilekernel_dsv4_fused_shared_top2_sum_gate` matches SGLang's
existing fused-shared top-k contract for the intended DSv4 case:
`router_logits=(N, 256)`, routed topk `6`, one fused shared expert, output topk
`7`, `sqrtsoftplus`, `routed_scaling_factor=1.5`, and
`apply_routed_scaling_factor_on_output=False`.

Why:
- The wrapper delegates routed expert selection to the existing
  `tilekernel_dsv4_top2_sum_gate` with `num_fused_shared_experts=0`, so it
  inherits the DSv4 guards for 256 experts, topk 6, grouped metadata 8/8,
  `correction_bias`, `sqrtsoftplus`, dtype/contiguity normalization, and the
  TileKernels scale undo.
- For the target non-output-scale path, routed weights come back normalized with
  sum 1. The helper appends shared weight `sum(routed) / 1.5`, i.e. `2/3`,
  which is the same fused-shared convention used by the existing DSv4
  `biased_topk_impl` path after routed-only renormalization.
- Expert id semantics match the fused weight layout: the shared expert is
  appended as id `num_experts == 256`, consistent with DSv4 fused loading that
  remaps `mlp.shared_experts` to `mlp.experts.256`.
- Ordering is acceptable for SGLang's fused-shared contract: routed experts are
  in the first six columns and the shared expert is always in the final column.
  The tests correctly canonicalize only the routed portion because TileKernels
  and `torch.topk(sorted=False)` do not promise identical routed ordering.

Concern:
- This should remain a narrow opt-in experiment. The new path appends logical id
  256 before `topk_ids_logical_to_physical()`. That is safe when fused shared
  expert id 256 is a real physical expert and no expert-location map lacking
  the shared entry is applied. It is not safe to broaden to EPLB/dynamic
  logical-to-physical dispatch without first extending the map or bypassing
  mapping for shared ids, because DSv4 expert-location metadata is currently
  sized to `n_routed_experts`.
- The helper ignores `renormalize` because the existing TileKernels DSv4 helper
  also hard-codes the normalized DSv4 contract. That is fine for the reviewed
  DSv4 shape, but it is another reason not to generalize the env flag beyond
  this exact configuration.

Verification:
- Source-checked against the existing DSv4 fused-shared implementations in
  `deepseek_v4_topk.py` and `topk.py`, plus DSv4 weight remapping in
  `deepseek_v4.py`.
- Tried `python -m pytest -q test/srt/test_dsv4_tilekernel_top2_gate.py`:
  blocked by `ModuleNotFoundError: No module named 'sglang'`.
- Retried with `PYTHONPATH=python`: collection still blocked by
  `ModuleNotFoundError: No module named 'triton'`.
