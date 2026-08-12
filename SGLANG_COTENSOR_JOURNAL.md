# SGLang DWDP coTensor P1 journal

## Pinned scope

- Upstream base: `93e9db5eb89d51e6818d43062d0296612bf53061` (`main`,
  2026-08-11).
- DWDP landing: `37a830b667098b41f355dfde58518154971efb32` (#29778,
  2026-07-21).
- Candidate changes only allocation/export/import/map/unmap ownership and
  lifecycle. Expert layout, double-buffer scheduling, copy/compute streams, and
  kernels remain unchanged.
- Runtime fallback: `--dwdp-vmm-backend native` remains the default;
  `--dwdp-vmm-backend cotensor` selects the candidate.
- Single-node Linux CUDA VMM/POSIX-FD is the first candidate transport. FABRIC,
  multicast, KV cache, decode, and scheduler changes are non-goals.

## E1 correctness

- Hypothesis: replacing DWDP's native VMM lifecycle with coTensor Endpoint does
  not change expert bytes, model outputs, or prefetch ordering.
- Baseline: `native`; candidate: `cotensor`.
- Configuration: same pinned SGLang/coTensor commits and model; `dwdp_size ==
  tp_size`; 2 ranks before 4/8 ranks.
- Metrics: expert byte/hash equality, output equality, both buffer slots,
  prefetch completion, startup/request/shutdown FD and HBM deltas.
- PASS: zero fallback, zero mismatch, zero silent remap, zero resource growth.
- FAIL: any corruption, hang, fallback, lifecycle error, or persistent FD/HBM
  growth. Stop E2 and return to the adapter/P0 surface.
- CUDA graph replay is recorded as `INVALID` for this upstream revision because
  `_handle_dwdp()` explicitly forces `disable_cuda_graph=True`; it is not
  evidence against either backend.

## E2 performance and capacity

- Hypothesis: coTensor centralizes lifecycle without increasing steady-state HBM
  or materially delaying prefill.
- Method: identical model, request set, rank topology, streams, and warmup;
  interleave native/cotensor; one warmup plus at least three measured runs.
- Metrics: HBM, prefetch wait, copy/compute overlap, prefill latency, TTFT,
  throughput, initialization, and teardown.
- PASS: candidate HBM <= baseline; median prefill and TTFT regression <= 3%; no
  material p95 tail regression. Report raw rows and lifecycle diff separately.
- KILL: stop the scale row on correctness failure or repeatable >3% median
  regression; do not tune scheduling or kernels inside P1.

## Evidence log

| Time | Commit | Environment | Command | Result |
| --- | --- | --- | --- | --- |
| 2026-08-12 | `93e9db5eb` | macOS static checkout | pin upstream and trace DWDP | `PASS`: exact base and landing commit established |
| 2026-08-12 | `c1216ea4f` | H20 2-rank synthetic | native then coTensor misaligned lifecycle E1 | `PASS`: hashes equal, zero mismatch, 200 prefetch ops/backend |
| 2026-08-12 | `c1216ea4f` | H20 gpt-oss-20b native | full service startup | `BLOCKED`: kernel ABI/version conflict, then >10 min post-shard stall |
| 2026-08-12 | `508db3af4` | H20 2-rank synthetic | full-byte SHA-256 native/cotensor | `PASS`: identical hashes, zero element mismatch, post-exit 1 MiB/GPU |
| 2026-08-12 | `d4164544a` + final lifecycle patch | H20 gpt-oss-20b, 2 ranks, radix cache disabled | native/cotensor true streaming TTFT and prefill, 1 warmup + 3 measured | TTFT `PASS` (-0.399% median), prefill `PASS` (+0.172% median), output `PASS`; strict HBM `FAIL` (+64 MiB), therefore E2 `FAIL` |
