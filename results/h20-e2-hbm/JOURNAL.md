# H20 DWDP E2 HBM closeout journal

Date: 2026-08-12 (Asia/Shanghai)

## Contract

- Hypothesis: the observed coTensor `+64 MiB` maximum HBM is either launch
  noise or one extra retained page caused by the one-Slab/Endpoint-per-page
  adapter representation.
- Baseline: `--dwdp-vmm-backend native` at `acb705ee7`.
- Candidate: `--dwdp-vmm-backend cotensor`, same commit, model, GPUs 0-1,
  TP=2, DWDP=2, request, cache policy, and allocator settings.
- Order: five independent `A B B A` blocks, where A=native and B=coTensor;
  every service process exits before the next row.
- Stages: pre-CUDA, CUDA initialized, model loaded, local Endpoint allocation,
  peer FD import/map, WeightBuffer/page-pool creation, warmup steady state, and
  teardown.
- Measurements: NVML bytes and `nvidia-smi` MiB per GPU, Torch allocated and
  reserved bytes, VMM physical bytes, allocation/page/Endpoint/import/map/lease
  counts, process FD count, exact phase timestamps, and output hash.
- Noise decision: compare coTensor and native steady-HBM medians within the
  served-DP-rank strata. ABBA block and unstratified deltas remain descriptive
  because requests may be routed to different DP ranks.
- Correctness PASS: native/coTensor output hashes match, no backend fallback,
  no lifecycle error, and post-row GPU/FD state returns to baseline. The
  expected FABRIC-to-POSIX capability fallback is not a backend fallback.
- Strict E2 PASS: coTensor steady HBM <= native within every served-DP-rank
  stratum, median TTFT and prefill regression <=3%, with full E1/lifecycle
  tests passing.
- Kill: any corruption, persistent resource growth, or repeated hang; do not
  touch P2 or scheduling/kernel behavior.

## Results

`PASS` at the final fixed-seed ABBA x 5 run.

- The first six-row pilot is `INVALID` because SGLang selected a different
  random seed per launch. It is preserved remotely as
  `hbm-abba-invalid-unpinned-seed` and excluded from classification.
- Final run: 20/20 rows completed, fixed `--random-seed 12345`, identical
  output SHA-256 `006bbd7d...`, and every teardown returned both GPUs to
  `1 MiB`, `0%`.
- Acceptance uses DP-rank-stratified steady medians: served DP0 is
  `[81,030, 80,636] MiB` for both backends; served DP1 is
  `[80,588, 81,078] MiB` for both. Candidate delta is exactly `[0, 0] MiB` in
  both strata. The first native cold row `[80,648, 81,098]` is retained; the
  DP1 median remains `[80,588, 81,078]` with that row included. Unstratified median row maxima are native
  `81,078 MiB`, coTensor `81,054 MiB` (`-24 MiB`, descriptive only).
  Block median deltas are
  `[-10, +48, -24, -24, -48] MiB`: no stable nonzero/page-sized mode in 4/5
  blocks. The prior `+64 MiB` is launch/request-DP state noise.
- The served `dp_rank` explains the two dominant steady-state shapes:
  approximately `[81,030, 80,636]` for rank 0 and `[80,588, 81,078]` for rank
  1, independent of backend.
- Candidate runtime counters match the native static allocation formula. Per rank, coTensor has 48 local
  allocations/Endpoints (`18,288 MiB` rank 0, `18,336 MiB` rank 1), 48 peer
  imports/maps, plus 96 page-pool allocations/Endpoints (`1,536 MiB`). It has
  48 weight slots and 1,200 mapped views/live leases (48 local + 1,152 pool).
  Endpoint/view retention adds no physical allocation.
- Commit-time diagnostic `torch.cuda.empty_cache()` changed neither Torch
  allocated/reserved bytes nor NVML used bytes. The proposed post-commit cache
  flush is therefore rejected; no production memory-policy change was made.
- Raw rows, server stage probes, FD/Torch/NVML counters, nvidia-smi samples,
  and output metadata are in `hbm-abba/`; summary is `SUMMARY.json`.

Strict E2 is now `PASS`: stratified coTensor steady HBM equals native, while the prior fixed
prompt run already established TTFT `-0.399%` and prefill `+0.172%` versus
native, both within the 3% gate.
