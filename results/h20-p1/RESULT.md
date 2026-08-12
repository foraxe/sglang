# H20 SGLang DWDP coTensor P1 result

## Classification

- Adapter/static integration and rollback ownership tests: `PASS` at `ccf5431e0f4d32eb1fc70f5eb400e700b0bd7c66`.
- Real 2-rank synthetic DWDP lifecycle E1: `PASS` for native and coTensor.
- Full `gpt-oss-20b` service: E1 correctness and strict E2 both `PASS`.

## Environment

- Node `lj-21d317105`, 8 x NVIDIA H20; experiment pins GPUs 0-1.
- Torch `2.11.0+cu130`; CUDA runtime 13.0; `sglang-kernel 0.4.5+cu130`.
- SGLang base `93e9db5eb89d51e6818d43062d0296612bf53061`.
- coTensor P0.4 final reviewed evidence head `86d223d`.
- Model: node-local verified copy at `/home/yunzhi.nyx/h20_team/models/gpt-oss-20b`.

## E1 evidence

The harness invokes the production `DWDPTransport`, `WeightBuffer`, and
`DWDPWeightManager` copy-stream/event protocol with two ranks, two layers, both
weight names, two alternating slots, peer POSIX-FD exchange, and misaligned
expert bytes (`granularity / sizeof(float) + 17` elements).

- Native and coTensor: zero mismatches on both ranks.
- Full-byte SHA-256 hashes match exactly across native/coTensor and both ranks;
  see `native-sha256-508db3a.log` and `cotensor-sha256-508db3a.log`.
- Final 200-prefetch check after warmup:
  - native rank 0/1: `7.068 / 7.087 ms` GPU
  - coTensor rank 0/1: `6.973 / 6.991 ms` GPU
- Post-process cleanup: all eight GPUs returned to `1 MiB`, `0%` utilization.
- In-process `fd_delta=24` and HBM deltas include live NCCL/c10d/CUDA contexts;
  they are diagnostic and not classified as a coTensor leak. The post-process
  GPU check is the valid cleanup gate; PID-scoped FD cleanup remains unproven.

The initial candidate mapped nonzero physical offsets from one large page-pool
Slab and failed with `cuMemMap ... CUDA_ERROR_NOT_SUPPORTED (801)`. Native DWDP
uses one allocation handle per page. The fixed candidate matches that structure
with one coTensor Endpoint per page and maps every allocation at offset zero.

The lifecycle patch also snapshots/restores model weight and EP/dispatcher state
on setup rollback, covers transport/weight-buffer/edge-fill failures, retains
component View roots while composite raw-pointer tensors are live, checks zero
root leases before unmap, and makes partial FD-exchange cleanup single-owner.

## Full-model E1/E2

The compatible image and node-local model copy removed the earlier kernel ABI
and NAS page-fault blockers. Both backends reached HTTP health and served the
same fixed requests. Raw durable rows are in `full-model-ab.json`.

- Exact output: `PASS`; prefill and 32-token decode SHA-256 match between
  native and coTensor.
- Fixed-seed ABBA x 5 steady HBM, stratified by served DP rank: DP0 is
  `[81,030, 80,636] MiB` and DP1 `[80,588, 81,078] MiB` for both backends;
  exact delta `0 MiB`: strict HBM gate `PASS`. The earlier
  single-row `+64 MiB` observation followed served DP-rank state and is noise.
- True streaming TTFT, one warmup plus three measured 3,500-token rows:
  native `[190.949, 191.015, 190.739] ms`, median `190.949 ms`; coTensor
  `[239.176, 189.508, 190.187] ms`, median `190.187 ms`. Regression `-0.399%`:
  `PASS` against the <=3% gate.
- Non-streaming 3,500-token prefill proxy, three measured rows: native
  `[236.012, 235.968, 235.045] ms`, median `235.968 ms`; coTensor
  `[237.470, 236.375, 235.576] ms`, median `236.375 ms`. Regression `+0.172%`:
  `PASS` against the <=3% gate.
- Median 32-token request throughput: native `19.69 tok/s`, coTensor
  `20.50 tok/s` (+4.1% directional).
- Raw client timestamps, SSE chunks, response JSON, prompt/output hashes, and
  postflight are committed as `ttft-*-raw.json` and `postflight-ttft.txt`.
- Shutdown: all eight GPUs returned to `1 MiB`, `0%`; the pod was retained.

Transaction regression tests cover eleven setup/cleanup failure cases and one
late SCM_RIGHTS timeout race. All twelve targeted tests pass, including
one-shot buffer/transport release failure followed by successful cleanup retry.
