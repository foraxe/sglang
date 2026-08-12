# H20 SGLang DWDP coTensor P1 result

## Classification

- Adapter/static integration and rollback ownership tests: `PASS` at `ccf5431e0f4d32eb1fc70f5eb400e700b0bd7c66`.
- Real 2-rank synthetic DWDP lifecycle E1: `PASS` for native and coTensor.
- Full `gpt-oss-20b` service E1/E2: `PASS` for native and coTensor.

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
- Steady HBM GPU 0/1: native `80,990 / 81,014 MiB`; coTensor
  `81,008 / 81,078 MiB` (candidate maximum +64 MiB).
- 3,500-token prefill plus one output token: native `0.240 s`, coTensor
  `0.657 s`. This is one HTTP E2E sample and is directional, not an acceptance
  performance claim.
- Median 32-token request throughput: native `19.69 tok/s`, coTensor
  `20.50 tok/s` (+4.1% directional).
- HTTP non-streaming max-new-token=1 latency is a prefill/E2E proxy. A true
  streaming TTFT was not captured and remains an explicit measurement gap.
- Shutdown: all eight GPUs returned to `1 MiB`, `0%`; the pod was retained.

Transaction regression tests cover nine injected setup/cleanup failures and
one late SCM_RIGHTS timeout race. All ten targeted tests pass, including exact
FD-count restoration, receiver join, model/EP rollback, release ordering, and
successful retry.
