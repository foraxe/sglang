# H20 SGLang DWDP coTensor P1 result

## Classification

- Adapter/static integration: `PASS` at `508db3af49334b0b34e85f0b31ed160959a962f3`.
- Real 2-rank synthetic DWDP lifecycle E1: `PASS` for native and coTensor.
- Full `gpt-oss-20b` service E1/E2: `BLOCKED`; no TTFT/throughput claim.

## Environment

- Node `lj-21d317105`, 8 x NVIDIA H20; experiment pins GPUs 0-1.
- Torch `2.11.0+cu130`; CUDA runtime 13.0.
- SGLang base `93e9db5eb89d51e6818d43062d0296612bf53061`.
- coTensor P0.4 final reviewed evidence head `86d223d`.
- Model attempted: `/data/nas/moyun.zty/models/OpenAI/gpt-oss-20b`.

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

## Full-model blocker

The image's `sglang-kernel 0.4.4+cu130` is ABI-compatible with Torch 2.11, but
the pinned SGLang main requires `0.4.6.post1`. The generic 0.4.6 wheel failed to
load with an undefined Torch symbol. With the 0.4.4 version check experimentally
lowered, native DWDP loaded all three model shards, then made no log/GPU/CPU
progress for more than 10 minutes: workers held about 10,164 MiB each at 0%
GPU utilization. It was killed per the no-progress gate and GPU memory returned
to 1 MiB.

Therefore model output equality, TTFT, prefill latency, throughput, and service
shutdown remain `BLOCKED`, not `PASS` or `FAIL` for coTensor.
