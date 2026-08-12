# H20 SGLang DWDP coTensor P1 result

## Classification

- Adapter/static integration: `PASS` at `c1216ea4f14f1e566a781777f2ac43c121b0749f`.
- Real 2-rank synthetic DWDP lifecycle E1: `PASS` for native and coTensor.
- Full `gpt-oss-20b` service E1/E2: `BLOCKED`; no TTFT/throughput claim.

## Environment

- Node `lj-21d317105`, 8 x NVIDIA H20; experiment pins GPUs 0-1.
- Torch `2.11.0+cu130`; CUDA runtime 13.0.
- SGLang base `93e9db5eb89d51e6818d43062d0296612bf53061`.
- coTensor P0.4 reviewed API/code `7b4d8b2`; final evidence head `86d223d`.
- Model attempted: `/data/nas/moyun.zty/models/OpenAI/gpt-oss-20b`.

## E1 evidence

The harness invokes the production `DWDPTransport`, `WeightBuffer`, and
`DWDPWeightManager` copy-stream/event protocol with two ranks, two layers, both
weight names, two alternating slots, peer POSIX-FD exchange, and misaligned
expert bytes (`granularity / sizeof(float) + 17` elements).

- Native and coTensor: zero mismatches on both ranks.
- Cross-backend hashes match exactly:
  - `0:w13_weight = 5243050`
  - `0:w2_weight = 26215250`
  - `1:w13_weight = 214965050`
  - `1:w2_weight = 235937250`
- 200 prefetch operations after warmup:
  - native rank 0/1: `7.832 / 7.147 ms` GPU
  - coTensor rank 0/1: `6.915 / 6.853 ms` GPU
- Post-process cleanup: all eight GPUs returned to `1 MiB`, `0%` utilization.
- In-process `fd_delta=24` and HBM deltas include live NCCL/c10d/CUDA contexts;
  they are diagnostic and not classified as a coTensor leak. The post-process
  GPU check is the valid cleanup gate; PID-scoped FD cleanup remains unproven.

The initial candidate mapped nonzero physical offsets from one large page-pool
Slab and failed with `cuMemMap ... CUDA_ERROR_NOT_SUPPORTED (801)`. Native DWDP
uses one allocation handle per page. The fixed candidate matches that structure
with one coTensor Endpoint per page and maps every allocation at offset zero.

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

