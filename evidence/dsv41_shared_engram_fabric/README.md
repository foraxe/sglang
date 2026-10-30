# SGLang DeepSeek-V4.1 shared Engram on one four-GB200 host

Captured 2026-10-30 (Asia/Shanghai date) from SGLang source commit
`10aa8622272a523483ce731de23d3e63c0a6a81a`, based on upstream main
`02cd9ff0a21f68d8adc716350baff653cc0eaaab`. The external owner was
vLLM draft #64082 at `46706ee5e5ade3f622b6f606bbd05b84b06a9b96`.
The model was `/data/models/DeepSeek-V4.1-Flash` (launch checkpoint), with
`config.json` SHA256
`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
and `model.safetensors.index.json` SHA256
`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.
Four GB200 GPUs were leased exclusively for each run; leases and processes
were released after teardown. All results below are from one host.

## Runtime and scope

The isolated SGLang overlay used Torch 2.14.1+cu130, sglang-kernel 0.4.10,
FlashInfer 0.7.2, sgl-deep-gemm 0.2.3.post2, sgl-flash-mla 1.0.0,
NIXL/nixl-cu13 1.5.1, sglang-router 0.3.2, Transformers 5.19.0 and
NVIDIA CUTLASS DSL 4.8.0. It inherited unrelated pure-Python dependencies
from the earlier SGLang venv through a `.pth` file; this is an exact tested
module/version list for the exercised paths, not a clean wheel-install gate.
Production source was imported from the `10aa8622` worktree. The vLLM owner
ran in its separate Torch 2.14.0+cu130 venv.

Serving used independent SGLang P TP2 on GPUs 0/1 and D TP2 on 2/3, DP1,
native NIXL P/D transfer through sglang-router, text-only, FP8 KV,
`--context-length 32768`,
`--mem-fraction-static 0.90`, `--max-running-requests 2`,
`--chunked-prefill-size 8192`, eager P/D execution, prefix cache disabled for
the 8K screen, and 32 forced decode tokens. Both arms used Torch BF16 and
Triton FP8 GEMM backends, FlashInfer MXFP4 MoE, FlashInfer autotune disabled,
and TileLang MHC pre/post disabled. These settings were held across storage
arms; they are a bounded integration workload, not the tuned production
recipe. The ordinary control used SGLang `placement=cpu, replication=none`.
The `per_node` host-table control could not start in this Python build because
`os.memfd_create` is absent.

## Evidence levels

| Gate | Observation | Limit |
| --- | --- | --- |
| Direct component | An independent SGLang process imported the full real generation on GPU1; both layers matched seven checkpoint boundary rows exactly through SGLang's gather and changed-ID CUDA Graph replay. Measured consumer RSS changes at the two attaches were 0 and 18 MiB, far below either 94.4 GiB table. | Same host, selected rows; RSS delta includes runtime overhead. |
| Physical table backing | The owner published 202,758,032,400 table bytes (188.833 GiB), 202,758,946,816 padded. In the ordinary P+D control, each role's four rank/layer host shards logged 193,365 MiB resident, so two roles carried two complete table extents. The shared generation removes one 188.833 GiB table backing. | Table backing, not total RSS, page cache or HBM. CPU control host MemFree varied with checkpoint page-cache reclaim. |
| Serving correctness | Native SGLang P TP2/D TP2 NIXL handoff completed one warmup plus three retained 13-input/32-output requests in each storage mode. Retained text and token-ID hashes matched 3/3 across modes. Both P and D exited zero; the shared owner logged eight attaches, eight detaches and exited zero. | Short fixed-prompt canary, no broad quality claim. |
| Graph/lifetime | A full SGLang TP2 engine captured decode graphs for batch sizes 1 and 2, then returned identical 32-token IDs for two fixed prompts. Four owner leases detached normally on shutdown. | Single-instance decode graph, eager P/D screen; no active-graph owner-failure test. |

## Matched 8K streaming screen

One saved 8,192-token retrieval prompt-ID fixture was used for each arm,
without prefix-cache reuse. One request warmed the P/D path; three retained
streaming requests each emitted 32 token-progress events, allowing client-side
TTFT and average TPOT measurement. One additional C4 wave offered four
simultaneous clients. All requests returned exactly 8,192 input and 32 output
tokens through NIXL. The table shows medians for retained sequential requests
and the one C4 wave; units are seconds except output tokens/s.

| Storage | TTFT | TPOT | P-to-D E2E | C4 wall | C4 output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGLang CPU host shards | 0.3346 | 0.1387 | 4.6347 | 10.0296 | 12.7622 |
| Shared HOST_NUMA/FABRIC owner | 0.3345 | 0.1376 | 4.6024 | 9.9351 | 12.8837 |

The shared/control ratios were 1.000 TTFT, 0.992 TPOT, 0.993 E2E and 1.010
C4 output rate. The preset first screen was no more than 20% regression in
each metric; it passed locally. This is one sequential arm order with three
retained single-request samples and one C4 wave. It does not prove a stable
speedup, tail SLO or peak QPS. Output text and token-ID hashes varied even
*within* each storage mode on this 8K fixture; it is performance/route
evidence, not a model-quality comparison. The CPU control logged 0 MiB huge
pages for all eight rank/layer shards, which may weaken it as an optimized
host-memory performance baseline. A better-page-size control should be
measured before making a production performance claim.

## Boundaries and reproduction

This implementation consumes the experimental vLLM owner protocol; the
current SGLang draft does not yet ship an independent provider binary. The
control socket needs a trusted network and a private token file. No second
host in the same NVL72/IMEX domain was available, so cross-host import,
remote Grace lookup latency and fabric contention remain unqualified. Full
3P1D serving would require more than these four GPUs. Image-bearing requests,
DSpark, full-model graph P/D serving, active-graph owner loss, broad model
evaluation and production readiness are outside this local gate.

Artifacts are under `/workspace/vllm_dsv41/artifacts/sglang-engram-shared/`.
`run_full_sglang_external_probe.py` checks real rows and graph replay;
`run_sglang_model_smoke.py --decode-graph` checks full-model decode graphs;
`run_sglang_pd_smoke.py` runs the short P/D canary, and `--long` runs the
8K streaming/C4 screen. Add `--baseline` for the SGLang CPU host-table
control. `summarize_sglang_engram.py` regenerates `local_summary.json`,
which holds per-artifact SHA256 values. The public evidence copy omits raw
8K prompt IDs and generated text; selected logs and compact reports retain
the derivation and source/runtime provenance.
