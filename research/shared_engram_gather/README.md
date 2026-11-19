# Shared Engram: SGLang and DOCA 3.5 experiment

This experiment shares one host-memory Engram table between independent serving instances. Decode reads its local backing; prefill fetches selected rows over RDMA. The optional gathered-row path uses a reader-side CPU worker and owner-side CPU submission to transfer several rows per RDMA WRITE. The native device-GET path remains available.

The experiment is separate from [SGLang #47460](https://github.com/sgl-project/sglang/pull/47460). The code tested here is [SGLang `b5714c4d6c`](https://github.com/foraxe/sglang/commit/b5714c4d6cff030fcae28996ab60df1569a32a17); the component test used provider `a0fc1c2add` and benchmark `f8c9d1fe98`. The current PR does not contain this gathered transport.

## Results

| Workload | Native device GET | Gathered rows |
| --- | ---: | ---: |
| H20 component, two-layer GPU p50 | 3.881 ms | 2.562 ms |
| H20-3e 1P1D, TTFT p50 / p99 | 8374.4 / 9219.1 ms | 8368.1 / 9213.5 ms |
| H20-3e 1P1D, TPOT p50 | 3.077 ms | 3.075 ms |
| H20-3e 1P1D, output throughput | 59.334 tok/s | 59.363 tok/s |

The component workload reads 32,768 distinct checkpoint-prefix rows for each of two layers. Both arms allocate 16 native and 16 gather QPs and use the same prepared graph, source table and 10 measured samples after three warmups. The GPU elapsed p50 decreased by 33.99%. The result covers lookup and consumption; it is not a serving latency measurement.

The serving comparison uses one fixed-order pair on two H20-3e nodes: P TP4/CP4, D TP4/DP4, DSpark, concurrency 8, 16K input and 64 output tokens. Each arm used eight warmups and 32 measured requests. Both arms completed all requests and all 32 paired input and output hashes matched. P's internal READ graph replayed on all four ranks; D's decode graph executed. The baseline configuration disabled outer prefill graph on both arms. TTFT and throughput were essentially unchanged in this sample; the component gain should not be interpreted as a serving gain.

Two separate reader processes also connected concurrently to the same 188.83 GiB checkpoint backing and completed both-layer lookups, native fallback and normal teardown. Both readers used one GPU; this does not establish multi-node or eight-rank scaling. [Anonymized numeric samples](samples.json) contain the component measurements and serving request latencies; prompts, outputs, addresses and keys are not included.

## Sources and build

- SGLang code: [tested commit `b5714c4d6c`](https://github.com/foraxe/sglang/tree/b5714c4d6cff030fcae28996ab60df1569a32a17).
- Native GET: [NIXL source `e56e4afa`](https://github.com/foraxe/nixl/tree/e56e4afa985ee2997f78675ae897c964d1d3568f), with the [ABI bridge and build instructions](https://github.com/foraxe/nixl/tree/c666b177/research/gpunetio_doca35_get_closeout). The measured bridge SHA-256 was `12977f4a648721077eb69a0129f398ba88551f3fa738dd9835a5d9276d976947`.
- Gather transport: [source and build script at `7f97592737`](https://github.com/foraxe/vllm/tree/7f97592737/research/engram-gather-20261118). The measured source `gather_graph.cu` had SHA-256 `2f753a21d20cfccd1569d1822789ce2d47c76b5ea8301fc74719f92cf486bdc7`; the tested library was `14e4881a0bfec0281e02af1a575082c405404441433ef41df711866049494abb`. That directory also contains a later v2 variant; the reported numbers use the original source, not v2.
- The tested stack used DOCA 3.5.0098. Build against matching NIXL core and GPUNETIO plugin libraries on both nodes and record the resulting hashes.

## Component reproduction

Use two Linux hosts with compatible GPUs, RDMA connectivity, the same indexed checkpoint, CUDA-enabled PyTorch, Triton, safetensors, DOCA 3.5 and libibverbs. Build the native bridge and gather library using the linked instructions. On both hosts, set `PYTHONPATH` to the tested SGLang `python` directory, `NIXL_PLUGIN_DIR` and library paths to the matching NIXL build, and configure `GPUNETIO_DEVICE_API_RDMA`, `GPUNETIO_DEVICE_API_GID_INDEX` and `GPUNETIO_DEVICE_API_OOB_INTERFACE` for the local device. Choose available control and native ports outside the host's ephemeral port range.

On the owner, set `MODEL`, `NATIVE_SO`, `GG_SO`, `OWNER_HOST`, `NATIVE_PORT`, `CONTROL_PORT`, `GG_PORT` and a private, empty `WORK` directory:

```bash
export ENGRAM_SHARED_GATHER_LIBRARY="$GG_SO"
export ENGRAM_SHARED_GATHER_MAX_READERS=1
export ENGRAM_SHARED_GATHER_PORT_BASE="$GG_PORT"
python -m sglang.srt.engram_share.rdma.owner \
  --checkpoint "$MODEL" --library "$NATIVE_SO" --owner-gpu 0 \
  --rdma-port "$NATIVE_PORT" --control-host 0.0.0.0 \
  --advertise-host "$OWNER_HOST" --control-port "$CONTROL_PORT" \
  --ready-file "$WORK/owner.json" --stop-file "$WORK/owner.stop" \
  --max-pinned-gib 190 --storage-layout row264
```

Copy the ready file securely to the reader and set `MODEL`, `NATIVE_SO`, `GG_SO`, `OWNER_READY`, and two distinct free port bases `PORT_A` and `PORT_B`. Run the matched native and gather arms sequentially:

```bash
export ENGRAM_SHARED_MODEL="$MODEL" ENGRAM_SHARED_LIBRARY="$NATIVE_SO"
export ENGRAM_SHARED_GATHER_LIBRARY="$GG_SO" ENGRAM_SHARED_READY="$OWNER_READY"
export ENGRAM_SHARED_PREFETCH=1 ENGRAM_SHARED_READ_PLAN=1 ENGRAM_SHARED_READ_LANES=16
export ENGRAM_SHARED_MAX_QUERIES=32768 ENGRAM_SHARED_READ_GRAPH_SIZES=32768
export ENGRAM_SHARED_DEFER_LAYER14=1 ENGRAM_SMOKE_TOKENS=2048 ENGRAM_SMOKE_TIMING_ROUNDS=10

ENGRAM_SMOKE_GATHER_NATIVE_CONTROL=1 ENGRAM_SHARED_READER_PORT_BASE="$PORT_A" \
  python test/manual/engram_remote_smoke.py
ENGRAM_SHARED_READER_PORT_BASE="$PORT_B" \
  python test/manual/engram_remote_smoke.py
```

Both commands verify checkpoint rows and report `TIMING` samples. The first retains the gathered endpoints but selects native GET for the measured sizes. The owner ready file contains RDMA addresses and keys; restrict access and use a trusted network. After both readers have closed, stop the owner with `touch "$WORK/owner.stop"` and wait for clean exit.

The serving numbers used a private request set, so the component commands do not reproduce the serving table. The public samples support inspection of the measured distributions without exposing request content. External CP outer graph capture, crash-safe reclamation, broader hardware and multi-P serving are outside the demonstrated scope.
