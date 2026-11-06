# Proposal: remote shared Engram tables

> Discussion draft. This branch contains an experimental adapter for an
> legacy GPU-GET library; it is not a production feature.
> The provider interface below is illustrative, not the implemented API.

## Motivation

Independent serving instances may each hold the same large, read-only Engram
table in host memory. A shared registered table could let multiple instances
fetch the rows needed by their own requests while keeping one backing copy. The
goal is to reduce duplicated table memory while keeping serving overhead small.

SGLang already supports GPU and CPU placements, per-node host sharing and
several lookup strategies. This proposal concerns sharing across independent
instances, potentially on different nodes; it should complement those existing
paths rather than replace their defaults. Remote access itself is not novel.

## Candidate integration

One possible integration is an opt-in table provider behind the existing Engram
configuration. The provider owns connection setup, row requests and completion;
the Engram layer retains hashing, TP/CP mapping, dequantization and output shape.
The native table provider remains the default. The following is pseudocode,
included to make the proposed boundary reviewable rather than prescribe an API:

```python
# Illustrative only: not implemented in this branch.
provider = open_remote_table_provider(config, layer_id, row_range)
provider.finish_load()  # establish views before CUDA graph capture
rows, scales = provider.fetch(indices, stream=current_stream())
# fetch must establish completion and visibility before these values are used.
output = dequantize_and_assemble(rows, scales, indices)
```

This needs a concrete decision on asynchronous completion and GPU visibility;
the pseudocode does not specify whether `fetch` waits, returns a completion
handle, or arranges a stream dependency. Configuration, error propagation,
checkpoint identity, table ownership, cancellation, graph lifetime, and
shutdown/drain behavior also need review. An inaccessible remote table must not
silently return incorrect data. Fallback policy needs an explicit design, since
readers that omit local weights cannot automatically fall back to a local table.

### Experimental adapter in this draft

Setting `SGLANG_EXPERIMENTAL_REMOTE_ENGRAM=1` and selecting
`--engram-config '{"placement":"cpu","replication":"none"}'` routes
TP-owned rows through the research adapter. The option is off by default.
The SGLang package contains the reader, owner service, and READ-plan code. It
requires the separately built experimental NIXL/GPUNETIO GPU-GET library from
the prior research runtime. It currently only supports 256-wide Engram layers 1 and 14.
The experimental path requires row264 storage, `ENGRAM_SHARED_PREFETCH=1`
and `ENGRAM_SHARED_READ_PLAN=1` for its prepared multi-QP READ plan.
These options require the matching native library and have not been qualified
for all graph, cancellation, or model configurations. The native NIXL GET
implementation remains a separate dependency.
The experimental runtime validates checkpoint identity and registered table
metadata. The following smoke and C8 measurements used this draft at commit
`a4edaf95a27657069ac32f417e0857e926a1cf62`, before its later main-branch
update; they are not measurements of the later implementation. With the experimental runtime,
one same-version H20 1P1D smoke ran
the real model with 16K input and produced 64 output tokens matching a prior
reference output exactly. The P used TP8/CP8, bounded replay disabled and the
remote reader; D used DP8, a native per-node host table and DSpark. Mooncake
handled P/D KV transfer. P recorded one KV transfer, and both P/D requests
returned successfully. This is a single functional smoke, not a performance,
concurrency or production qualification. That historical test used separately
provisioned reader and owner files; this draft now includes their Python code.

A subsequent paired C8 test used three rounds of 32 measured requests per arm
with 16K input and 64 output tokens on the same two H20 nodes. The Shared P
used 16 QPs and the experimental READ plan/prefetch; Native P used the
per-node local CPU table available at that tested revision. Both used the same D
process, Mooncake KV, model version, prompt sets, and bounded replay disabled. Median-of-round TTFT p50 was
5159.83 ms Native and 5204.95 ms Shared (+0.87%); output throughput was 94.33
and 93.52 tokens/s (-0.85%). All 96 paired input and output hashes matched,
with zero measured cache hits, retractions or client errors. P was restarted
between arms, Shared ran first, and no per-window NIC counter audit was made;
the sample is not a production or p99 qualification. The 1-QP path had a
5928.08 ms TTFT p50 in one earlier 32-request round; the combined READ plan and
prefetch changes improved that matching-input round by 12.44%, but individual
contributions and multi-run 1-QP variance were not isolated.

On the independently ported SGLang code at commit
`e9b0b3f9cca80a060292e3cd13fa575e63e9122d`, a separate H20 experiment
ran the real model in 1P1D with DSpark, TP8/CP8 prefill, DP8 decode and
Mooncake KV transfer. One 512-input/8-output request and repeated C8,
16K-input/64-output runs completed. Each measured round had 32 requests,
eight disjoint warmups and no measured prefix-cache hits. The same 32 outputs
matched across all reported rounds. With the same D process throughout,
the Native P TTFT p50 was 7.140, 7.736 and 8.743 seconds; Shared P was
7.810 seconds before those Native rounds and 7.534, 8.582 and 8.877 seconds
afterward. TPOT p50 was approximately 3.12 ms for both. These repeated rounds
reuse the same request set and show substantial time variation; they do not
establish a reliable percentage overhead, production tail latency, or a
comparison with the earlier measurements. The native peer-retirement
variant below was used for this experiment.

The later experimental DOCA 3.5 integration at SGLang commit
`7fde8d0ac3073a403fef0ef3be401ff7444c4f42` was also exercised on H20
1P1D, 16K input / 64 output, C8, TP8/CP8 prefill and DP8 decode with
DSpark. In three rounds per arm, 3.1 versus 3.5 median-of-round TTFT p50
was 5.162 versus 5.176 seconds; TPOT p50 was 3.124 versus 3.125 ms;
output throughput was 94.26 versus 94.04 tokens/s. Each round had eight
warmups and 32 measured requests, no measured prefix-cache hits, and all
96 paired output text hashes matched. The same 32 inputs were reused each
round; order was fixed (3.5 then 3.1) with one prefill-process launch per
arm. P node, SGLang commit, command and D process matched; owner GPU and
process, shim, NIXL and DOCA versions differed. D's Native Engram decode
graph executed; P prefill graph was disabled. This compares the two tested
runtime stacks, not isolated SDKs or production tail performance.

### Sharing the owner table with local decode

The experimental single-backing mode stores one row264 table in pinned CPU
memory on the owner node. Decode ranks map that same backing and gather rows
locally on their GPUs; prefill readers on other nodes fetch rows through
GPUNETIO. The decode ranks do not fetch their own table over RDMA or allocate a
second native table. Each local mapping registers the same host pages with CUDA;
it is not a separate table allocation.

Start the owner with `ENGRAM_SHARED_LOCAL_ACCESS=1` in its environment. Start
the decode engine with `SGLANG_EXPERIMENTAL_LOCAL_SHARED_ENGRAM=1`,
`ENGRAM_SHARED_READY` pointing to the owner's local ready file, and
`--engram-config '{"placement":"cpu","replication":"none"}'`. The owner
and decode processes must share a PID namespace and be able to access the
owner's `/proc/<pid>/fd` entry. The owner must outlive its readers. Configure
the prefill reader as described below, using the same owner's ready metadata.
Local and remote Engram modes are mutually exclusive within an engine. This is
an opt-in research path; GPU execution, lifecycle, failure recovery and
performance must be validated for the deployment before use.

### Starting the research owner and reader

The Python owner and reader are included in this PR. The native library is
separate: build the legacy DOCA 3.1 research snapshot following the [NIXL
reproducer](https://github.com/foraxe/nixl/tree/b5322e46f6e686ff16bdb3afb8a1a940a64e1163/research/vllm_engram_repro).
Use the matching NIXL core and GPUNETIO plugin on both nodes. This is not a
supported current-DOCA installation. Configure the local RDMA device, GID and
OOB interface as required by that reproducer, including the plugin search
path `NIXL_PLUGIN_DIR` on both owner and reader processes.

On the owner node, set `MODEL`, `SHIM`, `OWNER_HOST` and an empty private
`WORK` directory for your deployment, then run:

```bash
python -m sglang.srt.engram_share.rdma.owner \
  --checkpoint "$MODEL" --library "$SHIM" --owner-gpu 0 \
  --rdma-port 56001 --control-host 0.0.0.0 --advertise-host "$OWNER_HOST" \
  --control-port 56002 --ready-file "$WORK/owner.json" \
  --stop-file "$WORK/owner.stop" --max-pinned-gib 190 --storage-layout row264
```

The ready file contains registration metadata and remote memory keys. Transfer
it securely to the reader node and restrict the owner control network to
trusted peers; the research control channel is not authenticated or encrypted.
Choose unused ports and adjust the pinned-memory budget for the checkpoint.
Do not stop the owner while readers or their GPU work are active.

On the prefill node, set `MODEL`, `SHIM` and `OWNER_READY` to the matching
checkpoint, local native library, and securely copied ready file. The engine's
actual model path determines the checkpoint identity; it must be a local,
indexed safetensors checkpoint with bytes identical to the owner's checkpoint.
The owner and each reader hash every indexed shard at startup, which can add
substantial disk I/O and startup time for large models. Keep checkpoint files
immutable while the service is running. Online weight updates are rejected;
restart the engine and owner together to change checkpoint. The reader requires
the mmap safetensors loader (`auto` or `safetensors`), and supports CP sizes
1, 4 and 8. In addition
to the usual SGLang P/D configuration, set:

```bash
export SGLANG_EXPERIMENTAL_REMOTE_ENGRAM=1
export ENGRAM_SHARED_MODEL="$MODEL" ENGRAM_SHARED_LIBRARY="$SHIM"
export ENGRAM_SHARED_READY="$OWNER_READY" ENGRAM_SHARED_READER_PORT_BASE=56100
export ENGRAM_SHARED_PREFETCH=1 ENGRAM_SHARED_READ_PLAN=1
export ENGRAM_SHARED_READ_LANES=16 ENGRAM_SHARED_MAX_QUERIES=393216
export ENGRAM_SHARED_READ_GRAPH_SIZES=393216 ENGRAM_SHARED_DEFER_LAYER14=1
# Experimental: allocate an isolated QP and destination for decode graph GET.
export ENGRAM_SHARED_DECODE_GRAPH=1
# Optional: one RDMA device name per visible GPU, comma separated.
export ENGRAM_SHARED_READER_DEVICES="<gpu0-device>,<gpu1-device>,..."
```

Use `--engram-config '{"placement":"cpu","replication":"none"}'` for
this experimental path. Keep the decode node on its native Engram path. These
legacy `ENGRAM_SHARED_*` keys configure the research reader and native ABI;
their final configuration surface is part of the design discussion. To check
both checkpoint layers with one reader GPU, set `ENGRAM_SHARED_MAX_QUERIES=49152`
and `ENGRAM_SHARED_READ_GRAPH_SIZES=49152`, then run:

```bash
python test/manual/engram_remote_smoke.py
python test/manual/engram_remote_graph_smoke.py
```

For the experimental DOCA 3.5 device-GET route, the SGLang-specific native
ABI adapter and build script are in `python/sglang/srt/engram_share/rdma/native`.
They require the matching NIXL device-GET implementation under review in
[NIXL #2587](https://github.com/ai-dynamo/nixl/pull/2587), CUDA and DOCA 3.5;
see the build script's required environment variables. Use the matching NIXL
core and plugin on both nodes. The separate graph QP has its own arena and
completion state; the feature allocates an additional arena per reader. It is
opt-in. Two graph buckets with changing IDs and overlapping 16-lane prefetch
passed a synthetic component test. Concurrent graph replays on different
streams, failures, and Shared-Engram decode graph in a real P/D service have
not been qualified. Prefill-side graph capture is a separate limitation.
The default DOCA 3.5 adapter does not expose peer retirement. An experimental
owner-retirement build can set `ENGRAM_RESEARCH_PEER_RETIREMENT=1` when building
the adapter, with the matching experimental NIXL source, plugin and dependency
include paths. Enable `ENGRAM_SHARED_RETIRE_PEERS=1` on the owner only with that
matching plugin; startup rejects a missing capability. This is a research
interface, not the public NIXL API. Owner and readers must run the same control
protocol version. Graceful scheduler shutdown drains shared readers before
releasing their leases, with a longer scheduler exit window for this mode.
Failed readers and forced shutdown can still leave uncertain leases; automatic
crash recovery and safe in-flight fencing remain unproven.

After readers have stopped and drained, create the owner stop file and wait for
it to exit. Failed readers can leave orphan leases; this prototype intentionally
does not reclaim them on a timer. The older native backend can retain per-peer QP resources until owner exit.
As an optional research variant, a separately built [peer-retirement native
branch](https://github.com/foraxe/nixl/tree/f89cdaa2) supports normal drained
reader releases: build that native branch instead and set
`ENGRAM_SHARED_RETIRE_PEERS=1` on the owner before starting it. The owner
advertises this capability to its readers. A failed retirement retains the
lease for retry. Crashed readers remain uncertain; this is not automatic crash
recovery or a supported current-SDK interface.

## Research evidence and limits

An external research prototype, based on SGLang commit
`9a9ae36865e6c7c24fcbf3ce44a14b5e2945709a` and adapter commit
`c2ba4222787820e5d46069a163b49ded95a94aee`, used GPU-initiated READs from a
registered pinned-host table. It used an experimental NIXL/GPUNETIO path and
DOCA 3.1; it is not an implementation of this proposal on current main. Its
private TP/CP hooks and underlying NIXL interfaces need to be ported and
reviewed. NIXL device-GET discussion: https://github.com/ai-dynamo/nixl/issues/2543
and reference draft: https://github.com/ai-dynamo/nixl/pull/2544. The latter is
not a current-SDK runnable implementation.

In a paired H20 experiment, three independent TP8 prefill groups shared one
remote table while a TP8/DP8 decode group kept its native table. P/D KV
transport used Mooncake in both arms. With 16K input tokens, 64 output tokens,
client concurrency 24, DSPARK block size 5 and decode CUDA graphs, three rounds
of 96 measured requests per arm produced matching outputs for all 288 pairs.
The median of the three round medians was 1632.793 ms native versus 1647.930 ms
shared TTFT (+0.93%), and 834.304 versus 828.517 output tokens/s (-0.69%).
TTFT p99 was +2.68% and TPOT p90 +5.69%; per-round samples are small and these
are not production tail guarantees. All measured client requests completed;
prefill abort-submission counters were nonzero in both arms and their cause
remains unresolved.

The P-side logical table payload was three copies versus one shared copy, a
66.7% reduction. Including the unchanged D table, it was four copies versus two,
a 50% reduction. These are logical payload counts, not measured total RSS. The
owner still consumes pinned host memory and readers consume workspace. The
experiment used synthetic prompts and a debug router; it does not establish
production capacity or 1M-token performance. It does not compare against
GIN/NVSHMEM or the latest SGLang CPU/THP configuration.

## Questions for review

1. Should remote table access be expressed as an Engram provider, or integrated
   differently with the existing placement, replication and lookup options?
2. Where should table ownership, checkpoint verification and failures live when
   independent instances share a table?
3. Which completion, CUDA graph and lifecycle guarantees are required at the
   SGLang/NIXL boundary?
4. What would be the smallest useful functional test and reproducible benchmark
   for an initial implementation?

The intention is to discuss and agree on the design before converting this
draft into a ready-for-review implementation.
