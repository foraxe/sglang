# DSv4 TileKernels Router Progress

## 2026-04-26

### Baseline

- Source worktree: `/Users/nyx/projects/0explore/dsv4/sglang-dsv4-tilekernel-router`
- Branch: `dsv4-tilekernel-router-experiment`
- Base: `c8810af62 Fix request pool capacity for reserved slot`
- TileKernels dependency worktree on GB200 host: `/tmp/TileKernels-dsv4-router-dep`
- SGLang host copy on GB200 host: `/tmp/sglang-dsv4-tilekernel-router`

### Code Changes

- Added `SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE` in `python/sglang/srt/environ.py`.
- Added `tilekernel_dsv4_top2_sum_gate(...)` in `python/sglang/srt/layers/moe/topk.py`.
- Routed DSv4 `sqrtsoftplus` non-fused shared-expert top-k through TileKernels when the env flag is enabled.
- Added direct helper and `select_experts(...)` CUDA tests in `test/srt/test_dsv4_tilekernel_top2_gate.py`.

### Independent Review

Subagent `019dc9d6-6cc9-70f1-ba8f-952f30e1e98e` found a real integration bug: the first patch called TileKernels but then fell through into `biased_topk_impl(...)`, which was not imported in that branch. The review also flagged the need to preserve expert-location logical-to-physical mapping. Both points were addressed.

### Local Checks

```bash
python3 -m py_compile \
  python/sglang/srt/environ.py \
  python/sglang/srt/layers/moe/topk.py \
  test/srt/test_dsv4_tilekernel_top2_gate.py

git diff --check
```

Result: passed.

### GB200 Unit Test

Container: `dsv4-sglang-tilekernel-test`

```bash
cd /workspace/sglang
PYTHONPATH=/workspace/sglang/python:/workspace/TileKernels \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q test/srt/test_dsv4_tilekernel_top2_gate.py
```

Result:

```text
5 passed, 3 warnings in 13.61s
```

### First Live Smoke Failure

Launch succeeded through `/model_info`, then `/generate` failed after about 122 seconds with:

```text
UnboundLocalError: cannot access local variable 'biased_topk_impl' where it is not associated with a value
```

This confirmed the subagent review finding.

### Patched Live Smoke

Launch shape:

```bash
cd /workspace/sglang
export PYTHONPATH=/workspace/sglang/python:/workspace/TileKernels:
export SGLANG_APPLY_CONFIG_BACKUP=small
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
python3 -m sglang.launch_server \
  --model-path /data/models/DeepSeek-V4-Flash \
  --tp 1 \
  --port 30000 \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-flashinfer-autotune \
  --max-total-tokens 8192 \
  --max-running-requests 1 \
  --context-length 8192 \
  --json-model-override-args '{"num_hidden_layers": 4}'
```

Request:

```json
{"input_ids":[1,2,3,4],"sampling_params":{"max_new_tokens":1,"temperature":0}}
```

Result:

```text
HTTP 200
elapsed=126.32
prompt_tokens=4
completion_tokens=1
output_ids=[113899]
```

Response:

```json
{"text":" pagklasipika","output_ids":[113899],"meta_info":{"id":"3005cf162aed45989a1eba1cba2edd36","finish_reason":{"type":"length","length":1},"prompt_tokens":4,"weight_version":"default","total_retractions":0,"completion_tokens":1,"cached_tokens":0,"e2e_latency":126.02402925491333,"response_sent_to_client_ts":1777208360.671939}}
```

Artifacts:

- `/Users/nyx/projects/0explore/dsv4/artifacts/dsv4_tilekernel_router_sglang_smoke/dsv4_tilekernel_router_smoke.tgz`
- SHA256: `2e9d4b95cba5f00b0e20651d24cf297662873ab420641f2862239519229e469d`

### Cleanup

After the smoke, `dsv4-sglang-tilekernel-smoke` and `dsv4-sglang-tilekernel-test` were removed. GPU memory returned to 0 MiB on all four GB200 GPUs.

### TileLang Dependency Follow-Up

A compatibility matrix in the same DSv4 image found:

```text
native tilelang: 0.1.7.post3
tilelang 0.1.8: TileKernels imports, T.gemm has wg_wait, but TileKernels top2_sum_gate fails against current source
tilelang 0.1.9: TileKernels top2_sum_gate passes, but T.gemm no longer has wg_wait
```

A local TileKernels compatibility patch for 0.1.8 was rejected because it compiled but returned wrong expert IDs. The TileKernels branch was restored to its known-good state.

SGLang was patched instead in `python/sglang/srt/layers/mhc.py`:

- inspect `T.gemm` once at import
- call `T.gemm(..., wg_wait=0, ...)` when supported
- omit `wg_wait` when running against TileLang 0.1.9

Local checks:

```bash
python3 -m py_compile \
  python/sglang/srt/layers/mhc.py \
  python/sglang/srt/layers/moe/topk.py \
  test/srt/test_dsv4_tilekernel_top2_gate.py

git diff --check
```

Result: passed.

GB200 unit test with `tilelang==0.1.9`:

```text
5 passed, 3 warnings in 14.39s
```

MHC-enabled live smoke with `tilelang==0.1.9`, TileKernels router enabled, and no `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=0` workaround:

```text
HTTP 200
elapsed=132.33
prompt_tokens=4
completion_tokens=1
output_ids=[113899]
```

Artifact:

- `/Users/nyx/projects/0explore/dsv4/artifacts/dsv4_tilekernel_router_mhc_smoke/dsv4_tilekernel_router_mhc_smoke.tgz`
- SHA256: `e213ffe908530350e2427a19a6fd5203f9755570d50463195c0d3000986ae4e3`

### A/B Evidence

Clean A/B used port `31000` to avoid stale server collisions.

Common launch:

```bash
export SGLANG_APPLY_CONFIG_BACKUP=small
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
python3 -m sglang.launch_server \
  --model-path /data/models/DeepSeek-V4-Flash \
  --tp 1 \
  --port 31000 \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-flashinfer-autotune \
  --max-total-tokens 8192 \
  --max-running-requests 1 \
  --context-length 8192 \
  --json-model-override-args '{"num_hidden_layers": 4}'
```

TileKernels variant added:

```bash
export SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE=1
```

Results:

```text
baseline warm 4+1: HTTP 200, warm_elapsed_s=132.33
tilekernel warm 4+1: HTTP 200, warm_elapsed_s=138.33
baseline 4096+2: manually terminated, long_rc=143, no response file
tilekernel 4096+2: manually terminated, long_rc=143, no response file
```

The long requests stayed connected to the server but did not produce a scheduler prefill log or GPU work before manual termination. That shared behavior means the long-input stall is not introduced by the TileKernels router hook.

Artifact:

- `/Users/nyx/projects/0explore/dsv4/artifacts/dsv4_tilekernel_router_ab_clean/dsv4_tilekernel_router_ab_clean.tgz`
- SHA256: `1e0ba2394c5a0139c35664520041ffde4fa30b3c654ad84f423663d2647f2cc3`

Cleanup after A/B returned all four GPUs to 0 MiB.

### Fused-Shared Tail-Kernel Contract Test

Added an opt-in fused-shared tail wrapper for the DSv4 router shape:

```text
router logits: (N, 256)
routed topk: 6
fused shared experts: 1
total output topk: 7
groups/topk_groups: 8/8
scoring: sqrtsoftplus
```

Files changed:

- `python/sglang/srt/environ.py`
- `python/sglang/srt/layers/moe/topk.py`
- `test/srt/test_dsv4_tilekernel_top2_gate.py`

New env flag:

```bash
SGLANG_OPT_USE_TILEKERNEL_DSV4_FUSED_SHARED_TOP2_GATE=1
```

Local checks:

```bash
python3 -m py_compile \
  python/sglang/srt/environ.py \
  python/sglang/srt/layers/moe/topk.py \
  test/srt/test_dsv4_tilekernel_top2_gate.py

git diff --check
```

Result: passed.

GB200 CUDA unit test:

```bash
cd /workspace/sglang
PYTHONPATH=/workspace/sglang/python:/workspace/TileKernels \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q test/srt/test_dsv4_tilekernel_top2_gate.py
```

Result:

```text
11 passed, 3 warnings in 14.40s
```

Subagent `019dca54-b6af-7bb1-85d3-2f3ab25fdff3` reviewed the fused-shared contract and agreed that scale/id/order semantics match the target DSv4 shape. It flagged a real EPLB edge: a routed-only logical-to-physical dispatch table may not contain the appended shared id `256`. The patch now maps only routed columns and leaves the appended shared id unchanged; the CUDA test includes a regression for that case.

### 2026-04-26 TP4 A/B Preparation

Checked whether the new fused-shared tail-kernel path is live for the current GB200 DSv4 FP4 launch.

Result: not live for the current model/backend shape. `DeepseekV4ForCausalLM.determine_num_fused_shared_experts()` disables shared expert fusion for 2604B and 2604 FP4 because shared/routed experts have different constraints. Therefore the current useful live flag remains:

```bash
SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE=1
```

Added:

```bash
scripts/run_gb200_tilekernel_tp4_ab.sh
```

The script prepares the full TP=4 baseline vs TileKernels A/B profile for `4096+2` with `/start_profile` and `/stop_profile`, using `--max-prefill-tokens 8192` to avoid mixing this router experiment with the known FlashMLA mixed-prefill crash.

Local script check:

```bash
bash -n scripts/run_gb200_tilekernel_tp4_ab.sh
```

Result: passed.

Live execution blocked:

```text
kubectl get pod dsv4-gb200-hostdebug -o wide
Unable to connect to the server: remote error: tls: expired certificate

client cert:
notAfter=Apr 26 15:24:58 2026 GMT
```

Direct SSH fallback also failed:

```text
ssh: Could not resolve hostname gpuidi14aaf1029.idi1
```

Next live command after kubeconfig refresh:

```bash
cd /Users/nyx/projects/0explore/dsv4/sglang-dsv4-tilekernel-router
./scripts/run_gb200_tilekernel_tp4_ab.sh
```
