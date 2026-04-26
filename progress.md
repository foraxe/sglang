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

