# Current-main review gate, 2026-11-03

SGLang source: `3b86606a5961fc9de61c6ffab1b7017a819814c7`, after merging upstream main `96301ec47875e31827813a0307e4616ab63378ef`. Owner remains vLLM `46706ee5e5ade3f622b6f606bbd05b84b06a9b96`. Same four-GB200 host, original DeepSeek-V4.1-Flash checkpoint, and isolated runtime described in the parent evidence README. GPUs 0-3 were leased for the serving run and released afterward.

## Code and test review

- Registered CPU tests use `CustomTestCase` and `register_cpu_ci(..., suite="base-a-test-cpu")`, following the current SGLang test skill and registration validator.
- Loader test uses a real mixed safetensors shard and `DefaultModelLoader._get_weights_iterator` with published external config. A `safe_open.get_tensor` spy observes only the small projection; both external weight and scale are skipped. A CPU-placement control materializes all three. This tests the production config-to-iterator path, including the negative branch.
- Lease test checks unmap, detach, socket close order and confirms that a failed unmap sends no detach. The attachment registry now lives in process runtime resources. Shutdown imports the external code only when that placement is configured.
- Startup weight-load overlap was inspected: its allowlist admits only Llama/Qwen and rejects this DeepSeek architecture before prefetching. The ordinary external iterator disables shard prefetch and multithread loading.
- Focused external tests: 4/4 passed. Existing Engram history tests: 2/2 passed. All applicable pre-commit hooks passed. A synthetic separate-owner FABRIC import, gather, and changed-ID CUDA Graph replay passed with a 198-byte mapping.

## Serving recheck

One full-model SGLang P TP2 + D TP2 run used native NIXL, 13 input tokens, 32 forced output tokens, and eager P/D execution. It completed one warmup and three retained routed requests. The three retained output hashes matched each other. The owner logged eight attaches and eight protocol detaches; owner, P and D top-level processes exited zero. This recheck did not run a CPU comparator or 8K performance workload. See `pd_shared_short.json` and compressed raw logs.

Shutdown caveat: the prefill top-level process logged `Schedulers still alive 15s after ShutdownReq, killing them before they released` after the owner had already logged protocol detaches. Thus this run validates lease detach, but not fully graceful prefill scheduler exit. The top-level zero exit alone must not be interpreted as clean scheduler teardown. Decode had no such notice. This may be current-main shutdown behavior; it needs a separate investigation before claiming a clean lifecycle gate.

Cross-host NVL72 import and serving remain untested because only one host was available. No stable speedup is claimed; the earlier 8K local screen is in the parent evidence snapshot.
