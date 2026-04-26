#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG_PATH="${KUBECONFIG_PATH:-$HOME/.kube/gb200_kubeconfig_daily.yaml}"
POD="${POD:-dsv4-gb200-hostdebug}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_SGLANG="${REMOTE_SGLANG:-/tmp/sglang-dsv4-tilekernel-router}"
REMOTE_TILEKERNELS="${REMOTE_TILEKERNELS:-/tmp/TileKernels-dsv4-router-dep}"
IMAGE="${IMAGE:-docker.io/lmsysorg/sglang:deepseek-v4-grace-blackwell}"
MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-V4-Flash}"
REMOTE_OUT="${REMOTE_OUT:-/tmp/dsv4_tilekernel_router_tp4_ab}"
LOCAL_OUT="${LOCAL_OUT:-$LOCAL_REPO/artifacts/dsv4_tilekernel_router_tp4_ab}"

mkdir -p "$LOCAL_OUT"

kubectl_cmd=(kubectl --kubeconfig "$KUBECONFIG_PATH")

echo "[preflight] kube cert"
if ! "${kubectl_cmd[@]}" config view --minify --raw -o jsonpath='{.users[0].user.client-certificate-data}' \
  | base64 -d | openssl x509 -noout -subject -enddate; then
  echo "[error] cannot read kube client certificate from $KUBECONFIG_PATH" >&2
  exit 1
fi

echo "[preflight] pod and GPUs"
"${kubectl_cmd[@]}" get pod "$POD" -o wide
"${kubectl_cmd[@]}" exec "$POD" -- chroot /host bash -lc \
  'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits'

echo "[sync] focused files to host copy"
tar -C "$LOCAL_REPO" -cf - \
  python/sglang/srt/environ.py \
  python/sglang/srt/layers/moe/topk.py \
  python/sglang/srt/layers/mhc.py \
  test/srt/test_dsv4_tilekernel_top2_gate.py \
  scripts/run_gb200_tilekernel_tp4_ab.sh \
| "${kubectl_cmd[@]}" exec -i "$POD" -- chroot /host tar -C "$REMOTE_SGLANG" -xf -

echo "[remote] run A/B"
"${kubectl_cmd[@]}" exec "$POD" -- chroot /host bash -s <<REMOTE
set -euo pipefail
OUT="$REMOTE_OUT"
SGLANG_DIR="$REMOTE_SGLANG"
TILEKERNELS_DIR="$REMOTE_TILEKERNELS"
IMAGE="$IMAGE"
MODEL_PATH="$MODEL_PATH"
mkdir -p "\$OUT"
rm -rf "\$OUT"/*

cat > "\$OUT/run_one_variant.sh" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
variant="\$1"
tile_flag="\$2"
port="\$3"
OUT="\$4"
SGLANG_DIR="\$5"
TILEKERNELS_DIR="\$6"
IMAGE="\$7"
MODEL_PATH="\$8"
container="dsv4-tilekernel-tp4-\${variant}"
variant_dir="\$OUT/\$variant"
mkdir -p "\$variant_dir"
nerdctl rm -f "\$container" >/dev/null 2>&1 || true
nerdctl run --insecure-registry --name "\$container" \
  --privileged -td --shm-size=1024g --ulimit memlock=-1 \
  --cap-add=SYS_PTRACE \
  --gpus all --net host --ipc=host --uts=host \
  --security-opt seccomp=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --runtime /usr/bin/nvidia-container-runtime \
  -v "\$SGLANG_DIR:/workspace/sglang" \
  -v "\$TILEKERNELS_DIR:/workspace/TileKernels" \
  -v "\$OUT:/workspace/out" \
  --entrypoint bash "\$IMAGE" -lc "sleep infinity" >/dev/null
cleanup() {
  nerdctl exec "\$container" bash -lc "pkill -f 'sglang.launch_server' || true" >/dev/null 2>&1 || true
  nerdctl rm -f "\$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

nerdctl exec "\$container" bash -lc "python3 -m pip install -q --force-reinstall --no-deps tilelang==0.1.9"
cat > "\$variant_dir/client.py" <<'PY'
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
out = sys.argv[2]
payload = {
    "input_ids": list(range(1, 4097)),
    "sampling_params": {"max_new_tokens": 2, "temperature": 0},
}

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.status, resp.read().decode()

for _ in range(600):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/model_info", timeout=2) as resp:
            if resp.status == 200:
                break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("server not ready")

post("/flush_cache", {})
t0 = time.time()
post("/start_profile", {"activities": ["CPU", "CUDA"], "with_stack": True, "record_shapes": True})
status, body = post("/generate", payload)
elapsed = time.time() - t0
post("/stop_profile", {})
with open(out, "w") as f:
    json.dump({"status": status, "elapsed": elapsed, "body": json.loads(body)}, f, indent=2)
PY

server_env="cd /workspace/sglang && \
export PYTHONPATH=/workspace/sglang/python:/workspace/TileKernels: && \
export SGLANG_APPLY_CONFIG_BACKUP=auto && \
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 && \
export SGLANG_PROFILE_WITH_STACK=True && \
export SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE=\$tile_flag && \
python3 -m sglang.launch_server \
  --model-path \$MODEL_PATH \
  --tp 4 \
  --port \$port \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-flashinfer-autotune \
  --max-total-tokens 262144 \
  --max-running-requests 32 \
  --max-prefill-tokens 8192 \
  --context-length 8192"

nerdctl exec -d "\$container" bash -lc "\$server_env > /workspace/out/\$variant/server.log 2>&1"
nerdctl exec "\$container" bash -lc "python3 /workspace/out/\$variant/client.py \$port /workspace/out/\$variant/result.json" \
  > "\$variant_dir/client.stdout" 2> "\$variant_dir/client.stderr"
nerdctl exec "\$container" bash -lc "find /workspace/sglang -name '*chrome_trace*.json' -o -name '*trace*.json' -o -name '*.pt.trace.json' 2>/dev/null | head -50" \
  > "\$variant_dir/trace_files.txt" || true
nerdctl exec "\$container" bash -lc "find /workspace/sglang -name '*chrome_trace*.json' -o -name '*trace*.json' -o -name '*.pt.trace.json' 2>/dev/null | while read f; do cp \"\$f\" /workspace/out/\$variant/; done" || true
tar -C "\$variant_dir" -zcf "\$OUT/\${variant}_tp4_4k2.compress.json.gz" . >/dev/null 2>&1 || true
INNER
chmod +x "\$OUT/run_one_variant.sh"

bash "\$OUT/run_one_variant.sh" baseline 0 32000 "\$OUT" "\$SGLANG_DIR" "\$TILEKERNELS_DIR" "\$IMAGE" "\$MODEL_PATH"
bash "\$OUT/run_one_variant.sh" tilekernel 1 32001 "\$OUT" "\$SGLANG_DIR" "\$TILEKERNELS_DIR" "\$IMAGE" "\$MODEL_PATH"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "\$OUT/nvidia_smi_after_cleanup.csv"
tar -C "\$OUT" -zcf "\$OUT/dsv4_tilekernel_router_tp4_ab.tgz" .
sha256sum "\$OUT/dsv4_tilekernel_router_tp4_ab.tgz" > "\$OUT/dsv4_tilekernel_router_tp4_ab.tgz.sha256"
REMOTE

echo "[copy] archive to local"
"${kubectl_cmd[@]}" cp "$POD:$REMOTE_OUT/dsv4_tilekernel_router_tp4_ab.tgz" "$LOCAL_OUT/dsv4_tilekernel_router_tp4_ab.tgz"
"${kubectl_cmd[@]}" cp "$POD:$REMOTE_OUT/dsv4_tilekernel_router_tp4_ab.tgz.sha256" "$LOCAL_OUT/dsv4_tilekernel_router_tp4_ab.tgz.sha256"
"${kubectl_cmd[@]}" cp "$POD:$REMOTE_OUT/nvidia_smi_after_cleanup.csv" "$LOCAL_OUT/nvidia_smi_after_cleanup.csv"
echo "[done] $LOCAL_OUT"
