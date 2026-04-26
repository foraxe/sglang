#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG_PATH="${KUBECONFIG_PATH:-$HOME/.kube/gb200_kubeconfig_daily.yaml}"
POD="${POD:-dsv4-gb200-hostdebug}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_SGLANG="${REMOTE_SGLANG:-/tmp/sglang-dsv4-tilekernel-router}"
REMOTE_TILEKERNELS="${REMOTE_TILEKERNELS:-/tmp/TileKernels-dsv4-router-dep}"
IMAGE="${IMAGE:-docker.io/lmsysorg/sglang:deepseek-v4-grace-blackwell}"
MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-V4-Flash}"
REMOTE_OUT="${REMOTE_OUT:-/tmp/dsv4_tilekernel_router_tp4_ab_noprofile}"
LOCAL_OUT="${LOCAL_OUT:-$LOCAL_REPO/artifacts/dsv4_tilekernel_router_tp4_ab_noprofile}"
PROFILE="${PROFILE:-0}"

kubectl_cmd=(kubectl --kubeconfig "$KUBECONFIG_PATH")
mkdir -p "$LOCAL_OUT"

echo "[preflight] kube cert"
"${kubectl_cmd[@]}" config view --minify --raw -o jsonpath='{.users[0].user.client-certificate-data}' \
  | base64 -d | openssl x509 -noout -subject -enddate

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

echo "[remote] run TP4 A/B PROFILE=$PROFILE"
"${kubectl_cmd[@]}" exec -i "$POD" -- chroot /host env \
  "REMOTE_OUT=$REMOTE_OUT" \
  "REMOTE_SGLANG=$REMOTE_SGLANG" \
  "REMOTE_TILEKERNELS=$REMOTE_TILEKERNELS" \
  "IMAGE=$IMAGE" \
  "MODEL_PATH=$MODEL_PATH" \
  "PROFILE=$PROFILE" \
  bash -s <<'REMOTE'
set -euo pipefail
OUT="$REMOTE_OUT"
SGLANG_DIR="$REMOTE_SGLANG"
TILEKERNELS_DIR="$REMOTE_TILEKERNELS"
mkdir -p "$OUT"
rm -rf "$OUT"/*

run_variant() {
  local variant="$1"
  local tile_flag="$2"
  local port="$3"
  local container="dsv4-tilekernel-ab-${variant}"
  local variant_dir="$OUT/$variant"
  mkdir -p "$variant_dir"
  echo "[variant] start $variant tile=$tile_flag port=$port"

  nerdctl rm -f "$container" >/dev/null 2>&1 || true
  nerdctl run --insecure-registry --name "$container" \
    --privileged -td --shm-size=1024g --ulimit memlock=-1 \
    --cap-add=SYS_PTRACE --gpus all --net host --ipc=host --uts=host \
    --security-opt seccomp=unconfined \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    --runtime /usr/bin/nvidia-container-runtime \
    -v "$SGLANG_DIR:/workspace/sglang" \
    -v "$TILEKERNELS_DIR:/workspace/TileKernels" \
    -v "$OUT:/workspace/out" \
    -v /data:/data \
    --entrypoint bash "$IMAGE" -lc "sleep infinity" >/dev/null

  cleanup() {
    nerdctl exec "$container" bash -lc "pkill -f 'sglang.launch_server' || true" >/dev/null 2>&1 || true
    nerdctl rm -f "$container" >/dev/null 2>&1 || true
  }
  trap cleanup RETURN

  nerdctl exec "$container" bash -lc "python3 -m pip install -q --force-reinstall --no-deps tilelang==0.1.9"
  cat > "$variant_dir/client.py" <<'PY'
import json
import os
import sys
import time
import urllib.request

port = int(sys.argv[1])
out = sys.argv[2]
profile = os.environ.get("PROFILE", "0") == "1"
payload = {
    "input_ids": list(range(1, 4097)),
    "sampling_params": {"max_new_tokens": 2, "temperature": 0},
}

def post(path, body, timeout=1200):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode()

for _ in range(900):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/model_info", timeout=2) as resp:
            if resp.status == 200:
                break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("server not ready")

if profile:
    post("/start_profile", {
        "activities": ["CPU", "GPU"],
        "with_stack": True,
        "record_shapes": True,
        "profile_prefix": os.path.basename(os.path.dirname(out)),
    })

t0 = time.time()
status, body = post("/generate", payload)
elapsed = time.time() - t0

if profile:
    post("/stop_profile", {})

with open(out, "w") as f:
    json.dump({"status": status, "elapsed": elapsed, "body": json.loads(body)}, f, indent=2)
PY

  local server_env="cd /workspace/sglang && \
export PYTHONPATH=/workspace/sglang/python:/workspace/TileKernels: && \
export SGLANG_APPLY_CONFIG_BACKUP=auto && \
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 && \
export SGLANG_OPT_USE_TILEKERNEL_DSV4_TOP2_GATE=$tile_flag && \
python3 -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --tp 4 \
  --port $port \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-flashinfer-autotune \
  --max-total-tokens 262144 \
  --max-running-requests 32 \
  --max-prefill-tokens 8192 \
  --context-length 8192"

  nerdctl exec -d "$container" bash -lc "$server_env > /workspace/out/$variant/server.log 2>&1"
  nerdctl exec -e "PROFILE=$PROFILE" "$container" bash -lc \
    "python3 /workspace/out/$variant/client.py $port /workspace/out/$variant/result.json" \
    > "$variant_dir/client.stdout" 2> "$variant_dir/client.stderr"
  cleanup
  trap - RETURN
  echo "[variant] done $variant"
}

run_variant baseline 0 32100
run_variant tilekernel 1 32101
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$OUT/nvidia_smi_after_cleanup.csv"
tar -C "$OUT" -zcf /tmp/dsv4_tilekernel_router_tp4_ab.tgz .
sha256sum /tmp/dsv4_tilekernel_router_tp4_ab.tgz > /tmp/dsv4_tilekernel_router_tp4_ab.tgz.sha256
REMOTE

echo "[copy] archive to local"
"${kubectl_cmd[@]}" exec "$POD" -- chroot /host cat /tmp/dsv4_tilekernel_router_tp4_ab.tgz > "$LOCAL_OUT/dsv4_tilekernel_router_tp4_ab.tgz"
"${kubectl_cmd[@]}" exec "$POD" -- chroot /host cat /tmp/dsv4_tilekernel_router_tp4_ab.tgz.sha256 > "$LOCAL_OUT/dsv4_tilekernel_router_tp4_ab.tgz.sha256"
rm -rf "$LOCAL_OUT/extracted"
mkdir -p "$LOCAL_OUT/extracted"
tar -C "$LOCAL_OUT/extracted" -xzf "$LOCAL_OUT/dsv4_tilekernel_router_tp4_ab.tgz"
echo "[done] $LOCAL_OUT"
