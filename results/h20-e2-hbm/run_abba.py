#!/usr/bin/env python3
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request


BASE = "/home/yunzhi.nyx/h20_team/sglang-cotensor-dwdp"
SGLANG = f"{BASE}/sglang"
MODEL = "/home/yunzhi.nyx/h20_team/models/gpt-oss-20b"
OUT = f"{BASE}/results/hbm-abba"
ORDER = ["native", "cotensor", "cotensor", "native"] * 5


def gpu_raw():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [list(map(int, line.split(", "))) for line in output.splitlines()[:2]]


def request(port):
    payload = json.dumps(
        {
            "text": "HBM phase attribution fixed prompt. " * 500,
            "sampling_params": {"temperature": 0, "max_new_tokens": 16},
        }
    ).encode()
    start = time.perf_counter_ns()
    with urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        ),
        timeout=300,
    ) as response:
        result = json.load(response)
    return {
        "e2e_ms": (time.perf_counter_ns() - start) / 1e6,
        "output_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
        "meta_info": result["meta_info"],
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for index, backend in enumerate(ORDER):
        port = 32000 + index
        path = f"{OUT}/{index:02d}-{backend}"
        os.makedirs(path, exist_ok=True)
        pre = gpu_raw()
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="0,1",
            PYTHONPATH=f"{SGLANG}/python",
            SGLANG_DWDP_HBM_PROBE="1",
        )
        log = open(f"{path}/server.log", "w")
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL,
            "--tp-size",
            "2",
            "--dwdp-size",
            "2",
            "--dwdp-vmm-backend",
            backend,
            "--trust-remote-code",
            "--disable-flashinfer-autotune",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.80",
            "--random-seed",
            "12345",
            "--skip-server-warmup",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=SGLANG,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited rc={proc.returncode}")
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1
                    ).read()
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("server readiness timeout")
            warmup = request(port)
            steady = gpu_raw()
            row = {
                "index": index,
                "block": index // 4,
                "position": index % 4,
                "backend": backend,
                "pre_gpu": pre,
                "steady_gpu": steady,
                "warmup": warmup,
            }
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
        for _ in range(30):
            post = gpu_raw()
            if all(gpu[1] <= 1 for gpu in post):
                break
            time.sleep(1)
        row["post_gpu"] = post
        rows.append(row)
        with open(f"{OUT}/rows.json", "w") as output:
            json.dump(rows, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
