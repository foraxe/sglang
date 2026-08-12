#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
import urllib.request


PROMPT = "Measured virtual memory benchmark sentence. " * 500


def post(url, payload, *, streaming):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    wall_start = time.time_ns()
    mono_start = time.perf_counter_ns()
    first_byte_ns = None
    first_token_ns = None
    chunks = []
    with urllib.request.urlopen(request, timeout=300) as response:
        if streaming:
            while True:
                raw = response.readline()
                if not raw:
                    break
                if first_byte_ns is None:
                    first_byte_ns = time.perf_counter_ns()
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[len(b"data: ") :].strip()
                if payload == b"[DONE]":
                    continue
                item = json.loads(payload)
                chunks.append(item)
                if item.get("text") and first_token_ns is None:
                    first_token_ns = time.perf_counter_ns()
            result = chunks[-1]
        else:
            raw = response.read()
            first_byte_ns = time.perf_counter_ns()
            result = json.loads(raw)
    mono_end = time.perf_counter_ns()
    text = result.get("text", "")
    return {
        "wall_start_ns": wall_start,
        "first_byte_ms": (first_byte_ns - mono_start) / 1e6,
        "ttft_ms": None
        if first_token_ns is None
        else (first_token_ns - mono_start) / 1e6,
        "e2e_ms": (mono_end - mono_start) / 1e6,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "meta_info": result.get("meta_info"),
        "wire_chunks": chunks if streaming else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/generate"
    common = {"text": PROMPT, "sampling_params": {"temperature": 0}}
    rows = {
        "backend": args.backend,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "prompt_bytes": len(PROMPT.encode()),
        "stream_warmup": post(
            url,
            {**common, "sampling_params": {"temperature": 0, "max_new_tokens": 16}, "stream": True},
            streaming=True,
        ),
        "stream_measured": [],
        "prefill_measured": [],
    }
    for _ in range(3):
        rows["stream_measured"].append(
            post(
                url,
                {**common, "sampling_params": {"temperature": 0, "max_new_tokens": 16}, "stream": True},
                streaming=True,
            )
        )
    for _ in range(3):
        rows["prefill_measured"].append(
            post(
                url,
                {**common, "sampling_params": {"temperature": 0, "max_new_tokens": 1}},
                streaming=False,
            )
        )
    with open(args.output, "w") as output:
        json.dump(rows, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
