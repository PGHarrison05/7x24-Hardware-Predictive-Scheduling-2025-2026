#!/usr/bin/env python3
import argparse
import os
import time
import math
import json
import random
from datetime import datetime

import numpy as np


def cpu_burn(intensity: float, duration: float):
    """
    Simulate CPU-heavy work.
    intensity: 0.0 to 1.0
    duration: seconds
    """
    end = time.time() + duration
    block = max(1000, int(200000 * intensity))

    while time.time() < end:
        x = np.random.rand(block)
        _ = np.sqrt(x * x + 0.12345).sum()

        # lower intensity = more sleep
        if intensity < 1.0:
            time.sleep((1.0 - intensity) * 0.02)


def mem_burn(mem_mb: int, duration: float):
    """
    Reserve memory and touch it repeatedly.
    """
    if mem_mb <= 0:
        time.sleep(duration)
        return

    size = mem_mb * 1024 * 1024 // 8  # float64 count
    arr = np.ones(size, dtype=np.float64)

    end = time.time() + duration
    while time.time() < end:
        arr *= 1.0000001
        arr /= 1.0000001
        time.sleep(0.1)


def io_burn(io_mb: int, duration: float, scratch_dir: str):
    """
    Simulate checkpoint/data I/O by writing and reading a temp file.
    """
    if io_mb <= 0:
        time.sleep(duration)
        return

    os.makedirs(scratch_dir, exist_ok=True)
    path = os.path.join(scratch_dir, f"io_test_{os.getpid()}.bin")

    chunk = os.urandom(1024 * 1024)  # 1 MB
    end = time.time() + duration

    try:
        while time.time() < end:
            with open(path, "wb") as f:
                for _ in range(io_mb):
                    f.write(chunk)

            with open(path, "rb") as f:
                while f.read(1024 * 1024):
                    pass
    finally:
        if os.path.exists(path):
            os.remove(path)


def fake_checkpoint(checkpoint_mb: int, scratch_dir: str):
    """
    Simulate checkpoint cost.
    """
    if checkpoint_mb <= 0:
        return 0.0

    os.makedirs(scratch_dir, exist_ok=True)
    path = os.path.join(scratch_dir, f"checkpoint_{os.getpid()}.ckpt")

    chunk = os.urandom(1024 * 1024)
    start = time.time()
    with open(path, "wb") as f:
        for _ in range(checkpoint_mb):
            f.write(chunk)
    elapsed = time.time() - start

    if os.path.exists(path):
        os.remove(path)

    return elapsed


def log_event(log_file: str, data: dict):
    with open(log_file, "a") as f:
        f.write(json.dumps(data) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Synthetic AI workload simulator")
    parser.add_argument("--job-name", type=str, default="sim_job")
    parser.add_argument("--flexibility", type=str, choices=["rigid", "semi", "flexible"], required=True)
    parser.add_argument("--length", type=str, choices=["short", "medium", "long"], required=True)
    parser.add_argument("--stress", type=str, choices=["low", "medium", "high"], required=True)
    parser.add_argument("--mem-mb", type=int, default=512)
    parser.add_argument("--io-mb", type=int, default=0)
    parser.add_argument("--checkpoint-mb", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Seconds between checkpoints, 0 = none")
    parser.add_argument("--scratch-dir", type=str, default="./scratch")
    parser.add_argument("--log-file", type=str, default="workload_log.jsonl")
    args = parser.parse_args()

    length_map = {
        "short": 60,
        "medium": 180,
        "long": 420
    }

    stress_map = {
        "low": 0.25,
        "medium": 0.60,
        "high": 0.95
    }

    total_runtime = length_map[args.length]
    cpu_intensity = stress_map[args.stress]

    start_ts = datetime.now().isoformat()

    metadata = {
        "timestamp": start_ts,
        "pid": os.getpid(),
        "job_name": args.job_name,
        "flexibility": args.flexibility,
        "length": args.length,
        "stress": args.stress,
        "mem_mb": args.mem_mb,
        "io_mb": args.io_mb,
        "checkpoint_mb": args.checkpoint_mb,
        "checkpoint_interval": args.checkpoint_interval
    }
    log_event(args.log_file, {"event": "job_start", **metadata})

    print(f"[START] {args.job_name}")
    print(f"  flexibility = {args.flexibility}")
    print(f"  length      = {args.length} ({total_runtime}s)")
    print(f"  stress      = {args.stress}")
    print(f"  mem_mb      = {args.mem_mb}")
    print(f"  io_mb       = {args.io_mb}")
    print(f"  checkpoint  = {args.checkpoint_mb} MB every {args.checkpoint_interval}s")

    start_time = time.time()
    last_checkpoint = start_time

    # simple mixed simulation loop
    while True:
        elapsed = time.time() - start_time
        if elapsed >= total_runtime:
            break

        step = min(10, total_runtime - elapsed)

        # CPU activity
        cpu_burn(cpu_intensity, step * 0.5)

        # memory pressure
        mem_burn(args.mem_mb, step * 0.2)

        # I/O activity
        io_burn(args.io_mb, step * 0.2, args.scratch_dir)

        # checkpoint if enabled
        now = time.time()
        if args.checkpoint_interval > 0 and (now - last_checkpoint) >= args.checkpoint_interval:
            ckpt_time = fake_checkpoint(args.checkpoint_mb, args.scratch_dir)
            log_event(args.log_file, {
                "event": "checkpoint",
                "job_name": args.job_name,
                "elapsed_sec": round(now - start_time, 2),
                "checkpoint_time_sec": round(ckpt_time, 3),
                "checkpoint_mb": args.checkpoint_mb
            })
            print(f"[CHECKPOINT] {args.job_name}: {ckpt_time:.2f}s")
            last_checkpoint = now

        # jitter to imitate variable behavior
        time.sleep(random.uniform(0.1, 0.5))

    total_elapsed = time.time() - start_time
    log_event(args.log_file, {
        "event": "job_end",
        "job_name": args.job_name,
        "runtime_sec": round(total_elapsed, 2)
    })
    print(f"[END] {args.job_name} runtime={total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
