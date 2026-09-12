#!/usr/bin/env python3

import csv
import math
import os
import random
import re
import signal
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ============================================================
# USER SETTINGS
# ============================================================

# -----------------------------
# Experiment
# -----------------------------
EXPERIMENT_MINUTES = 4
TICK_SECONDS = 5
SEED = 42

# -----------------------------
# GPUs
# -----------------------------
GPU_IDS = [0, 1, 2]
GPU_BURN_PATH = "./gpu_burn"

# -----------------------------
# Job generation
# -----------------------------
AVG_JOBS_PER_MINUTE = 2.0
TRAINING_RATIO = 0.60
INFERENCE_RATIO = 0.40

TRAINING_DURATION_RANGE_S = (90, 300)
TRAINING_FLEXIBLE_PROB = 0.80
TRAINING_DEFER_SLACK_RANGE_S = (120, 600)

INFERENCE_DURATION_RANGE_S = (30, 120)
INFERENCE_DUTY_CYCLE_CHOICES = [0.20, 0.30, 0.40, 0.50]
INFERENCE_BURST_ON_S = 3

# -----------------------------
# Policy
# one of: baseline, pack, thermal, temporal, full
# baseline = first free
# pack     = rack/pack/stack (fill low index first)
# thermal  = avoid hot conditions
# temporal = defer flexible training during expensive hours
# full     = combine temporal + thermal + pack
# -----------------------------
POLICY = "full"

# -----------------------------
# Thermal policy
# -----------------------------
GPU_TEMP_BLOCK_C = 72.0
GPU_TEMP_RESUME_C = 65.0

# -----------------------------
# Temporal policy
# Replace later with real NYISO / TOU data if desired
# -----------------------------
EXPENSIVE_HOURS = {14, 15, 16, 17, 18, 19, 20}
CHEAP_RATE_PER_KWH = 0.12
EXPENSIVE_RATE_PER_KWH = 0.28

# -----------------------------
# Shelly / PUE settings
# BASE Shelly = main system/node power
# AUX Shelly  = optional extra cooling / auxiliary load added to numerator
# -----------------------------
USE_BASE_SHELLY = True
BASE_SHELLY_IP = "149.61.249.0"

USE_AUX_SHELLY = True
AUX_SHELLY_IP = "149.61.201.189"

BASE_SHELLY_URL = f"http://{BASE_SHELLY_IP}/rpc/Shelly.GetStatus"
AUX_SHELLY_URL = f"http://{AUX_SHELLY_IP}/rpc/Shelly.GetStatus"

ALLOW_GPU_ONLY_FALLBACK = False  # if CPU power unavailable, denominator can fallback to GPU-only

# -----------------------------
# Output
# -----------------------------
OUTPUT_ROOT = Path("scheduler_with_pue_outputs")
RUN_NAME = f"{POLICY}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = OUTPUT_ROOT / RUN_NAME

JOBS_CSV = RUN_DIR / "jobs_generated.csv"
DISPATCH_CSV = RUN_DIR / "dispatch_log.csv"
TELEMETRY_CSV = RUN_DIR / "telemetry_log.csv"
PUE_CSV = RUN_DIR / "pue_log.csv"
SUMMARY_CSV = RUN_DIR / "summary.csv"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Job:
    job_id: str
    job_type: str
    arrival_epoch: float
    requested_duration_s: int
    flexible: bool
    defer_slack_s: int
    duty_cycle: Optional[float] = None
    assigned_gpu: Optional[int] = None
    status: str = "queued"
    scheduled_start_epoch: Optional[float] = None
    actual_start_epoch: Optional[float] = None
    actual_end_epoch: Optional[float] = None
    delay_s: float = 0.0
    retries: int = 0


@dataclass
class RunningJob:
    job: Job
    process: subprocess.Popen
    wrapper_path: Optional[Path]


# ============================================================
# BASIC HELPERS
# ============================================================

def now_epoch() -> float:
    return time.time()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hour_from_epoch(ts: float) -> int:
    return datetime.fromtimestamp(ts).hour


def current_rate_per_kwh(ts: float) -> float:
    return EXPENSIVE_RATE_PER_KWH if hour_from_epoch(ts) in EXPENSIVE_HOURS else CHEAP_RATE_PER_KWH


def is_expensive_hour(ts: float) -> bool:
    return hour_from_epoch(ts) in EXPENSIVE_HOURS


def ensure_dirs():
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def poisson_sample(lmbda: float) -> int:
    L = math.exp(-lmbda)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return max(0, k - 1)


def run_command(cmd: List[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip()
    except Exception:
        return ""


# ============================================================
# SHELLY / PUE
# ============================================================

def get_shelly_power_from_url(url: str) -> Tuple[Optional[float], Optional[dict], Optional[str]]:
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()

        if "switch:0" in data and isinstance(data["switch:0"], dict):
            apower = data["switch:0"].get("apower")
            if apower is not None:
                return float(apower), data, None

        total = 0.0
        found = False
        for key, value in data.items():
            if key.startswith("switch:") and isinstance(value, dict):
                apower = value.get("apower")
                if apower is not None:
                    total += float(apower)
                    found = True

        if found:
            return total, data, None

        return None, data, "No apower field found"

    except Exception as e:
        return None, None, str(e)


def get_base_and_aux_power() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str], Optional[str]]:
    base_power = None
    aux_power = None
    base_err = None
    aux_err = None

    if USE_BASE_SHELLY:
        base_power, _, base_err = get_shelly_power_from_url(BASE_SHELLY_URL)

    if USE_AUX_SHELLY:
        aux_power, _, aux_err = get_shelly_power_from_url(AUX_SHELLY_URL)

    total = None
    if base_power is not None or aux_power is not None:
        total = (base_power or 0.0) + (aux_power or 0.0)

    return base_power, aux_power, total, base_err, aux_err


# ============================================================
# CPU / GPU TELEMETRY
# ============================================================

def get_cpu_temp_and_power_from_sensors() -> Tuple[Optional[float], Optional[float], Optional[str]]:
    out = run_command(["sensors"])

    cpu_temp = None
    cpu_power = None
    cpu_power_source = None

    for line in out.splitlines():
        line_lower = line.lower().strip()

        if (
            "package id 0:" in line_lower
            or line_lower.startswith("tctl:")
            or line_lower.startswith("tdie:")
            or "cpu temp:" in line_lower
        ):
            match = re.search(r"([+-]?\d+(\.\d+)?)°c", line_lower)
            if match and cpu_temp is None:
                cpu_temp = float(match.group(1))

        if any(token in line_lower for token in ["ppt:", "package power", "power1:"]):
            match = re.search(r"([+-]?\d+(\.\d+)?)\s*w", line_lower)
            if match:
                val = float(match.group(1))
                if cpu_power is None or val > cpu_power:
                    cpu_power = val
                    if "ppt:" in line_lower:
                        cpu_power_source = "sensors:PPT"
                    elif "package power" in line_lower:
                        cpu_power_source = "sensors:package_power"
                    else:
                        cpu_power_source = "sensors:power1"

    return cpu_temp, cpu_power, cpu_power_source


def find_rapl_energy_file() -> Optional[str]:
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return None

    preferred_words = ["package", "pkg", "core", "psys"]

    for root, dirs, files in os.walk(base):
        if "energy_uj" in files:
            name_file = os.path.join(root, "name")
            if os.path.exists(name_file):
                try:
                    with open(name_file, "r") as f:
                        name = f.read().strip().lower()
                    if any(word in name for word in preferred_words):
                        return os.path.join(root, "energy_uj")
                except Exception:
                    pass

    for root, dirs, files in os.walk(base):
        if "energy_uj" in files:
            return os.path.join(root, "energy_uj")

    return None


def read_energy_uj(path: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def estimate_cpu_power_w_from_rapl(energy_file: Optional[str], prev_energy: Optional[int], prev_time: Optional[float]):
    if not energy_file:
        return None, prev_energy, prev_time

    now_energy = read_energy_uj(energy_file)
    now_time = time.time()

    if now_energy is None or prev_energy is None or prev_time is None:
        return None, now_energy, now_time

    delta_energy_uj = now_energy - prev_energy
    delta_time = now_time - prev_time

    if delta_energy_uj < 0 or delta_time <= 0:
        return None, now_energy, now_time

    power_w = (delta_energy_uj / 1_000_000.0) / delta_time
    return power_w, now_energy, now_time


def get_cpu_power(energy_file, prev_energy, prev_time):
    cpu_power_rapl, new_prev_energy, new_prev_time = estimate_cpu_power_w_from_rapl(
        energy_file, prev_energy, prev_time
    )
    if cpu_power_rapl is not None:
        return cpu_power_rapl, new_prev_energy, new_prev_time, "rapl"

    _, cpu_power_sensors, cpu_power_source = get_cpu_temp_and_power_from_sensors()
    if cpu_power_sensors is not None:
        return cpu_power_sensors, new_prev_energy, new_prev_time, cpu_power_source

    return None, new_prev_energy, new_prev_time, None


def get_gpu_telemetry() -> Dict[int, Dict]:
    out = run_command([
        "nvidia-smi",
        "--query-gpu=index,temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ])

    data = {}
    if not out:
        return data

    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpu_idx = int(parts[0])
            data[gpu_idx] = {
                "temp_c": float(parts[1]),
                "util_percent": float(parts[2]),
                "power_w": float(parts[3]),
                "mem_used_mb": float(parts[4]),
                "mem_total_mb": float(parts[5]),
            }
        except ValueError:
            continue

    return data


def total_gpu_power_w(telemetry: Dict[int, Dict]) -> float:
    return sum(v.get("power_w", 0.0) for v in telemetry.values())


# ============================================================
# JOB GENERATION
# ============================================================

def choose_job_type() -> str:
    total = TRAINING_RATIO + INFERENCE_RATIO
    r = random.random() * total
    return "training" if r < TRAINING_RATIO else "inference"


def generate_jobs_for_minute(minute_index: int, experiment_start_epoch: float, counter_start: int) -> List[Job]:
    minute_start = experiment_start_epoch + minute_index * 60
    count = poisson_sample(AVG_JOBS_PER_MINUTE)

    jobs = []
    for i in range(count):
        job_type = choose_job_type()
        arrival_offset_s = random.randint(0, 59)
        arrival_epoch = minute_start + arrival_offset_s

        if job_type == "training":
            duration_s = random.randint(*TRAINING_DURATION_RANGE_S)
            flexible = random.random() < TRAINING_FLEXIBLE_PROB
            slack_s = random.randint(*TRAINING_DEFER_SLACK_RANGE_S) if flexible else 0
            duty_cycle = None
            job_id = f"TRAIN_{counter_start + i:05d}"
        else:
            duration_s = random.randint(*INFERENCE_DURATION_RANGE_S)
            flexible = False
            slack_s = 0
            duty_cycle = random.choice(INFERENCE_DUTY_CYCLE_CHOICES)
            job_id = f"INFER_{counter_start + i:05d}"

        jobs.append(Job(
            job_id=job_id,
            job_type=job_type,
            arrival_epoch=arrival_epoch,
            requested_duration_s=duration_s,
            flexible=flexible,
            defer_slack_s=slack_s,
            duty_cycle=duty_cycle,
            scheduled_start_epoch=arrival_epoch
        ))

    return jobs


def generate_all_jobs(experiment_start_epoch: float) -> List[Job]:
    jobs = []
    counter = 1
    for minute_idx in range(EXPERIMENT_MINUTES):
        batch = generate_jobs_for_minute(minute_idx, experiment_start_epoch, counter)
        jobs.extend(batch)
        counter += len(batch)

    jobs.sort(key=lambda j: j.arrival_epoch)
    return jobs


# ============================================================
# WORKLOAD EXECUTION
# ============================================================

def build_training_command(gpu_id: int, duration_s: int) -> List[str]:
    return [
        "bash",
        "-lc",
        f"CUDA_VISIBLE_DEVICES={gpu_id} {GPU_BURN_PATH} {duration_s}"
    ]


def build_inference_wrapper(path: Path, gpu_id: int, duration_s: int, duty_cycle: float):
    on_s = INFERENCE_BURST_ON_S
    off_s = max(1, int(round(on_s * (1.0 - duty_cycle) / duty_cycle)))

    script = f"""#!/usr/bin/env bash
set -euo pipefail
END_TIME=$(( $(date +%s) + {duration_s} ))

while [ "$(date +%s)" -lt "$END_TIME" ]; do
    REM=$(( END_TIME - $(date +%s) ))
    if [ "$REM" -le 0 ]; then
        break
    fi

    RUN_FOR={on_s}
    if [ "$REM" -lt "$RUN_FOR" ]; then
        RUN_FOR="$REM"
    fi

    CUDA_VISIBLE_DEVICES={gpu_id} {GPU_BURN_PATH} "$RUN_FOR" || true

    REM=$(( END_TIME - $(date +%s) ))
    if [ "$REM" -le 0 ]; then
        break
    fi

    SLEEP_FOR={off_s}
    if [ "$REM" -lt "$SLEEP_FOR" ]; then
        SLEEP_FOR="$REM"
    fi

    sleep "$SLEEP_FOR"
done
"""
    path.write_text(script)
    path.chmod(0o755)


def launch_job(job: Job, gpu_id: int) -> RunningJob:
    wrapper = None

    if job.job_type == "training":
        cmd = build_training_command(gpu_id, job.requested_duration_s)
    else:
        wrapper = RUN_DIR / f"{job.job_id}_wrapper.sh"
        build_inference_wrapper(wrapper, gpu_id, job.requested_duration_s, job.duty_cycle or 0.3)
        cmd = ["bash", str(wrapper)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )

    job.assigned_gpu = gpu_id
    job.actual_start_epoch = now_epoch()
    job.status = "running"
    job.delay_s = job.actual_start_epoch - job.arrival_epoch

    return RunningJob(job=job, process=proc, wrapper_path=wrapper)


def stop_process_tree(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


# ============================================================
# POLICY LOGIC
# ============================================================

def free_gpu_ids(running_jobs: List[RunningJob]) -> List[int]:
    busy = set()
    for rj in running_jobs:
        if rj.process.poll() is None and rj.job.assigned_gpu is not None:
            busy.add(rj.job.assigned_gpu)
    return [g for g in GPU_IDS if g not in busy]


def hot_system(telemetry: Dict[int, Dict]) -> bool:
    return any(telemetry.get(g, {}).get("temp_c", 0.0) >= GPU_TEMP_BLOCK_C for g in GPU_IDS)


def pick_gpu_baseline(free_gpus: List[int], telemetry: Dict[int, Dict]) -> Optional[int]:
    return free_gpus[0] if free_gpus else None


def pick_gpu_pack(free_gpus: List[int], telemetry: Dict[int, Dict]) -> Optional[int]:
    # rack/pack/stack: always fill lowest-index free GPU first
    return min(free_gpus) if free_gpus else None


def pick_gpu_coolest(free_gpus: List[int], telemetry: Dict[int, Dict]) -> Optional[int]:
    if not free_gpus:
        return None
    ranked = sorted(
        free_gpus,
        key=lambda g: (
            telemetry.get(g, {}).get("temp_c", 9999),
            telemetry.get(g, {}).get("util_percent", 9999),
            telemetry.get(g, {}).get("power_w", 9999),
            g
        )
    )
    return ranked[0]


def select_gpu(policy: str, free_gpus: List[int], telemetry: Dict[int, Dict]) -> Optional[int]:
    if not free_gpus:
        return None

    if policy == "baseline":
        return pick_gpu_baseline(free_gpus, telemetry)

    if policy == "pack":
        return pick_gpu_pack(free_gpus, telemetry)

    if policy == "thermal":
        return pick_gpu_coolest(free_gpus, telemetry)

    if policy == "temporal":
        return pick_gpu_pack(free_gpus, telemetry)

    if policy == "full":
        # full: use packing by default, but if multiple are free and one is much cooler, choose cooler
        coolest = pick_gpu_coolest(free_gpus, telemetry)
        packed = pick_gpu_pack(free_gpus, telemetry)
        if coolest is None:
            return packed
        packed_temp = telemetry.get(packed, {}).get("temp_c", 9999) if packed is not None else 9999
        coolest_temp = telemetry.get(coolest, {}).get("temp_c", 9999)
        if coolest_temp + 4 < packed_temp:
            return coolest
        return packed

    return pick_gpu_baseline(free_gpus, telemetry)


def should_defer(job: Job, policy: str, telemetry: Dict[int, Dict], now_ts: float) -> Tuple[bool, str]:
    if not job.flexible:
        return False, ""

    latest_allowed = job.arrival_epoch + job.defer_slack_s
    if now_ts >= latest_allowed:
        return False, ""

    if policy in ("temporal", "full"):
        if is_expensive_hour(now_ts):
            return True, "expensive_hour"

    if policy in ("thermal", "full"):
        if hot_system(telemetry):
            return True, "thermal_hot"

    return False, ""


# ============================================================
# LOGGING
# ============================================================

def init_csv(path: Path, fieldnames: List[str]):
    exists = path.exists()
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer


def write_jobs_csv(jobs: List[Job]):
    with open(JOBS_CSV, "w", newline="") as f:
        fieldnames = [
            "job_id", "job_type", "arrival_epoch", "requested_duration_s", "flexible",
            "defer_slack_s", "duty_cycle", "assigned_gpu", "status",
            "scheduled_start_epoch", "actual_start_epoch", "actual_end_epoch",
            "delay_s", "retries"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow(asdict(job))


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dirs()
    random.seed(SEED)

    if not Path(GPU_BURN_PATH).exists():
        raise FileNotFoundError(f"gpu_burn not found at {GPU_BURN_PATH}")

    energy_file = find_rapl_energy_file()
    prev_energy = read_energy_uj(energy_file) if energy_file else None
    prev_time = time.time() if energy_file else None

    experiment_start = now_epoch()
    experiment_end = experiment_start + EXPERIMENT_MINUTES * 60

    jobs = generate_all_jobs(experiment_start)
    write_jobs_csv(jobs)

    dispatch_f, dispatch_w = init_csv(DISPATCH_CSV, [
        "timestamp", "event", "job_id", "job_type", "gpu_id", "status",
        "delay_s", "requested_duration_s", "flexible", "defer_slack_s",
        "reason", "policy"
    ])

    telemetry_f, telemetry_w = init_csv(TELEMETRY_CSV, [
        "timestamp", "gpu_id", "temp_c", "util_percent", "power_w",
        "mem_used_mb", "mem_total_mb", "policy"
    ])

    pue_f, pue_w = init_csv(PUE_CSV, [
        "timestamp",
        "base_shelly_power_w",
        "aux_shelly_power_w",
        "total_numerator_power_w",
        "cpu_power_w",
        "cpu_power_source",
        "cpu_temp_c",
        "total_gpu_power_w",
        "compute_power_w",
        "pue_proxy",
        "running_avg_pue_proxy",
        "energy_based_avg_pue_proxy",
        "energy_price_per_kwh",
        "instantaneous_it_cost_per_hour",
        "base_shelly_error",
        "aux_shelly_error",
        "policy",
    ])

    running_jobs: List[RunningJob] = []
    queue: List[Job] = []
    submitted_idx = 0

    launched_jobs = 0
    completed_jobs = 0
    deferred_events = 0
    cumulative_delay_s = 0.0
    thermal_defers = 0
    temporal_defers = 0

    pue_sum = 0.0
    pue_count = 0
    wall_energy_sum_ws = 0.0
    compute_energy_sum_ws = 0.0
    wall_cost_sum = 0.0

    peak_temp_observed = 0.0
    peak_total_gpu_power = 0.0

    print("=" * 72)
    print("GPU SCHEDULER WITH PUE LOGGER")
    print("=" * 72)
    print(f"Policy:         {POLICY}")
    print(f"Run dir:        {RUN_DIR}")
    print(f"Experiment min: {EXPERIMENT_MINUTES}")
    print()

    try:
        while now_epoch() < experiment_end or running_jobs or submitted_idx < len(jobs) or queue:
            now_ts = now_epoch()

            # --------------------------------------------------
            # Telemetry snapshot
            # --------------------------------------------------
            telemetry = get_gpu_telemetry()
            cpu_temp, _, _ = get_cpu_temp_and_power_from_sensors()
            cpu_power, prev_energy, prev_time, cpu_power_source = get_cpu_power(
                energy_file, prev_energy, prev_time
            )

            total_gpu_power = total_gpu_power_w(telemetry)
            peak_total_gpu_power = max(peak_total_gpu_power, total_gpu_power)

            for gpu_id in GPU_IDS:
                row = telemetry.get(gpu_id, {})
                temp_c = row.get("temp_c")
                if temp_c is not None:
                    peak_temp_observed = max(peak_temp_observed, temp_c)

                telemetry_w.writerow({
                    "timestamp": now_str(),
                    "gpu_id": gpu_id,
                    "temp_c": temp_c,
                    "util_percent": row.get("util_percent"),
                    "power_w": row.get("power_w"),
                    "mem_used_mb": row.get("mem_used_mb"),
                    "mem_total_mb": row.get("mem_total_mb"),
                    "policy": POLICY,
                })
            telemetry_f.flush()

            # --------------------------------------------------
            # PUE snapshot
            # --------------------------------------------------
            base_power, aux_power, total_numerator_power, base_err, aux_err = get_base_and_aux_power()

            compute_power = None
            if cpu_power is not None:
                compute_power = cpu_power + total_gpu_power
            elif ALLOW_GPU_ONLY_FALLBACK and total_gpu_power > 0:
                compute_power = total_gpu_power

            pue_proxy = None
            if total_numerator_power is not None and compute_power is not None and compute_power > 0:
                pue_proxy = total_numerator_power / compute_power
                pue_sum += pue_proxy
                pue_count += 1

                wall_energy_sum_ws += total_numerator_power * TICK_SECONDS
                compute_energy_sum_ws += compute_power * TICK_SECONDS

                rate = current_rate_per_kwh(now_ts)
                wall_cost_sum += ((total_numerator_power / 1000.0) * (TICK_SECONDS / 3600.0)) * rate
            else:
                rate = current_rate_per_kwh(now_ts)

            running_avg_pue = (pue_sum / pue_count) if pue_count > 0 else None
            energy_based_avg_pue = (wall_energy_sum_ws / compute_energy_sum_ws) if compute_energy_sum_ws > 0 else None
            instantaneous_it_cost_per_hour = ((total_numerator_power / 1000.0) * rate) if total_numerator_power is not None else None

            pue_w.writerow({
                "timestamp": now_str(),
                "base_shelly_power_w": base_power,
                "aux_shelly_power_w": aux_power,
                "total_numerator_power_w": total_numerator_power,
                "cpu_power_w": cpu_power,
                "cpu_power_source": cpu_power_source,
                "cpu_temp_c": cpu_temp,
                "total_gpu_power_w": total_gpu_power,
                "compute_power_w": compute_power,
                "pue_proxy": pue_proxy,
                "running_avg_pue_proxy": running_avg_pue,
                "energy_based_avg_pue_proxy": energy_based_avg_pue,
                "energy_price_per_kwh": rate,
                "instantaneous_it_cost_per_hour": instantaneous_it_cost_per_hour,
                "base_shelly_error": base_err,
                "aux_shelly_error": aux_err,
                "policy": POLICY,
            })
            pue_f.flush()

            # --------------------------------------------------
            # Move arrived jobs into queue
            # --------------------------------------------------
            while submitted_idx < len(jobs) and jobs[submitted_idx].arrival_epoch <= now_ts:
                queue.append(jobs[submitted_idx])
                dispatch_w.writerow({
                    "timestamp": now_str(),
                    "event": "ARRIVE",
                    "job_id": jobs[submitted_idx].job_id,
                    "job_type": jobs[submitted_idx].job_type,
                    "gpu_id": "",
                    "status": jobs[submitted_idx].status,
                    "delay_s": "",
                    "requested_duration_s": jobs[submitted_idx].requested_duration_s,
                    "flexible": jobs[submitted_idx].flexible,
                    "defer_slack_s": jobs[submitted_idx].defer_slack_s,
                    "reason": "",
                    "policy": POLICY,
                })
                submitted_idx += 1
            dispatch_f.flush()

            # --------------------------------------------------
            # Clean up finished jobs
            # --------------------------------------------------
            still_running = []
            for rj in running_jobs:
                rc = rj.process.poll()
                if rc is None:
                    still_running.append(rj)
                else:
                    rj.job.actual_end_epoch = now_epoch()
                    rj.job.status = "finished"
                    completed_jobs += 1

                    dispatch_w.writerow({
                        "timestamp": now_str(),
                        "event": "FINISH",
                        "job_id": rj.job.job_id,
                        "job_type": rj.job.job_type,
                        "gpu_id": rj.job.assigned_gpu,
                        "status": rj.job.status,
                        "delay_s": round(rj.job.delay_s, 3),
                        "requested_duration_s": rj.job.requested_duration_s,
                        "flexible": rj.job.flexible,
                        "defer_slack_s": rj.job.defer_slack_s,
                        "reason": "",
                        "policy": POLICY,
                    })

                    if rj.wrapper_path and rj.wrapper_path.exists():
                        try:
                            rj.wrapper_path.unlink()
                        except Exception:
                            pass

            running_jobs = still_running
            dispatch_f.flush()

            # --------------------------------------------------
            # Scheduling pass
            # --------------------------------------------------
            queue.sort(key=lambda j: (j.arrival_epoch, 0 if j.job_type == "inference" else 1))
            free_gpus = free_gpu_ids(running_jobs)

            for job in list(queue):
                if not free_gpus:
                    break

                defer, reason = should_defer(job, POLICY, telemetry, now_ts)
                if defer:
                    job.status = "deferred"
                    job.retries += 1
                    deferred_events += 1

                    if reason == "thermal_hot":
                        thermal_defers += 1
                    if reason == "expensive_hour":
                        temporal_defers += 1

                    dispatch_w.writerow({
                        "timestamp": now_str(),
                        "event": "DEFER",
                        "job_id": job.job_id,
                        "job_type": job.job_type,
                        "gpu_id": "",
                        "status": job.status,
                        "delay_s": round(now_ts - job.arrival_epoch, 3),
                        "requested_duration_s": job.requested_duration_s,
                        "flexible": job.flexible,
                        "defer_slack_s": job.defer_slack_s,
                        "reason": reason,
                        "policy": POLICY,
                    })
                    continue

                gpu_id = select_gpu(POLICY, free_gpus, telemetry)
                if gpu_id is None:
                    break

                rj = launch_job(job, gpu_id)
                running_jobs.append(rj)

                launched_jobs += 1
                cumulative_delay_s += job.delay_s

                dispatch_w.writerow({
                    "timestamp": now_str(),
                    "event": "START",
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "gpu_id": gpu_id,
                    "status": job.status,
                    "delay_s": round(job.delay_s, 3),
                    "requested_duration_s": job.requested_duration_s,
                    "flexible": job.flexible,
                    "defer_slack_s": job.defer_slack_s,
                    "reason": POLICY,
                    "policy": POLICY,
                })

                queue.remove(job)
                free_gpus.remove(gpu_id)

            dispatch_f.flush()

            active = ", ".join(f"{rj.job.job_id}@GPU{rj.job.assigned_gpu}" for rj in running_jobs) or "none"
            print(
                f"{now_str()} | active={active} | queue={len(queue)} | "
                f"gpu_total={total_gpu_power:.1f} W | wall={total_numerator_power} W | pue={pue_proxy}"
            )

            time.sleep(TICK_SECONDS)

    except KeyboardInterrupt:
        print("\nInterrupted, stopping jobs...")

    finally:
        for rj in running_jobs:
            stop_process_tree(rj.process)
            if rj.wrapper_path and rj.wrapper_path.exists():
                try:
                    rj.wrapper_path.unlink()
                except Exception:
                    pass
            if rj.job.status == "running":
                rj.job.status = "terminated"
                rj.job.actual_end_epoch = now_epoch()

        write_jobs_csv(jobs)

        avg_delay_s = cumulative_delay_s / launched_jobs if launched_jobs > 0 else 0.0
        queue_remaining = len([j for j in jobs if j.status in ("queued", "deferred")])
        training_count = len([j for j in jobs if j.job_type == "training"])
        inference_count = len([j for j in jobs if j.job_type == "inference"])
        energy_based_avg_pue = (wall_energy_sum_ws / compute_energy_sum_ws) if compute_energy_sum_ws > 0 else None

        total_wall_energy_kwh = wall_energy_sum_ws / 3_600_000.0
        total_compute_energy_kwh = compute_energy_sum_ws / 3_600_000.0

        with open(SUMMARY_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()

            rows = [
                {"metric": "policy", "value": POLICY},
                {"metric": "experiment_minutes", "value": EXPERIMENT_MINUTES},
                {"metric": "jobs_generated_total", "value": len(jobs)},
                {"metric": "training_jobs_generated", "value": training_count},
                {"metric": "inference_jobs_generated", "value": inference_count},
                {"metric": "jobs_launched", "value": launched_jobs},
                {"metric": "jobs_completed", "value": completed_jobs},
                {"metric": "jobs_remaining_in_queue", "value": queue_remaining},
                {"metric": "defer_events_total", "value": deferred_events},
                {"metric": "thermal_defers", "value": thermal_defers},
                {"metric": "temporal_defers", "value": temporal_defers},
                {"metric": "avg_launch_delay_s", "value": round(avg_delay_s, 3)},
                {"metric": "peak_gpu_temp_c", "value": round(peak_temp_observed, 3)},
                {"metric": "peak_total_gpu_power_w", "value": round(peak_total_gpu_power, 3)},
                {"metric": "total_wall_energy_kwh", "value": round(total_wall_energy_kwh, 6)},
                {"metric": "total_compute_energy_kwh", "value": round(total_compute_energy_kwh, 6)},
                {"metric": "estimated_wall_energy_cost_$", "value": round(wall_cost_sum, 6)},
                {"metric": "energy_based_avg_pue_proxy", "value": round(energy_based_avg_pue, 6) if energy_based_avg_pue is not None else ""},
            ]

            for row in rows:
                writer.writerow(row)

        dispatch_f.close()
        telemetry_f.close()
        pue_f.close()

        print("\nRun finished.")
        print(f"Run dir:        {RUN_DIR}")
        print(f"Jobs CSV:       {JOBS_CSV}")
        print(f"Dispatch CSV:   {DISPATCH_CSV}")
        print(f"Telemetry CSV:  {TELEMETRY_CSV}")
        print(f"PUE CSV:        {PUE_CSV}")
        print(f"Summary CSV:    {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
