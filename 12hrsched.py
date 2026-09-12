#!/usr/bin/env python3

import csv
import math
import os
import random
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import matplotlib

HAS_DISPLAY = False
try:
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    HAS_DISPLAY = True
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_DISPLAY = False


# ============================================================
# USER SETTINGS
# ============================================================

# ---- total test duration ----
TOTAL_TEST_MINUTES = 720          # 12 hours
BASELINE_FRACTION = 0.50          # 6 hours baseline, 6 hours scheduled

# ---- tick / sampling ----
TICK_SECONDS = 5
SEED = 42

# ---- GPU system ----
GPU_IDS = [0, 1, 2]
GPU_BURN_PATH = "/home/paul/gpu-burn/gpu_burn"

# ---- workload mix ----
AVG_JOBS_PER_MINUTE = 2.2
TRAINING_RATIO = 0.85
INFERENCE_RATIO = 0.15

TRAINING_DURATION_RANGE_S = (300, 1800)
TRAINING_FLEXIBLE_PROB = 0.90
TRAINING_DEFER_SLACK_RANGE_S = (600, 5400)

INFERENCE_DURATION_RANGE_S = (20, 60)
INFERENCE_DUTY_CYCLE_CHOICES = [0.20, 0.30]
INFERENCE_BURST_ON_S = 3

# ---- thermal policy ----
GPU_TEMP_BLOCK_C = 58.0
GPU_TEMP_ALERT_C = 55.0

# ---- compressed peak window inside EACH phase ----
PEAK_START_FRAC = 0.20
PEAK_END_FRAC = 0.65

CHEAP_RATE_PER_KWH = 0.10
PEAK_RATE_PER_KWH = 0.35

# ---- launch throttling ----
MAX_NEW_STARTS_PER_TICK_BASELINE = 3
MAX_NEW_STARTS_PER_TICK_SCHEDULED = 2

# ---- Shelly / PUE ----
USE_BASE_SHELLY = True
BASE_SHELLY_IP = "149.61.237.244"

USE_AUX_SHELLY = True
AUX_SHELLY_IP = "149.61.201.189"

BASE_SHELLY_URL = f"http://{BASE_SHELLY_IP}/rpc/Shelly.GetStatus"
AUX_SHELLY_URL = f"http://{AUX_SHELLY_IP}/rpc/Shelly.GetStatus"

ALLOW_GPU_ONLY_FALLBACK = False

# ---- scaling assumptions ----
SCALING_NODE_COUNT = 1000
ANNUALIZED_HOURS = 24 * 365

# ---- output ----
OUTPUT_ROOT = Path("/home/paul/gpu-burn/demo_compare_outputs")
RUN_NAME = f"demo_compare_12h_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = OUTPUT_ROOT / RUN_NAME
PLOTS_DIR = RUN_DIR / "plots"

PHASE_SUMMARY_CSV = RUN_DIR / "phase_summary.csv"
TIME_SERIES_CSV = RUN_DIR / "time_series.csv"
EVENTS_CSV = RUN_DIR / "events.csv"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Job:
    job_id: str
    phase: str
    job_type: str
    arrival_offset_s: int
    requested_duration_s: int
    flexible: bool
    defer_slack_s: int
    duty_cycle: Optional[float] = None


@dataclass
class RuntimeJob:
    spec: Job
    process: subprocess.Popen
    assigned_gpu: int
    wrapper_path: Optional[Path]
    arrival_abs: float
    start_abs: float
    end_expected_abs: float


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def save_plot(name: str):
    out = PLOTS_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    if HAS_DISPLAY:
        plt.show()
    plt.close()
    print(f"Saved plot: {out}")


# ============================================================
# POWER / TELEMETRY
# ============================================================

def get_shelly_power_from_url(url: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()

        if "switch:0" in data and isinstance(data["switch:0"], dict):
            apower = data["switch:0"].get("apower")
            if apower is not None:
                return float(apower), None

        total = 0.0
        found = False
        for key, value in data.items():
            if key.startswith("switch:") and isinstance(value, dict):
                apower = value.get("apower")
                if apower is not None:
                    total += float(apower)
                    found = True

        if found:
            return total, None

        return None, "No apower field found"
    except Exception as e:
        return None, str(e)

def get_wall_power():
    base_power = None
    aux_power = None
    base_err = None
    aux_err = None

    if USE_BASE_SHELLY:
        base_power, base_err = get_shelly_power_from_url(BASE_SHELLY_URL)
    if USE_AUX_SHELLY:
        aux_power, aux_err = get_shelly_power_from_url(AUX_SHELLY_URL)

    total = None
    if base_power is not None or aux_power is not None:
        total = (base_power or 0.0) + (aux_power or 0.0)

    return base_power, aux_power, total, base_err, aux_err

def get_cpu_temp_and_power_from_sensors():
    out = run_command(["sensors"])
    cpu_temp = None
    cpu_power = None
    cpu_power_source = None

    for line in out.splitlines():
        ll = line.lower().strip()

        if ("package id 0:" in ll) or ll.startswith("tctl:") or ll.startswith("tdie:") or ("cpu temp:" in ll):
            match = re.search(r"([+-]?\d+(\.\d+)?)°c", ll)
            if match and cpu_temp is None:
                cpu_temp = float(match.group(1))

        if any(token in ll for token in ["ppt:", "package power", "power1:"]):
            match = re.search(r"([+-]?\d+(\.\d+)?)\s*w", ll)
            if match:
                val = float(match.group(1))
                if cpu_power is None or val > cpu_power:
                    cpu_power = val
                    if "ppt:" in ll:
                        cpu_power_source = "sensors:PPT"
                    elif "package power" in ll:
                        cpu_power_source = "sensors:package_power"
                    else:
                        cpu_power_source = "sensors:power1"

    return cpu_temp, cpu_power, cpu_power_source

def find_rapl_energy_file():
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return None

    preferred_words = ["package", "pkg", "core", "psys"]
    for root, _, files in os.walk(base):
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

    for root, _, files in os.walk(base):
        if "energy_uj" in files:
            return os.path.join(root, "energy_uj")
    return None

def read_energy_uj(path):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

def estimate_cpu_power_w_from_rapl(energy_file, prev_energy, prev_time):
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

def get_gpu_telemetry():
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
            idx = int(parts[0])
            data[idx] = {
                "temp_c": float(parts[1]),
                "util_percent": float(parts[2]),
                "power_w": float(parts[3]),
                "mem_used_mb": float(parts[4]),
                "mem_total_mb": float(parts[5]),
            }
        except ValueError:
            continue
    return data


# ============================================================
# JOB GENERATION
# ============================================================

def choose_job_type():
    total = TRAINING_RATIO + INFERENCE_RATIO
    r = random.random() * total
    return "training" if r < TRAINING_RATIO else "inference"

def generate_phase_jobs(phase_name: str, phase_seconds: int):
    jobs = []
    minutes = max(1, int(math.ceil(phase_seconds / 60)))
    counter = 1

    for minute_idx in range(minutes):
        count = poisson_sample(AVG_JOBS_PER_MINUTE)
        minute_start = minute_idx * 60

        for _ in range(count):
            job_type = choose_job_type()
            arrival_offset_s = min(phase_seconds - 1, minute_start + random.randint(0, 59))

            if job_type == "training":
                duration_s = random.randint(*TRAINING_DURATION_RANGE_S)
                flexible = random.random() < TRAINING_FLEXIBLE_PROB
                slack_s = random.randint(*TRAINING_DEFER_SLACK_RANGE_S) if flexible else 0
                duty_cycle = None
                job_id = f"{phase_name.upper()}_TRAIN_{counter:05d}"
            else:
                duration_s = random.randint(*INFERENCE_DURATION_RANGE_S)
                flexible = False
                slack_s = 0
                duty_cycle = random.choice(INFERENCE_DUTY_CYCLE_CHOICES)
                job_id = f"{phase_name.upper()}_INFER_{counter:05d}"

            jobs.append(Job(
                job_id=job_id,
                phase=phase_name,
                job_type=job_type,
                arrival_offset_s=arrival_offset_s,
                requested_duration_s=duration_s,
                flexible=flexible,
                defer_slack_s=slack_s,
                duty_cycle=duty_cycle
            ))
            counter += 1

    jobs.sort(key=lambda j: j.arrival_offset_s)
    return jobs


# ============================================================
# EXECUTION
# ============================================================

def build_training_command(gpu_id: int, duration_s: int):
    # hotter than the original version
    return ["bash", "-lc", f"CUDA_VISIBLE_DEVICES={gpu_id} {GPU_BURN_PATH} -d {duration_s}"]

def build_inference_wrapper(path: Path, gpu_id: int, duration_s: int, duty_cycle: float):
    on_s = INFERENCE_BURST_ON_S
    off_s = max(1, int(round(on_s * (1.0 - duty_cycle) / duty_cycle)))
    text = f"""#!/usr/bin/env bash
set -euo pipefail
END_TIME=$(( $(date +%s) + {duration_s} ))
while [ "$(date +%s)" -lt "$END_TIME" ]; do
    REM=$(( END_TIME - $(date +%s) ))
    if [ "$REM" -le 0 ]; then break; fi
    RUN_FOR={on_s}
    if [ "$REM" -lt "$RUN_FOR" ]; then RUN_FOR="$REM"; fi
    CUDA_VISIBLE_DEVICES={gpu_id} {GPU_BURN_PATH} "$RUN_FOR" || true
    REM=$(( END_TIME - $(date +%s) ))
    if [ "$REM" -le 0 ]; then break; fi
    SLEEP_FOR={off_s}
    if [ "$REM" -lt "$SLEEP_FOR" ]; then SLEEP_FOR="$REM"; fi
    sleep "$SLEEP_FOR"
done
"""
    path.write_text(text)
    path.chmod(0o755)

def launch_job(spec: Job, gpu_id: int, arrival_abs: float):
    wrapper = None
    if spec.job_type == "training":
        cmd = build_training_command(gpu_id, spec.requested_duration_s)
    else:
        wrapper = RUN_DIR / f"{spec.job_id}_wrapper.sh"
        build_inference_wrapper(wrapper, gpu_id, spec.requested_duration_s, spec.duty_cycle or 0.3)
        cmd = ["bash", str(wrapper)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )

    start_abs = time.time()
    return RuntimeJob(
        spec=spec,
        process=proc,
        assigned_gpu=gpu_id,
        wrapper_path=wrapper,
        arrival_abs=arrival_abs,
        start_abs=start_abs,
        end_expected_abs=start_abs + spec.requested_duration_s
    )

def stop_process_tree(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


# ============================================================
# PHASE POLICY
# ============================================================

def phase_peak_window(phase_seconds: int):
    return int(PEAK_START_FRAC * phase_seconds), int(PEAK_END_FRAC * phase_seconds)

def in_peak_window(phase_elapsed_s: float, phase_seconds: int):
    start_s, end_s = phase_peak_window(phase_seconds)
    return start_s <= phase_elapsed_s < end_s

def phase_rate_per_kwh(phase_elapsed_s: float, phase_seconds: int):
    return PEAK_RATE_PER_KWH if in_peak_window(phase_elapsed_s, phase_seconds) else CHEAP_RATE_PER_KWH

def free_gpu_ids(running_jobs: List[RuntimeJob]):
    busy = {rj.assigned_gpu for rj in running_jobs if rj.process.poll() is None}
    return [g for g in GPU_IDS if g not in busy]

def hot_system(telemetry: Dict[int, Dict]):
    return any(telemetry.get(g, {}).get("temp_c", 0.0) >= GPU_TEMP_BLOCK_C for g in GPU_IDS)

def select_gpu_baseline(free_gpus: List[int], telemetry: Dict[int, Dict]):
    return free_gpus[0] if free_gpus else None

def select_gpu_pack(free_gpus: List[int], telemetry: Dict[int, Dict]):
    return min(free_gpus) if free_gpus else None

def select_gpu_coolest(free_gpus: List[int], telemetry: Dict[int, Dict]):
    if not free_gpus:
        return None
    ranked = sorted(
        free_gpus,
        key=lambda g: (
            telemetry.get(g, {}).get("temp_c", 9999),
            telemetry.get(g, {}).get("util_percent", 9999),
            telemetry.get(g, {}).get("power_w", 9999),
            g,
        )
    )
    return ranked[0]

def select_gpu(policy_name: str, free_gpus: List[int], telemetry: Dict[int, Dict]):
    if not free_gpus:
        return None

    if policy_name == "baseline":
        return select_gpu_baseline(free_gpus, telemetry)

    packed = select_gpu_pack(free_gpus, telemetry)
    coolest = select_gpu_coolest(free_gpus, telemetry)

    if packed is None:
        return coolest
    if coolest is None:
        return packed

    packed_temp = telemetry.get(packed, {}).get("temp_c", 9999)
    coolest_temp = telemetry.get(coolest, {}).get("temp_c", 9999)

    # more aggressive thermal-aware placement
    if coolest_temp + 2 < packed_temp:
        return coolest
    return packed

def should_defer(policy_name: str, job: Job, phase_elapsed_s: float, phase_seconds: int, telemetry: Dict[int, Dict]):
    if policy_name == "baseline":
        return False, ""

    if not job.flexible:
        return False, ""

    latest_allowed = job.arrival_offset_s + job.defer_slack_s
    if phase_elapsed_s >= latest_allowed:
        return False, ""

    if in_peak_window(phase_elapsed_s, phase_seconds):
        return True, "mini_peak_price"

    if hot_system(telemetry):
        return True, "thermal_hot"

    return False, ""


# ============================================================
# CORE PHASE RUNNER
# ============================================================

def run_phase(phase_name: str, policy_name: str, phase_seconds: int, jobs: List[Job], event_writer, ts_writer):
    print(f"\n=== Starting phase: {phase_name} | policy={policy_name} | duration={phase_seconds}s ===")

    energy_file = find_rapl_energy_file()
    prev_energy = read_energy_uj(energy_file) if energy_file else None
    prev_time = time.time() if energy_file else None

    phase_start_abs = time.time()
    phase_end_abs = phase_start_abs + phase_seconds

    job_idx = 0
    queue: List[Job] = []
    running_jobs: List[RuntimeJob] = []

    wall_energy_ws = 0.0
    compute_energy_ws = 0.0
    wall_cost = 0.0
    shifted_energy_kwh = 0.0

    pue_sum = 0.0
    pue_count = 0

    defer_events = 0
    thermal_defers = 0
    price_defers = 0
    completed_jobs = 0
    launched_jobs = 0
    total_delay_s = 0.0

    peak_temp_c = 0.0
    hot_samples = 0

    try:
        while time.time() < phase_end_abs or running_jobs or job_idx < len(jobs) or queue:
            now_abs = time.time()
            phase_elapsed_s = now_abs - phase_start_abs

            telemetry = get_gpu_telemetry()
            cpu_temp, _, _ = get_cpu_temp_and_power_from_sensors()
            cpu_power, prev_energy, prev_time, cpu_power_source = get_cpu_power(
                energy_file, prev_energy, prev_time
            )

            base_power, aux_power, wall_power, base_err, aux_err = get_wall_power()
            total_gpu_power = sum(v.get("power_w", 0.0) for v in telemetry.values())

            compute_power = None
            if cpu_power is not None:
                compute_power = cpu_power + total_gpu_power
            elif ALLOW_GPU_ONLY_FALLBACK and total_gpu_power > 0:
                compute_power = total_gpu_power

            rate = phase_rate_per_kwh(phase_elapsed_s, phase_seconds)

            pue_proxy = None
            if wall_power is not None and compute_power is not None and compute_power > 0:
                pue_proxy = wall_power / compute_power
                pue_sum += pue_proxy
                pue_count += 1
                wall_energy_ws += wall_power * TICK_SECONDS
                compute_energy_ws += compute_power * TICK_SECONDS
                wall_cost += ((wall_power / 1000.0) * (TICK_SECONDS / 3600.0)) * rate

            for g in GPU_IDS:
                t = telemetry.get(g, {}).get("temp_c")
                if t is not None:
                    peak_temp_c = max(peak_temp_c, t)
                    if t >= GPU_TEMP_ALERT_C:
                        hot_samples += 1

            ts_writer.writerow({
                "timestamp": now_str(),
                "phase": phase_name,
                "policy": policy_name,
                "phase_elapsed_s": round(phase_elapsed_s, 2),
                "is_peak_window": int(in_peak_window(phase_elapsed_s, phase_seconds)),
                "energy_price_per_kwh": rate,
                "base_shelly_power_w": base_power,
                "aux_shelly_power_w": aux_power,
                "wall_power_w": wall_power,
                "cpu_power_w": cpu_power,
                "cpu_power_source": cpu_power_source,
                "cpu_temp_c": cpu_temp,
                "total_gpu_power_w": total_gpu_power,
                "compute_power_w": compute_power,
                "pue_proxy": pue_proxy,
                "gpu0_temp_c": telemetry.get(0, {}).get("temp_c"),
                "gpu1_temp_c": telemetry.get(1, {}).get("temp_c"),
                "gpu2_temp_c": telemetry.get(2, {}).get("temp_c"),
                "gpu0_power_w": telemetry.get(0, {}).get("power_w"),
                "gpu1_power_w": telemetry.get(1, {}).get("power_w"),
                "gpu2_power_w": telemetry.get(2, {}).get("power_w"),
                "base_shelly_error": base_err,
                "aux_shelly_error": aux_err,
            })

            while job_idx < len(jobs) and jobs[job_idx].arrival_offset_s <= phase_elapsed_s:
                queue.append(jobs[job_idx])
                event_writer.writerow({
                    "timestamp": now_str(),
                    "phase": phase_name,
                    "policy": policy_name,
                    "event": "ARRIVE",
                    "job_id": jobs[job_idx].job_id,
                    "job_type": jobs[job_idx].job_type,
                    "reason": "",
                    "gpu_id": "",
                    "delay_s": "",
                })
                job_idx += 1

            still_running = []
            for rj in running_jobs:
                rc = rj.process.poll()
                if rc is None:
                    still_running.append(rj)
                else:
                    completed_jobs += 1
                    event_writer.writerow({
                        "timestamp": now_str(),
                        "phase": phase_name,
                        "policy": policy_name,
                        "event": "FINISH",
                        "job_id": rj.spec.job_id,
                        "job_type": rj.spec.job_type,
                        "reason": "",
                        "gpu_id": rj.assigned_gpu,
                        "delay_s": round(rj.start_abs - rj.arrival_abs, 3),
                    })
                    if rj.wrapper_path and rj.wrapper_path.exists():
                        try:
                            rj.wrapper_path.unlink()
                        except Exception:
                            pass
            running_jobs = still_running

            queue.sort(key=lambda j: (j.arrival_offset_s, 0 if j.job_type == "inference" else 1))
            free_gpus = free_gpu_ids(running_jobs)

            starts_this_tick = 0
            max_starts_this_tick = (
                MAX_NEW_STARTS_PER_TICK_BASELINE
                if policy_name == "baseline"
                else MAX_NEW_STARTS_PER_TICK_SCHEDULED
            )

            for job in list(queue):
                if not free_gpus:
                    break
                if starts_this_tick >= max_starts_this_tick:
                    break

                defer, reason = should_defer(policy_name, job, phase_elapsed_s, phase_seconds, telemetry)
                if defer:
                    defer_events += 1
                    if reason == "thermal_hot":
                        thermal_defers += 1
                    elif reason == "mini_peak_price":
                        price_defers += 1

                    event_writer.writerow({
                        "timestamp": now_str(),
                        "phase": phase_name,
                        "policy": policy_name,
                        "event": "DEFER",
                        "job_id": job.job_id,
                        "job_type": job.job_type,
                        "reason": reason,
                        "gpu_id": "",
                        "delay_s": round(phase_elapsed_s - job.arrival_offset_s, 3),
                    })
                    continue

                gpu_id = select_gpu(policy_name, free_gpus, telemetry)
                if gpu_id is None:
                    break

                arrival_abs = phase_start_abs + job.arrival_offset_s
                rj = launch_job(job, gpu_id, arrival_abs)
                running_jobs.append(rj)
                queue.remove(job)
                free_gpus.remove(gpu_id)
                starts_this_tick += 1

                launched_jobs += 1
                total_delay_s += (rj.start_abs - arrival_abs)

                if policy_name != "baseline" and job.flexible and not in_peak_window(phase_elapsed_s, phase_seconds):
                    if compute_power is not None:
                        shifted_energy_kwh += (compute_power * min(job.requested_duration_s, TICK_SECONDS)) / 3_600_000.0

                event_writer.writerow({
                    "timestamp": now_str(),
                    "phase": phase_name,
                    "policy": policy_name,
                    "event": "START",
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "reason": policy_name,
                    "gpu_id": gpu_id,
                    "delay_s": round(rj.start_abs - arrival_abs, 3),
                })

            time.sleep(TICK_SECONDS)

    finally:
        for rj in running_jobs:
            stop_process_tree(rj.process)
            if rj.wrapper_path and rj.wrapper_path.exists():
                try:
                    rj.wrapper_path.unlink()
                except Exception:
                    pass

    avg_delay_s = total_delay_s / launched_jobs if launched_jobs > 0 else 0.0
    total_wall_energy_kwh = wall_energy_ws / 3_600_000.0
    total_compute_energy_kwh = compute_energy_ws / 3_600_000.0
    energy_based_pue = (wall_energy_ws / compute_energy_ws) if compute_energy_ws > 0 else None
    avg_pue = (pue_sum / pue_count) if pue_count > 0 else None

    return {
        "phase": phase_name,
        "policy": policy_name,
        "phase_seconds": phase_seconds,
        "jobs_total": len(jobs),
        "jobs_launched": launched_jobs,
        "jobs_completed": completed_jobs,
        "defer_events": defer_events,
        "thermal_defers": thermal_defers,
        "price_defers": price_defers,
        "avg_delay_s": round(avg_delay_s, 3),
        "peak_temp_c": round(peak_temp_c, 3),
        "hot_samples_ge_55c": int(hot_samples),
        "total_wall_energy_kwh": round(total_wall_energy_kwh, 6),
        "total_compute_energy_kwh": round(total_compute_energy_kwh, 6),
        "estimated_energy_cost_$": round(wall_cost, 6),
        "avg_pue_proxy": round(avg_pue, 6) if avg_pue is not None else None,
        "energy_based_pue_proxy": round(energy_based_pue, 6) if energy_based_pue is not None else None,
        "shifted_energy_kwh": round(shifted_energy_kwh, 6),
    }


# ============================================================
# PLOTS
# ============================================================

def make_owner_plots(summary_df: pd.DataFrame, ts_df: pd.DataFrame, events_df: pd.DataFrame):
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])

    baseline = summary_df[summary_df["phase"] == "baseline"].iloc[0]
    scheduled = summary_df[summary_df["phase"] == "scheduled"].iloc[0]

    # KPI comparison
    kpi_names = [
        "estimated_energy_cost_$",
        "total_wall_energy_kwh",
        "energy_based_pue_proxy",
        "peak_temp_c",
        "hot_samples_ge_55c",
        "shifted_energy_kwh",
    ]
    labels = [
        "Cost ($)",
        "Wall Energy (kWh)",
        "Energy-based PUE",
        "Peak GPU Temp (C)",
        "Hot Samples >=55C",
        "Energy Shifted (kWh)",
    ]

    base_vals = [baseline[k] if pd.notna(baseline[k]) else 0 for k in kpi_names]
    sched_vals = [scheduled[k] if pd.notna(scheduled[k]) else 0 for k in kpi_names]

    plt.figure(figsize=(12, 6))
    x = range(len(kpi_names))
    width = 0.38
    plt.bar([i - width / 2 for i in x], base_vals, width=width, label="Without Scheduler")
    plt.bar([i + width / 2 for i in x], sched_vals, width=width, label="With Scheduler")
    plt.xticks(list(x), labels, rotation=22, ha="right")
    plt.title("Owner-Focused KPI Comparison: With vs Without Scheduler")
    plt.legend()
    save_plot("01_owner_kpi_comparison.png")

    # cumulative cost
    cost_df = ts_df.sort_values("timestamp").copy()
    cost_df["delta_hours"] = TICK_SECONDS / 3600.0
    cost_df["incremental_cost_$"] = (cost_df["wall_power_w"] / 1000.0) * cost_df["energy_price_per_kwh"] * cost_df["delta_hours"]
    cost_df["cumulative_cost_$"] = cost_df.groupby("phase")["incremental_cost_$"].cumsum()

    plt.figure(figsize=(12, 5))
    for phase in ["baseline", "scheduled"]:
        sub = cost_df[cost_df["phase"] == phase]
        plt.plot(sub["timestamp"], sub["cumulative_cost_$"], label=phase)
    plt.title("Cumulative Energy Cost")
    plt.xlabel("Time")
    plt.ylabel("Cost ($)")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot("02_cumulative_cost_comparison.png")

    # PUE
    plt.figure(figsize=(12, 5))
    for phase in ["baseline", "scheduled"]:
        sub = ts_df[ts_df["phase"] == phase]
        plt.plot(sub["timestamp"], sub["pue_proxy"], label=phase)
    plt.title("PUE Proxy Over Time")
    plt.xlabel("Time")
    plt.ylabel("PUE Proxy")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot("03_pue_over_time_comparison.png")

    # peak temp
    plt.figure(figsize=(8, 5))
    plt.bar(["Without Scheduler", "With Scheduler"], [baseline["peak_temp_c"], scheduled["peak_temp_c"]])
    plt.title("Peak GPU Temperature Comparison")
    plt.ylabel("Peak Temperature (C)")
    save_plot("04_peak_temperature_comparison.png")

    # energy
    plt.figure(figsize=(9, 5))
    x = [0, 1]
    width = 0.35
    plt.bar([i - width/2 for i in x],
            [baseline["total_wall_energy_kwh"], scheduled["total_wall_energy_kwh"]],
            width=width, label="Wall Energy")
    plt.bar([i + width/2 for i in x],
            [baseline["total_compute_energy_kwh"], scheduled["total_compute_energy_kwh"]],
            width=width, label="Compute Energy")
    plt.xticks(x, ["Without Scheduler", "With Scheduler"])
    plt.title("Energy Consumption Comparison")
    plt.ylabel("kWh")
    plt.legend()
    save_plot("05_energy_comparison.png")

    # shifted energy
    plt.figure(figsize=(8, 5))
    plt.bar(["Without Scheduler", "With Scheduler"], [baseline["shifted_energy_kwh"], scheduled["shifted_energy_kwh"]])
    plt.title("Energy Shifted Out of Peak Window")
    plt.ylabel("Shifted Energy (kWh)")
    save_plot("06_energy_shifted.png")

    # thermal stress
    plt.figure(figsize=(10, 5))
    x = [0, 1]
    width = 0.35
    plt.bar([i - width/2 for i in x],
            [baseline["hot_samples_ge_55c"], scheduled["hot_samples_ge_55c"]],
            width=width, label="Hot Samples >=55C")
    plt.bar([i + width/2 for i in x],
            [baseline["peak_temp_c"], scheduled["peak_temp_c"]],
            width=width, label="Peak Temp (C)")
    plt.xticks(x, ["Without Scheduler", "With Scheduler"])
    plt.title("Thermal Stress Comparison")
    plt.legend()
    save_plot("07_thermal_stress_comparison.png")

    # scaled estimate
    base_cost = float(baseline["estimated_energy_cost_$"])
    sched_cost = float(scheduled["estimated_energy_cost_$"])
    base_energy = float(baseline["total_wall_energy_kwh"])
    sched_energy = float(scheduled["total_wall_energy_kwh"])

    demo_hours = TOTAL_TEST_MINUTES / 60.0
    annual_factor = ANNUALIZED_HOURS / max(demo_hours, 1e-9)

    node_annual_cost_saving = max(0.0, (base_cost - sched_cost) * annual_factor)
    node_annual_energy_saving = max(0.0, (base_energy - sched_energy) * annual_factor)

    dc_annual_cost_saving = node_annual_cost_saving * SCALING_NODE_COUNT
    dc_annual_energy_saving = node_annual_energy_saving * SCALING_NODE_COUNT

    scale_df = pd.DataFrame({
        "metric": ["Annual Energy Saved (kWh)", "Annual Cost Saved ($)"],
        "value": [dc_annual_energy_saving, dc_annual_cost_saving]
    })

    plt.figure(figsize=(9, 5))
    plt.bar(scale_df["metric"], scale_df["value"])
    plt.title(f"Scaled Full-Size Data Center Estimate ({SCALING_NODE_COUNT} similar nodes)")
    plt.ylabel("Estimated Annual Savings")
    plt.xticks(rotation=20, ha="right")
    save_plot("08_scaled_datacenter_estimate.png")

    # job summary
    counts = events_df.groupby(["phase", "event"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=["baseline", "scheduled"], fill_value=0)

    plt.figure(figsize=(10, 5))
    for event in counts.columns:
        plt.plot(counts.index, counts[event], marker="o", label=event)
    plt.title("Job Handling Summary")
    plt.ylabel("Count")
    plt.legend()
    save_plot("09_job_handling_summary.png")


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dirs()
    random.seed(SEED)

    if not Path(GPU_BURN_PATH).exists():
        raise FileNotFoundError(f"gpu_burn not found at {GPU_BURN_PATH}")

    baseline_minutes = TOTAL_TEST_MINUTES * BASELINE_FRACTION
    sched_minutes = TOTAL_TEST_MINUTES - baseline_minutes

    baseline_seconds = max(60, int(round(baseline_minutes * 60)))
    sched_seconds = max(60, int(round(sched_minutes * 60)))

    random.seed(SEED)
    baseline_jobs = generate_phase_jobs("baseline", baseline_seconds)
    random.seed(SEED)
    scheduled_jobs = generate_phase_jobs("scheduled", sched_seconds)

    events_f = open(EVENTS_CSV, "w", newline="")
    events_w = csv.DictWriter(events_f, fieldnames=[
        "timestamp", "phase", "policy", "event", "job_id", "job_type", "reason", "gpu_id", "delay_s"
    ])
    events_w.writeheader()

    ts_f = open(TIME_SERIES_CSV, "w", newline="")
    ts_w = csv.DictWriter(ts_f, fieldnames=[
        "timestamp", "phase", "policy", "phase_elapsed_s", "is_peak_window", "energy_price_per_kwh",
        "base_shelly_power_w", "aux_shelly_power_w", "wall_power_w",
        "cpu_power_w", "cpu_power_source", "cpu_temp_c",
        "total_gpu_power_w", "compute_power_w", "pue_proxy",
        "gpu0_temp_c", "gpu1_temp_c", "gpu2_temp_c",
        "gpu0_power_w", "gpu1_power_w", "gpu2_power_w",
        "base_shelly_error", "aux_shelly_error"
    ])
    ts_w.writeheader()

    try:
        baseline_summary = run_phase("baseline", "baseline", baseline_seconds, baseline_jobs, events_w, ts_w)
        scheduled_summary = run_phase("scheduled", "full", sched_seconds, scheduled_jobs, events_w, ts_w)
    finally:
        events_f.close()
        ts_f.close()

    summary_df = pd.DataFrame([baseline_summary, scheduled_summary])
    summary_df.to_csv(PHASE_SUMMARY_CSV, index=False)

    ts_df = pd.read_csv(TIME_SERIES_CSV)
    events_df = pd.read_csv(EVENTS_CSV)
    make_owner_plots(summary_df, ts_df, events_df)

    print("\nDone.")
    print(f"Run dir: {RUN_DIR}")
    print(f"Summary: {PHASE_SUMMARY_CSV}")
    print(f"Time series: {TIME_SERIES_CSV}")
    print(f"Events: {EVENTS_CSV}")
    print(f"Plots: {PLOTS_DIR}")
    if not HAS_DISPLAY:
        print("No GUI display detected, so plots were saved but not shown on screen.")


if __name__ == "__main__":
    main()
