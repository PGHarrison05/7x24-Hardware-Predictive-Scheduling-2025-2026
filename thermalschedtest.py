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

# ---- total run ----
TOTAL_TEST_MINUTES = 5          # 2 hours total
BASELINE_FRACTION = 0.50          # first half baseline, second half thermal

# ---- scheduler tick ----
TICK_SECONDS = 5
SEED = 42

# ---- GPU setup ----
GPU_IDS = [0, 1, 2]
GPU_BURN_PATH = "/home/paul/gpu-burn/gpu_burn"

# ---- heat-focused workload ----
AVG_JOBS_PER_MINUTE = 2.5
TRAINING_RATIO = 0.95
INFERENCE_RATIO = 0.05

TRAINING_DURATION_RANGE_S = (300, 1800)
TRAINING_FLEXIBLE_PROB = 0.95
TRAINING_DEFER_SLACK_RANGE_S = (300, 3600)

INFERENCE_DURATION_RANGE_S = (20, 45)
INFERENCE_DUTY_CYCLE_CHOICES = [0.20, 0.30]
INFERENCE_BURST_ON_S = 3

# ---- thermal scheduler thresholds ----
# Tune these to where your GPUs actually run under load.
GPU_TEMP_BLOCK_C = 58.0
GPU_TEMP_RESUME_C = 54.0
GPU_TEMP_ALERT_C = 55.0

# ---- launch behavior ----
MAX_NEW_STARTS_PER_TICK_BASELINE = 3
MAX_NEW_STARTS_PER_TICK_THERMAL = 1

# ---- output ----
OUTPUT_ROOT = Path("/home/paul/gpu-burn/thermal_test_outputs")
RUN_NAME = f"thermal_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = OUTPUT_ROOT / RUN_NAME
EVENTS_CSV = RUN_DIR / "events.csv"
TIME_SERIES_CSV = RUN_DIR / "time_series.csv"
SUMMARY_CSV = RUN_DIR / "summary.csv"
PLOTS_DIR = RUN_DIR / "plots"


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
# TELEMETRY
# ============================================================

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

def get_cpu_temp():
    out = run_command(["sensors"])
    for line in out.splitlines():
        ll = line.lower().strip()
        if ("package id 0:" in ll) or ll.startswith("tctl:") or ll.startswith("tdie:") or ("cpu temp:" in ll):
            match = re.search(r"([+-]?\d+(\.\d+)?)°c", ll)
            if match:
                return float(match.group(1))
    return None


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
    # Double precision mode for extra heat / power
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
# THERMAL POLICY
# ============================================================

def free_gpu_ids(running_jobs: List[RuntimeJob]):
    busy = {rj.assigned_gpu for rj in running_jobs if rj.process.poll() is None}
    return [g for g in GPU_IDS if g not in busy]

def hot_gpus(telemetry: Dict[int, Dict]) -> List[int]:
    return [g for g in GPU_IDS if telemetry.get(g, {}).get("temp_c", 0.0) >= GPU_TEMP_BLOCK_C]

def cooled_gpus(telemetry: Dict[int, Dict]) -> List[int]:
    return [g for g in GPU_IDS if telemetry.get(g, {}).get("temp_c", 9999.0) <= GPU_TEMP_RESUME_C]

def select_gpu_baseline(free_gpus: List[int], telemetry: Dict[int, Dict]):
    return free_gpus[0] if free_gpus else None

def select_gpu_thermal(free_gpus: List[int], telemetry: Dict[int, Dict]):
    if not free_gpus:
        return None

    # prefer coolest GPU
    ranked = sorted(
        free_gpus,
        key=lambda g: (
            telemetry.get(g, {}).get("temp_c", 9999.0),
            telemetry.get(g, {}).get("power_w", 9999.0),
            telemetry.get(g, {}).get("util_percent", 9999.0),
            g
        )
    )

    # choose the coolest one
    return ranked[0]

def should_defer_for_thermal(policy_name: str, job: Job, phase_elapsed_s: float, telemetry: Dict[int, Dict]):
    if policy_name == "baseline":
        return False, ""

    if not job.flexible:
        return False, ""

    latest_allowed = job.arrival_offset_s + job.defer_slack_s
    if phase_elapsed_s >= latest_allowed:
        return False, ""

    if any(telemetry.get(g, {}).get("temp_c", 0.0) >= GPU_TEMP_BLOCK_C for g in GPU_IDS):
        return True, "thermal_hot"

    return False, ""


# ============================================================
# CORE PHASE RUNNER
# ============================================================

def run_phase(phase_name: str, policy_name: str, phase_seconds: int, jobs: List[Job], event_writer, ts_writer):
    print(f"\n=== Starting phase: {phase_name} | policy={policy_name} | duration={phase_seconds}s ===")

    phase_start_abs = time.time()
    phase_end_abs = phase_start_abs + phase_seconds

    job_idx = 0
    queue: List[Job] = []
    running_jobs: List[RuntimeJob] = []

    launched_jobs = 0
    completed_jobs = 0
    thermal_defers = 0
    total_delay_s = 0.0

    peak_temp_c = 0.0
    hot_samples = 0
    avg_temp_sum = 0.0
    avg_temp_count = 0

    try:
        while time.time() < phase_end_abs or running_jobs or job_idx < len(jobs) or queue:
            now_abs = time.time()
            phase_elapsed_s = now_abs - phase_start_abs

            telemetry = get_gpu_telemetry()
            cpu_temp = get_cpu_temp()

            gpu_temps = []
            total_gpu_power = 0.0
            total_gpu_util = 0.0

            for g in GPU_IDS:
                t = telemetry.get(g, {}).get("temp_c")
                p = telemetry.get(g, {}).get("power_w")
                u = telemetry.get(g, {}).get("util_percent")

                if t is not None:
                    gpu_temps.append(t)
                    peak_temp_c = max(peak_temp_c, t)
                    if t >= GPU_TEMP_ALERT_C:
                        hot_samples += 1
                if p is not None:
                    total_gpu_power += p
                if u is not None:
                    total_gpu_util += u

            if gpu_temps:
                avg_temp_sum += sum(gpu_temps) / len(gpu_temps)
                avg_temp_count += 1

            ts_writer.writerow({
                "timestamp": now_str(),
                "phase": phase_name,
                "policy": policy_name,
                "phase_elapsed_s": round(phase_elapsed_s, 2),
                "cpu_temp_c": cpu_temp,
                "total_gpu_power_w": total_gpu_power,
                "avg_gpu_util_percent": total_gpu_util / max(len(GPU_IDS), 1),
                "gpu0_temp_c": telemetry.get(0, {}).get("temp_c"),
                "gpu1_temp_c": telemetry.get(1, {}).get("temp_c"),
                "gpu2_temp_c": telemetry.get(2, {}).get("temp_c"),
                "gpu0_power_w": telemetry.get(0, {}).get("power_w"),
                "gpu1_power_w": telemetry.get(1, {}).get("power_w"),
                "gpu2_power_w": telemetry.get(2, {}).get("power_w"),
                "gpu0_util_percent": telemetry.get(0, {}).get("util_percent"),
                "gpu1_util_percent": telemetry.get(1, {}).get("util_percent"),
                "gpu2_util_percent": telemetry.get(2, {}).get("util_percent"),
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
                else MAX_NEW_STARTS_PER_TICK_THERMAL
            )

            for job in list(queue):
                if not free_gpus:
                    break
                if starts_this_tick >= max_starts_this_tick:
                    break

                defer, reason = should_defer_for_thermal(policy_name, job, phase_elapsed_s, telemetry)
                if defer:
                    thermal_defers += 1
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

                gpu_id = (
                    select_gpu_baseline(free_gpus, telemetry)
                    if policy_name == "baseline"
                    else select_gpu_thermal(free_gpus, telemetry)
                )
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
    avg_gpu_temp_c = avg_temp_sum / avg_temp_count if avg_temp_count > 0 else None

    return {
        "phase": phase_name,
        "policy": policy_name,
        "phase_seconds": phase_seconds,
        "jobs_total": len(jobs),
        "jobs_launched": launched_jobs,
        "jobs_completed": completed_jobs,
        "thermal_defers": thermal_defers,
        "avg_delay_s": round(avg_delay_s, 3),
        "peak_temp_c": round(peak_temp_c, 3),
        "avg_gpu_temp_c": round(avg_gpu_temp_c, 3) if avg_gpu_temp_c is not None else None,
        "hot_samples_ge_55c": int(hot_samples),
    }


# ============================================================
# PLOTS
# ============================================================

def make_plots(summary_df: pd.DataFrame, ts_df: pd.DataFrame, events_df: pd.DataFrame):
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])

    baseline = summary_df[summary_df["phase"] == "baseline"].iloc[0]
    thermal = summary_df[summary_df["phase"] == "thermal"].iloc[0]

    # 1. peak temp
    plt.figure(figsize=(8, 5))
    plt.bar(["Without Thermal Scheduler", "With Thermal Scheduler"],
            [baseline["peak_temp_c"], thermal["peak_temp_c"]])
    plt.title("Peak GPU Temperature Comparison")
    plt.ylabel("Peak Temperature (C)")
    save_plot("01_peak_gpu_temperature_comparison.png")

    # 2. average temp
    plt.figure(figsize=(8, 5))
    plt.bar(["Without Thermal Scheduler", "With Thermal Scheduler"],
            [baseline["avg_gpu_temp_c"], thermal["avg_gpu_temp_c"]])
    plt.title("Average GPU Temperature Comparison")
    plt.ylabel("Average Temperature (C)")
    save_plot("02_average_gpu_temperature_comparison.png")

    # 3. hot samples
    plt.figure(figsize=(8, 5))
    plt.bar(["Without Thermal Scheduler", "With Thermal Scheduler"],
            [baseline["hot_samples_ge_55c"], thermal["hot_samples_ge_55c"]])
    plt.title("Thermal Stress Samples >=55C")
    plt.ylabel("Count")
    save_plot("03_hot_samples_comparison.png")

    # 4. per-phase temperature timeline
    plt.figure(figsize=(12, 5))
    for phase in ["baseline", "thermal"]:
        sub = ts_df[ts_df["phase"] == phase].copy()
        temp_cols = ["gpu0_temp_c", "gpu1_temp_c", "gpu2_temp_c"]
        sub["avg_gpu_temp_c"] = sub[temp_cols].mean(axis=1)
        plt.plot(sub["timestamp"], sub["avg_gpu_temp_c"], label=phase)
    plt.title("Average GPU Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Average GPU Temperature (C)")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot("04_average_gpu_temperature_over_time.png")

    # 5. per-GPU temperature timeline
    plt.figure(figsize=(12, 5))
    for gpu_col in ["gpu0_temp_c", "gpu1_temp_c", "gpu2_temp_c"]:
        plt.plot(ts_df["timestamp"], ts_df[gpu_col], label=gpu_col)
    plt.title("Per-GPU Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot("05_per_gpu_temperature_over_time.png")

    # 6. GPU power timeline
    plt.figure(figsize=(12, 5))
    for phase in ["baseline", "thermal"]:
        sub = ts_df[ts_df["phase"] == phase]
        plt.plot(sub["timestamp"], sub["total_gpu_power_w"], label=phase)
    plt.title("Total GPU Power Over Time")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot("06_total_gpu_power_over_time.png")

    # 7. defer events
    if "event" in events_df.columns:
        defer_df = events_df[events_df["event"] == "DEFER"]
        if not defer_df.empty:
            defer_counts = defer_df.groupby("phase").size()
            plt.figure(figsize=(8, 5))
            plt.bar(defer_counts.index, defer_counts.values)
            plt.title("Thermal Defer Events")
            plt.ylabel("Count")
            save_plot("07_thermal_defer_events.png")

    # 8. summary comparison
    metrics = ["jobs_completed", "thermal_defers", "peak_temp_c", "hot_samples_ge_55c"]
    labels = ["Jobs Completed", "Thermal Defers", "Peak Temp (C)", "Hot Samples >=55C"]

    base_vals = [baseline[m] for m in metrics]
    therm_vals = [thermal[m] for m in metrics]

    plt.figure(figsize=(10, 5))
    x = range(len(metrics))
    width = 0.38
    plt.bar([i - width/2 for i in x], base_vals, width=width, label="Without Thermal Scheduler")
    plt.bar([i + width/2 for i in x], therm_vals, width=width, label="With Thermal Scheduler")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.title("Thermal Scheduler Comparison")
    plt.legend()
    save_plot("08_thermal_scheduler_summary.png")


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dirs()
    random.seed(SEED)

    if not Path(GPU_BURN_PATH).exists():
        raise FileNotFoundError(f"gpu_burn not found at {GPU_BURN_PATH}")

    baseline_minutes = TOTAL_TEST_MINUTES * BASELINE_FRACTION
    thermal_minutes = TOTAL_TEST_MINUTES - baseline_minutes

    baseline_seconds = max(60, int(round(baseline_minutes * 60)))
    thermal_seconds = max(60, int(round(thermal_minutes * 60)))

    # identical workload pattern across both halves
    random.seed(SEED)
    baseline_jobs = generate_phase_jobs("baseline", baseline_seconds)
    random.seed(SEED)
    thermal_jobs = generate_phase_jobs("thermal", thermal_seconds)

    events_f = open(EVENTS_CSV, "w", newline="")
    events_w = csv.DictWriter(events_f, fieldnames=[
        "timestamp", "phase", "policy", "event", "job_id", "job_type", "reason", "gpu_id", "delay_s"
    ])
    events_w.writeheader()

    ts_f = open(TIME_SERIES_CSV, "w", newline="")
    ts_w = csv.DictWriter(ts_f, fieldnames=[
        "timestamp", "phase", "policy", "phase_elapsed_s", "cpu_temp_c",
        "total_gpu_power_w", "avg_gpu_util_percent",
        "gpu0_temp_c", "gpu1_temp_c", "gpu2_temp_c",
        "gpu0_power_w", "gpu1_power_w", "gpu2_power_w",
        "gpu0_util_percent", "gpu1_util_percent", "gpu2_util_percent"
    ])
    ts_w.writeheader()

    try:
        baseline_summary = run_phase("baseline", "baseline", baseline_seconds, baseline_jobs, events_w, ts_w)
        thermal_summary = run_phase("thermal", "thermal", thermal_seconds, thermal_jobs, events_w, ts_w)
    finally:
        events_f.close()
        ts_f.close()

    summary_df = pd.DataFrame([baseline_summary, thermal_summary])
    summary_df.to_csv(SUMMARY_CSV, index=False)

    ts_df = pd.read_csv(TIME_SERIES_CSV)
    events_df = pd.read_csv(EVENTS_CSV)
    make_plots(summary_df, ts_df, events_df)

    print("\nDone.")
    print(f"Run dir:   {RUN_DIR}")
    print(f"Summary:   {SUMMARY_CSV}")
    print(f"Series:    {TIME_SERIES_CSV}")
    print(f"Events:    {EVENTS_CSV}")
    print(f"Plots:     {PLOTS_DIR}")
    if not HAS_DISPLAY:
        print("No GUI display detected, so plots were saved but not shown on screen.")


if __name__ == "__main__":
    main()


print("\n=== THERMAL SCHEDULER RESULTS ===\n")

baseline = summary_df[summary_df["phase"] == "baseline"].iloc[0]
thermal = summary_df[summary_df["phase"] == "thermal"].iloc[0]

peak_reduction = baseline["peak_temp_c"] - thermal["peak_temp_c"]
avg_reduction = baseline["avg_gpu_temp_c"] - thermal["avg_gpu_temp_c"]
hot_reduction = baseline["hot_samples_ge_55c"] - thermal["hot_samples_ge_55c"]

print(f"Peak temp (baseline):     {baseline['peak_temp_c']} C")
print(f"Peak temp (thermal):      {thermal['peak_temp_c']} C")
print(f"Peak reduction:           {peak_reduction:.2f} C\n")

print(f"Avg temp (baseline):      {baseline['avg_gpu_temp_c']} C")
print(f"Avg temp (thermal):       {thermal['avg_gpu_temp_c']} C")
print(f"Avg reduction:            {avg_reduction:.2f} C\n")

print(f"Hot samples (baseline):   {baseline['hot_samples_ge_55c']}")
print(f"Hot samples (thermal):    {thermal['hot_samples_ge_55c']}")
print(f"Hot samples avoided:      {hot_reduction}\n")

print(f"Thermal defers used:      {thermal['thermal_defers']}")
