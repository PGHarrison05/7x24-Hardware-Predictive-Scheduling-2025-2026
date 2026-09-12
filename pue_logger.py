#!/usr/bin/env python3

import csv
import os
import re
import time
import subprocess
from datetime import datetime

import requests

# ============================================================
# USER SETTINGS
# ============================================================

# Main Shelly measuring your base system / wall power
# Put your original Shelly IP here.
BASE_SHELLY_IP = "149.61.237.244"   # <-- change if needed

# Extra Shelly whose measured power should be ADDED to numerator
AUX_SHELLY_IP = "149.61.201.189"

BASE_SHELLY_URL = f"http://{BASE_SHELLY_IP}/rpc/Shelly.GetStatus"
AUX_SHELLY_URL = f"http://{AUX_SHELLY_IP}/rpc/Shelly.GetStatus"

LOG_FILE = "pue_log.csv"
INTERVAL_SECONDS = 2

# If True, writes rows even when PUE can't be computed yet
LOG_INVALID_ROWS = True

# If True and CPU power is unavailable, denominator can fall back to GPU-only.
# If False, denominator requires CPU + GPU.
ALLOW_GPU_ONLY_FALLBACK = False


# ============================================================
# BASIC HELPERS
# ============================================================

def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip()
    except Exception:
        return ""


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


# ============================================================
# SHELLY FUNCTIONS
# ============================================================

def get_shelly_power_from_url(url):
    """
    Returns (power_w, raw_json, error_string)

    Expected common Gen2/Gen3 location:
      data["switch:0"]["apower"]

    Fallback:
      sum all switch:N apower values
    """
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

        return None, data, "No apower field found in Shelly status."

    except Exception as e:
        return None, None, str(e)


# ============================================================
# CPU / SENSORS FUNCTIONS
# ============================================================

def get_cpu_temp_and_power_from_sensors():
    """
    Try lm-sensors for CPU temp and package power.

    Returns:
      cpu_temp_c, cpu_power_w, cpu_power_source
    """
    out = run_command(["sensors"])

    cpu_temp = None
    cpu_power = None
    cpu_power_source = None

    for line in out.splitlines():
        line_lower = line.lower().strip()

        # Temperature candidates
        if (
            "package id 0:" in line_lower
            or line_lower.startswith("tctl:")
            or line_lower.startswith("tdie:")
            or "cpu temp:" in line_lower
        ):
            match = re.search(r"([+-]?\d+(\.\d+)?)°c", line_lower)
            if match and cpu_temp is None:
                cpu_temp = float(match.group(1))

        # Power candidates
        if any(token in line_lower for token in ["ppt:", "package power", "power1:"]):
            match = re.search(r"([+-]?\d+(\.\d+)?)\s*w", line_lower)
            if match:
                value = float(match.group(1))
                if cpu_power is None or value > cpu_power:
                    cpu_power = value
                    if "ppt:" in line_lower:
                        cpu_power_source = "sensors:PPT"
                    elif "package power" in line_lower:
                        cpu_power_source = "sensors:package_power"
                    else:
                        cpu_power_source = "sensors:power1"

    return cpu_temp, cpu_power, cpu_power_source


def find_rapl_energy_file():
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

    # Handle counter reset / wraparound conservatively
    if delta_energy_uj < 0 or delta_time <= 0:
        return None, now_energy, now_time

    power_w = (delta_energy_uj / 1_000_000.0) / delta_time
    return power_w, now_energy, now_time


def get_cpu_power(energy_file, prev_energy, prev_time):
    """
    Returns:
      cpu_power_w, new_prev_energy, new_prev_time, source
    """
    cpu_power_rapl, new_prev_energy, new_prev_time = estimate_cpu_power_w_from_rapl(
        energy_file, prev_energy, prev_time
    )
    if cpu_power_rapl is not None:
        return cpu_power_rapl, new_prev_energy, new_prev_time, "rapl"

    _, cpu_power_sensors, cpu_power_source = get_cpu_temp_and_power_from_sensors()
    if cpu_power_sensors is not None:
        return cpu_power_sensors, new_prev_energy, new_prev_time, cpu_power_source

    return None, new_prev_energy, new_prev_time, None


# ============================================================
# GPU FUNCTIONS
# ============================================================

def get_gpu_info():
    out = run_command([
        "nvidia-smi",
        "--query-gpu=index,name,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ])

    gpus = []
    if not out:
        return gpus

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            try:
                gpus.append({
                    "gpu_index": parts[0],
                    "gpu_name": parts[1],
                    "gpu_power_w": float(parts[2]),
                    "gpu_temp_c": float(parts[3]),
                    "gpu_util_percent": float(parts[4]),
                    "gpu_mem_used_mb": float(parts[5]),
                    "gpu_mem_total_mb": float(parts[6]),
                })
            except ValueError:
                pass

    return gpus


# ============================================================
# MAIN
# ============================================================

def main():
    energy_file = find_rapl_energy_file()
    prev_energy = read_energy_uj(energy_file) if energy_file else None
    prev_time = time.time() if energy_file else None

    file_exists = os.path.exists(LOG_FILE)

    # Running stats for average of instantaneous PUE
    pue_sum = 0.0
    pue_count = 0

    # Energy-based ratio accumulator
    wall_energy_sum_ws = 0.0
    compute_energy_sum_ws = 0.0

    with open(LOG_FILE, "a", newline="") as csvfile:
        fieldnames = [
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
            "gpu_index",
            "gpu_name",
            "gpu_power_w",
            "gpu_temp_c",
            "gpu_util_percent",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "base_shelly_error",
            "aux_shelly_error",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        print(f"Logging to {LOG_FILE} every {INTERVAL_SECONDS} seconds...")
        print(f"Base Shelly URL: {BASE_SHELLY_URL}")
        print(f"Aux  Shelly URL: {AUX_SHELLY_URL}")

        if energy_file:
            print(f"Using CPU RAPL energy file: {energy_file}")
        else:
            print("No CPU RAPL energy file found. Falling back to lm-sensors for CPU power if available.")

        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Shelly readings
            base_shelly_power, _, base_shelly_error = get_shelly_power_from_url(BASE_SHELLY_URL)
            aux_shelly_power, _, aux_shelly_error = get_shelly_power_from_url(AUX_SHELLY_URL)

            # CPU readings
            cpu_temp, _, _ = get_cpu_temp_and_power_from_sensors()
            cpu_power, prev_energy, prev_time, cpu_power_source = get_cpu_power(
                energy_file, prev_energy, prev_time
            )

            # GPU readings
            gpus = get_gpu_info()
            total_gpu_power = sum(g["gpu_power_w"] for g in gpus) if gpus else 0.0

            # Denominator: compute power
            compute_power = None
            if cpu_power is not None:
                compute_power = cpu_power + total_gpu_power
            elif ALLOW_GPU_ONLY_FALLBACK and total_gpu_power > 0:
                compute_power = total_gpu_power

            # Numerator: total power draw = base Shelly + aux Shelly
            total_numerator_power = None
            if base_shelly_power is not None or aux_shelly_power is not None:
                total_numerator_power = (base_shelly_power or 0.0) + (aux_shelly_power or 0.0)

            # Instantaneous PUE
            pue_proxy = None
            if (
                total_numerator_power is not None
                and compute_power is not None
                and compute_power > 0
            ):
                pue_proxy = total_numerator_power / compute_power
                pue_sum += pue_proxy
                pue_count += 1

            running_avg_pue = (pue_sum / pue_count) if pue_count > 0 else None

            # Energy-based PUE average
            energy_based_avg_pue = None
            if (
                total_numerator_power is not None
                and compute_power is not None
                and compute_power > 0
            ):
                # Since interval is fixed, watt-seconds are fine for relative energy
                wall_energy_sum_ws += total_numerator_power * INTERVAL_SECONDS
                compute_energy_sum_ws += compute_power * INTERVAL_SECONDS

                if compute_energy_sum_ws > 0:
                    energy_based_avg_pue = wall_energy_sum_ws / compute_energy_sum_ws
            elif compute_energy_sum_ws > 0:
                energy_based_avg_pue = wall_energy_sum_ws / compute_energy_sum_ws

            # If no GPUs, still log one row
            if not gpus:
                row = {
                    "timestamp": timestamp,
                    "base_shelly_power_w": base_shelly_power,
                    "aux_shelly_power_w": aux_shelly_power,
                    "total_numerator_power_w": total_numerator_power,
                    "cpu_power_w": cpu_power,
                    "cpu_power_source": cpu_power_source,
                    "cpu_temp_c": cpu_temp,
                    "total_gpu_power_w": total_gpu_power,
                    "compute_power_w": compute_power,
                    "pue_proxy": pue_proxy,
                    "running_avg_pue_proxy": running_avg_pue,
                    "energy_based_avg_pue_proxy": energy_based_avg_pue,
                    "gpu_index": "",
                    "gpu_name": "",
                    "gpu_power_w": "",
                    "gpu_temp_c": "",
                    "gpu_util_percent": "",
                    "gpu_mem_used_mb": "",
                    "gpu_mem_total_mb": "",
                    "base_shelly_error": base_shelly_error,
                    "aux_shelly_error": aux_shelly_error,
                }
                if LOG_INVALID_ROWS or pue_proxy is not None:
                    writer.writerow(row)
            else:
                for g in gpus:
                    row = {
                        "timestamp": timestamp,
                        "base_shelly_power_w": base_shelly_power,
                        "aux_shelly_power_w": aux_shelly_power,
                        "total_numerator_power_w": total_numerator_power,
                        "cpu_power_w": cpu_power,
                        "cpu_power_source": cpu_power_source,
                        "cpu_temp_c": cpu_temp,
                        "total_gpu_power_w": total_gpu_power,
                        "compute_power_w": compute_power,
                        "pue_proxy": pue_proxy,
                        "running_avg_pue_proxy": running_avg_pue,
                        "energy_based_avg_pue_proxy": energy_based_avg_pue,
                        "gpu_index": g["gpu_index"],
                        "gpu_name": g["gpu_name"],
                        "gpu_power_w": g["gpu_power_w"],
                        "gpu_temp_c": g["gpu_temp_c"],
                        "gpu_util_percent": g["gpu_util_percent"],
                        "gpu_mem_used_mb": g["gpu_mem_used_mb"],
                        "gpu_mem_total_mb": g["gpu_mem_total_mb"],
                        "base_shelly_error": base_shelly_error,
                        "aux_shelly_error": aux_shelly_error,
                    }
                    if LOG_INVALID_ROWS or pue_proxy is not None:
                        writer.writerow(row)

            csvfile.flush()

            print(
                f"{timestamp} | "
                f"Base Shelly: {base_shelly_power} W | "
                f"Aux Shelly: {aux_shelly_power} W | "
                f"Numerator: {total_numerator_power} W | "
                f"CPU: {cpu_power} W | "
                f"GPU Total: {total_gpu_power:.2f} W | "
                f"Compute: {compute_power} W | "
                f"PUE: {pue_proxy} | "
                f"Avg PUE: {running_avg_pue} | "
                f"Energy Avg PUE: {energy_based_avg_pue}"
            )

            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped logging.")
