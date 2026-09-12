#!/usr/bin/env python3

import csv
import os
import re
import time
import subprocess
from datetime import datetime

LOG_FILE = "power_log.csv"
INTERVAL_SECONDS = 2


def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip()
    except Exception:
        return ""


def get_cpu_temp_and_power_from_sensors():
    """
    Tries to read CPU temp and CPU/package power from lm-sensors output.
    Returns:
        cpu_temp_c, cpu_power_w, cpu_power_source
    """
    out = run_command(["sensors"])

    cpu_temp = None
    cpu_power = None
    cpu_power_source = None

    for line in out.splitlines():
        line_lower = line.lower().strip()

        # Temperature
        if (
            "package id 0:" in line_lower
            or line_lower.startswith("tctl:")
            or line_lower.startswith("tdie:")
            or "cpu temp:" in line_lower
        ):
            match = re.search(r"([+-]?\d+(\.\d+)?)°C", line)
            if match and cpu_temp is None:
                cpu_temp = float(match.group(1))

        # Power
        # Examples this may catch:
        # PPT:          45.23 W
        # Package power: 62.10 W
        # power1:       38.00 W
        if any(x in line_lower for x in ["ppt:", "package power", "power1:"]):
            match = re.search(r"([+-]?\d+(\.\d+)?)\s*W", line, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # avoid obviously bogus tiny values if there are multiple lines
                if cpu_power is None or value > cpu_power:
                    cpu_power = value
                    if "ppt" in line_lower:
                        cpu_power_source = "sensors:PPT"
                    elif "package power" in line_lower:
                        cpu_power_source = "sensors:package_power"
                    else:
                        cpu_power_source = "sensors:power1"

    return cpu_temp, cpu_power, cpu_power_source


def get_cpu_temp():
    cpu_temp, _, _ = get_cpu_temp_and_power_from_sensors()
    return cpu_temp


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

    if delta_energy_uj < 0:
        return None, now_energy, now_time

    if delta_time <= 0:
        return None, now_energy, now_time

    power_w = (delta_energy_uj / 1_000_000.0) / delta_time
    return power_w, now_energy, now_time


def get_cpu_power(energy_file, prev_energy, prev_time):
    """
    Returns:
        cpu_power_w, new_prev_energy, new_prev_time, cpu_power_source
    """
    # First try RAPL
    cpu_power_rapl, new_prev_energy, new_prev_time = estimate_cpu_power_w_from_rapl(
        energy_file, prev_energy, prev_time
    )
    if cpu_power_rapl is not None:
        return cpu_power_rapl, new_prev_energy, new_prev_time, "rapl"

    # Fallback to sensors
    _, cpu_power_sensors, cpu_power_source = get_cpu_temp_and_power_from_sensors()
    if cpu_power_sensors is not None:
        return cpu_power_sensors, new_prev_energy, new_prev_time, cpu_power_source

    return None, new_prev_energy, new_prev_time, None


def main():
    energy_file = find_rapl_energy_file()
    prev_energy = read_energy_uj(energy_file) if energy_file else None
    prev_time = time.time() if energy_file else None

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "cpu_power_w",
            "cpu_power_source",
            "cpu_temp_c",
            "gpu_index",
            "gpu_name",
            "gpu_power_w",
            "gpu_temp_c",
            "gpu_util_percent",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "total_measured_power_w"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        print(f"Logging to {LOG_FILE} every {INTERVAL_SECONDS} seconds...")
        if energy_file:
            print(f"Using CPU RAPL energy file: {energy_file}")
        else:
            print("No CPU RAPL energy file found. Will try lm-sensors power readings instead.")

        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cpu_temp = get_cpu_temp()
            cpu_power, prev_energy, prev_time, cpu_power_source = get_cpu_power(
                energy_file, prev_energy, prev_time
            )

            gpus = get_gpu_info()

            total_gpu_power = sum(
                g["gpu_power_w"] for g in gpus if g.get("gpu_power_w") is not None
            ) if gpus else 0.0

            total_measured = total_gpu_power + (cpu_power if cpu_power is not None else 0.0)

            if not gpus:
                writer.writerow({
                    "timestamp": timestamp,
                    "cpu_power_w": cpu_power,
                    "cpu_power_source": cpu_power_source,
                    "cpu_temp_c": cpu_temp,
                    "gpu_index": "",
                    "gpu_name": "",
                    "gpu_power_w": "",
                    "gpu_temp_c": "",
                    "gpu_util_percent": "",
                    "gpu_mem_used_mb": "",
                    "gpu_mem_total_mb": "",
                    "total_measured_power_w": total_measured if cpu_power is not None else ""
                })
            else:
                for g in gpus:
                    writer.writerow({
                        "timestamp": timestamp,
                        "cpu_power_w": cpu_power,
                        "cpu_power_source": cpu_power_source,
                        "cpu_temp_c": cpu_temp,
                        "gpu_index": g["gpu_index"],
                        "gpu_name": g["gpu_name"],
                        "gpu_power_w": g["gpu_power_w"],
                        "gpu_temp_c": g["gpu_temp_c"],
                        "gpu_util_percent": g["gpu_util_percent"],
                        "gpu_mem_used_mb": g["gpu_mem_used_mb"],
                        "gpu_mem_total_mb": g["gpu_mem_total_mb"],
                        "total_measured_power_w": total_measured
                    })

            csvfile.flush()

            print(
                f"{timestamp} | "
                f"CPU: {cpu_power} W | "
                f"CPU Source: {cpu_power_source} | "
                f"CPU Temp: {cpu_temp} C | "
                f"Total GPU: {total_gpu_power:.2f} W | "
                f"Total Measured: {total_measured:.2f} W"
            )

            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped logging.")
