#!/usr/bin/env python3

import os
from pathlib import Path

import matplotlib

# Try to use a GUI backend, fall back to Agg if it fails
try:
    matplotlib.use("TkAgg")  # or "Qt5Agg"
    import matplotlib.pyplot as plt
    HAS_DISPLAY = True
except:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_DISPLAY = False

import pandas as pd

# ============================================================
# USER SETTING: CHANGE THIS TO YOUR RUN FOLDER
# ============================================================
RUN_DIR = Path("/home/paul/gpu-burn/scheduler_with_pue_outputs/full_20260417_181226")

PLOTS_DIR = RUN_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def save_and_maybe_show(filename: str):
    out_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if HAS_DISPLAY:
        plt.show()
    plt.close()
    print(f"Saved plot: {out_path}")


def load_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    print(f"Missing file, skipping: {path}")
    return None


# ============================================================
# LOAD DATA
# ============================================================
telemetry = load_csv_if_exists(RUN_DIR / "telemetry_log.csv")
pue = load_csv_if_exists(RUN_DIR / "pue_log.csv")
dispatch = load_csv_if_exists(RUN_DIR / "dispatch_log.csv")
summary = load_csv_if_exists(RUN_DIR / "summary.csv")

if telemetry is None and pue is None and dispatch is None:
    raise FileNotFoundError(
        f"No usable CSV logs found in: {RUN_DIR}\n"
        f"Check that RUN_DIR points to the correct completed run."
    )

if telemetry is not None and "timestamp" in telemetry.columns:
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"])

if pue is not None and "timestamp" in pue.columns:
    pue["timestamp"] = pd.to_datetime(pue["timestamp"])

if dispatch is not None and "timestamp" in dispatch.columns:
    dispatch["timestamp"] = pd.to_datetime(dispatch["timestamp"])


# ============================================================
# 1) TOTAL GPU POWER OVER TIME
# ============================================================
if telemetry is not None and {"timestamp", "power_w"}.issubset(telemetry.columns):
    gpu_power = (
        telemetry.groupby("timestamp", as_index=False)["power_w"]
        .sum()
        .rename(columns={"power_w": "total_gpu_power_w"})
    )

    plt.figure(figsize=(12, 5))
    plt.plot(gpu_power["timestamp"], gpu_power["total_gpu_power_w"])
    plt.title("Total GPU Power Over Time")
    plt.xlabel("Time")
    plt.ylabel("GPU Power (W)")
    plt.xticks(rotation=45)
    save_and_maybe_show("01_total_gpu_power_over_time.png")


# ============================================================
# 2) WALL POWER VS COMPUTE POWER
# ============================================================
if pue is not None and {"timestamp", "total_numerator_power_w", "compute_power_w"}.issubset(pue.columns):
    plt.figure(figsize=(12, 5))
    plt.plot(pue["timestamp"], pue["total_numerator_power_w"], label="Wall/Aux Numerator Power")
    plt.plot(pue["timestamp"], pue["compute_power_w"], label="Compute Power")
    plt.title("Wall Power vs Compute Power")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.xticks(rotation=45)
    save_and_maybe_show("02_wall_vs_compute_power.png")


# ============================================================
# 3) PUE OVER TIME
# ============================================================
if pue is not None and {"timestamp", "pue_proxy"}.issubset(pue.columns):
    plt.figure(figsize=(12, 5))
    plt.plot(pue["timestamp"], pue["pue_proxy"])
    plt.title("PUE Proxy Over Time")
    plt.xlabel("Time")
    plt.ylabel("PUE Proxy")
    plt.xticks(rotation=45)
    save_and_maybe_show("03_pue_proxy_over_time.png")


# ============================================================
# 4) RUNNING / ENERGY-BASED AVERAGE PUE
# ============================================================
if pue is not None and "timestamp" in pue.columns:
    cols_present = []
    if "running_avg_pue_proxy" in pue.columns:
        cols_present.append(("running_avg_pue_proxy", "Running Avg PUE"))
    if "energy_based_avg_pue_proxy" in pue.columns:
        cols_present.append(("energy_based_avg_pue_proxy", "Energy-Based Avg PUE"))

    if cols_present:
        plt.figure(figsize=(12, 5))
        for col, label in cols_present:
            plt.plot(pue["timestamp"], pue[col], label=label)
        plt.title("Average PUE Metrics Over Time")
        plt.xlabel("Time")
        plt.ylabel("PUE")
        plt.legend()
        plt.xticks(rotation=45)
        save_and_maybe_show("04_average_pue_metrics.png")


# ============================================================
# 5) AVERAGE GPU TEMPERATURE OVER TIME
# ============================================================
if telemetry is not None and {"timestamp", "temp_c"}.issubset(telemetry.columns):
    gpu_temp = (
        telemetry.groupby("timestamp", as_index=False)["temp_c"]
        .mean()
        .rename(columns={"temp_c": "avg_gpu_temp_c"})
    )

    plt.figure(figsize=(12, 5))
    plt.plot(gpu_temp["timestamp"], gpu_temp["avg_gpu_temp_c"])
    plt.title("Average GPU Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Temperature (C)")
    plt.xticks(rotation=45)
    save_and_maybe_show("05_average_gpu_temperature_over_time.png")


# ============================================================
# 6) PER-GPU TEMPERATURE OVER TIME
# ============================================================
if telemetry is not None and {"timestamp", "gpu_id", "temp_c"}.issubset(telemetry.columns):
    plt.figure(figsize=(12, 5))
    for gpu_id in sorted(telemetry["gpu_id"].dropna().unique()):
        subset = telemetry[telemetry["gpu_id"] == gpu_id]
        plt.plot(subset["timestamp"], subset["temp_c"], label=f"GPU {gpu_id}")
    plt.title("Per-GPU Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.xticks(rotation=45)
    save_and_maybe_show("06_per_gpu_temperature_over_time.png")


# ============================================================
# 7) PER-GPU POWER OVER TIME
# ============================================================
if telemetry is not None and {"timestamp", "gpu_id", "power_w"}.issubset(telemetry.columns):
    plt.figure(figsize=(12, 5))
    for gpu_id in sorted(telemetry["gpu_id"].dropna().unique()):
        subset = telemetry[telemetry["gpu_id"] == gpu_id]
        plt.plot(subset["timestamp"], subset["power_w"], label=f"GPU {gpu_id}")
    plt.title("Per-GPU Power Over Time")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.xticks(rotation=45)
    save_and_maybe_show("07_per_gpu_power_over_time.png")


# ============================================================
# 8) ESTIMATED CUMULATIVE COST OVER TIME
# ============================================================
if pue is not None and {"timestamp", "total_numerator_power_w", "energy_price_per_kwh"}.issubset(pue.columns):
    pue_cost = pue.sort_values("timestamp").copy()
    pue_cost["delta_hours"] = pue_cost["timestamp"].diff().dt.total_seconds().fillna(0) / 3600.0
    pue_cost["incremental_cost_$"] = (
        (pue_cost["total_numerator_power_w"] / 1000.0)
        * pue_cost["energy_price_per_kwh"]
        * pue_cost["delta_hours"]
    )
    pue_cost["cumulative_cost_$"] = pue_cost["incremental_cost_$"].cumsum()

    plt.figure(figsize=(12, 5))
    plt.plot(pue_cost["timestamp"], pue_cost["cumulative_cost_$"])
    plt.title("Estimated Cumulative Energy Cost")
    plt.xlabel("Time")
    plt.ylabel("Cost ($)")
    plt.xticks(rotation=45)
    save_and_maybe_show("08_estimated_cumulative_energy_cost.png")


# ============================================================
# 9) INSTANTANEOUS COST RATE OVER TIME
# ============================================================
if pue is not None and {"timestamp", "instantaneous_it_cost_per_hour"}.issubset(pue.columns):
    plt.figure(figsize=(12, 5))
    plt.plot(pue["timestamp"], pue["instantaneous_it_cost_per_hour"])
    plt.title("Instantaneous Estimated Cost Rate")
    plt.xlabel("Time")
    plt.ylabel("Cost per Hour ($/h)")
    plt.xticks(rotation=45)
    save_and_maybe_show("09_instantaneous_cost_rate.png")


# ============================================================
# 10) ENERGY PRICE OVER TIME
# ============================================================
if pue is not None and {"timestamp", "energy_price_per_kwh"}.issubset(pue.columns):
    plt.figure(figsize=(12, 5))
    plt.plot(pue["timestamp"], pue["energy_price_per_kwh"])
    plt.title("Energy Price Over Time")
    plt.xlabel("Time")
    plt.ylabel("Price ($/kWh)")
    plt.xticks(rotation=45)
    save_and_maybe_show("10_energy_price_over_time.png")


# ============================================================
# 11) DISPATCH EVENTS BY TYPE
# ============================================================
if dispatch is not None and "event" in dispatch.columns:
    event_counts = dispatch["event"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(event_counts.index, event_counts.values)
    plt.title("Dispatch Events by Type")
    plt.xlabel("Event")
    plt.ylabel("Count")
    save_and_maybe_show("11_dispatch_events_by_type.png")


# ============================================================
# 12) DEFER EVENTS BY REASON
# ============================================================
if dispatch is not None and {"event", "reason"}.issubset(dispatch.columns):
    defer_df = dispatch[dispatch["event"] == "DEFER"].copy()
    if not defer_df.empty:
        defer_counts = defer_df["reason"].fillna("unknown").value_counts()

        plt.figure(figsize=(8, 5))
        plt.bar(defer_counts.index, defer_counts.values)
        plt.title("Defer Events by Reason")
        plt.xlabel("Reason")
        plt.ylabel("Count")
        save_and_maybe_show("12_defer_events_by_reason.png")


# ============================================================
# 13) QUEUE / START TIMELINE COUNTS
# ============================================================
if dispatch is not None and {"timestamp", "event"}.issubset(dispatch.columns):
    timeline = (
        dispatch.groupby(["timestamp", "event"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(12, 5))
    for col in timeline.columns:
        plt.plot(timeline.index, timeline[col], label=col)
    plt.title("Dispatch Event Counts Over Time")
    plt.xlabel("Time")
    plt.ylabel("Count per Timestamp")
    plt.legend()
    plt.xticks(rotation=45)
    save_and_maybe_show("13_dispatch_event_counts_over_time.png")


# ============================================================
# 14) TIME ABOVE THRESHOLD BY GPU
# ============================================================
if telemetry is not None and {"gpu_id", "temp_c"}.issubset(telemetry.columns):
    threshold = 70.0
    hot = telemetry[telemetry["temp_c"] >= threshold].copy()
    if not hot.empty:
        hot_counts = hot.groupby("gpu_id").size()

        plt.figure(figsize=(8, 5))
        plt.bar([str(i) for i in hot_counts.index], hot_counts.values)
        plt.title(f"Samples Above {threshold:.0f} C by GPU")
        plt.xlabel("GPU ID")
        plt.ylabel("Number of Samples")
        save_and_maybe_show("14_samples_above_70c_by_gpu.png")


# ============================================================
# 15) SUMMARY BAR CHART
# ============================================================
if summary is not None and {"metric", "value"}.issubset(summary.columns):
    summary_copy = summary.copy()

    wanted_metrics = [
        "jobs_generated_total",
        "jobs_launched",
        "jobs_completed",
        "defer_events_total",
        "thermal_defers",
        "temporal_defers",
    ]

    summary_small = summary_copy[summary_copy["metric"].isin(wanted_metrics)].copy()
    if not summary_small.empty:
        summary_small["value"] = pd.to_numeric(summary_small["value"], errors="coerce")

        plt.figure(figsize=(10, 5))
        plt.bar(summary_small["metric"], summary_small["value"])
        plt.title("Run Summary Metrics")
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        save_and_maybe_show("15_run_summary_metrics.png")


print(f"\nDone. Plots saved in: {PLOTS_DIR}")
if not HAS_DISPLAY:
    print("No GUI display detected, so plots were saved but not shown on screen.")
