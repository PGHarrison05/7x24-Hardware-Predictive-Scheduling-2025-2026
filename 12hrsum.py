#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

RUN_DIR = Path("/home/paul/gpu-burn/demo_compare_outputs/demo_compare_12h_20260419_221557")
TOTAL_TEST_MINUTES = 720
NODE_COUNT = 1000
HOURS_IN_YEAR = 8760

summary_path = RUN_DIR / "phase_summary.csv"
df = pd.read_csv(summary_path)

baseline = df[df["phase"] == "baseline"].iloc[0]
scheduled = df[df["phase"] == "scheduled"].iloc[0]

cost_baseline = float(baseline["estimated_energy_cost_$"])
cost_scheduled = float(scheduled["estimated_energy_cost_$"])
cost_savings = cost_baseline - cost_scheduled
cost_savings_pct = 100 * cost_savings / cost_baseline if cost_baseline else 0.0

wall_energy_baseline = float(baseline["total_wall_energy_kwh"])
wall_energy_scheduled = float(scheduled["total_wall_energy_kwh"])
wall_energy_savings = wall_energy_baseline - wall_energy_scheduled
wall_energy_savings_pct = 100 * wall_energy_savings / wall_energy_baseline if wall_energy_baseline else 0.0

compute_energy_baseline = float(baseline["total_compute_energy_kwh"])
compute_energy_scheduled = float(scheduled["total_compute_energy_kwh"])
compute_energy_savings = compute_energy_baseline - compute_energy_scheduled
compute_energy_savings_pct = 100 * compute_energy_savings / compute_energy_baseline if compute_energy_baseline else 0.0

pue_baseline = float(baseline["energy_based_pue_proxy"])
pue_scheduled = float(scheduled["energy_based_pue_proxy"])
pue_improvement = pue_baseline - pue_scheduled
pue_improvement_pct = 100 * pue_improvement / pue_baseline if pue_baseline else 0.0

peak_temp_baseline = float(baseline["peak_temp_c"])
peak_temp_scheduled = float(scheduled["peak_temp_c"])
peak_temp_avoided = peak_temp_baseline - peak_temp_scheduled

hot_baseline = float(baseline["hot_samples_ge_55c"])
hot_scheduled = float(scheduled["hot_samples_ge_55c"])
hot_samples_avoided = hot_baseline - hot_scheduled
hot_samples_avoided_pct = 100 * hot_samples_avoided / hot_baseline if hot_baseline else 0.0

shifted_energy = float(scheduled["shifted_energy_kwh"])
thermal_defers = float(scheduled["thermal_defers"])
price_defers = float(scheduled["price_defers"])

jobs_completed_baseline = float(baseline["jobs_completed"])
jobs_completed_scheduled = float(scheduled["jobs_completed"])
jobs_completed_delta = jobs_completed_scheduled - jobs_completed_baseline

print("\n=== NUMERIC COMPARISON: WITH VS WITHOUT SCHEDULER ===\n")

print(f"Cost without scheduler:           ${cost_baseline:.6f}")
print(f"Cost with scheduler:              ${cost_scheduled:.6f}")
print(f"Total cost savings:               ${cost_savings:.6f}")
print(f"Percent cost savings:             {cost_savings_pct:.2f}%\n")

print(f"Wall energy without scheduler:    {wall_energy_baseline:.6f} kWh")
print(f"Wall energy with scheduler:       {wall_energy_scheduled:.6f} kWh")
print(f"Wall energy savings:              {wall_energy_savings:.6f} kWh")
print(f"Percent wall energy savings:      {wall_energy_savings_pct:.2f}%\n")

print(f"Compute energy without scheduler: {compute_energy_baseline:.6f} kWh")
print(f"Compute energy with scheduler:    {compute_energy_scheduled:.6f} kWh")
print(f"Compute energy savings:           {compute_energy_savings:.6f} kWh")
print(f"Percent compute energy savings:   {compute_energy_savings_pct:.2f}%\n")

print(f"PUE without scheduler:            {pue_baseline:.6f}")
print(f"PUE with scheduler:               {pue_scheduled:.6f}")
print(f"PUE improvement:                  {pue_improvement:.6f}")
print(f"Percent PUE improvement:          {pue_improvement_pct:.2f}%\n")

print(f"Peak temp without scheduler:      {peak_temp_baseline:.3f} C")
print(f"Peak temp with scheduler:         {peak_temp_scheduled:.3f} C")
print(f"Peak temp avoided:                {peak_temp_avoided:.3f} C\n")

print(f"Hot samples without scheduler:    {hot_baseline:.0f}")
print(f"Hot samples with scheduler:       {hot_scheduled:.0f}")
print(f"Hot samples avoided:              {hot_samples_avoided:.0f}")
print(f"Percent hot samples avoided:      {hot_samples_avoided_pct:.2f}%\n")

print(f"Energy shifted by scheduler:      {shifted_energy:.6f} kWh")
print(f"Thermal defers:                   {thermal_defers:.0f}")
print(f"Price defers:                     {price_defers:.0f}\n")

print(f"Jobs completed without scheduler: {jobs_completed_baseline:.0f}")
print(f"Jobs completed with scheduler:    {jobs_completed_scheduled:.0f}")
print(f"Completed jobs delta:             {jobs_completed_delta:.0f}")

print("\n=== LONG-TERM DATA CENTER IMPACT ESTIMATE ===\n")

demo_hours = TOTAL_TEST_MINUTES / 60.0
annual_scale_factor = HOURS_IN_YEAR / demo_hours

annual_cost_savings_node = cost_savings * annual_scale_factor
annual_energy_savings_node = wall_energy_savings * annual_scale_factor
shifted_energy_annual_node = shifted_energy * annual_scale_factor

dc_annual_cost_savings = annual_cost_savings_node * NODE_COUNT
dc_annual_energy_savings = annual_energy_savings_node * NODE_COUNT
dc_shifted_energy_annual = shifted_energy_annual_node * NODE_COUNT

print(f"Demo duration:                    {demo_hours:.2f} hours")
print(f"Scaling factor to yearly:         x{annual_scale_factor:.1f}\n")

print("---- PER NODE (ANNUALIZED) ----")
print(f"Annual cost savings per node:     ${annual_cost_savings_node:.2f}")
print(f"Annual energy savings per node:   {annual_energy_savings_node:.2f} kWh")
print(f"Annual peak energy shifted/node:  {shifted_energy_annual_node:.2f} kWh\n")

print("---- DATA CENTER (1000 NODES) ----")
print(f"Annual cost savings:              ${dc_annual_cost_savings:,.2f}")
print(f"Annual energy savings:            {dc_annual_energy_savings:,.2f} kWh")
print(f"Annual peak energy shifted:       {dc_shifted_energy_annual:,.2f} kWh")
