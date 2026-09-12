#!/usr/bin/env python3

import subprocess
import time

def c_to_f(c):
    return (c * 9/5) + 32

def get_gpu_stats():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,temperature.gpu,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    
    gpus = []
    for line in lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        
        idx = int(parts[0])
        temp_c = float(parts[1])
        temp_f = c_to_f(temp_c)
        util = float(parts[2])
        power = float(parts[3])
        
        gpus.append((idx, temp_c, temp_f, util, power))
    
    return gpus

print("GPU Monitor (Ctrl+C to stop)\n")

try:
    while True:
        stats = get_gpu_stats()
        
        print("-" * 50)
        for gpu in stats:
            idx, temp_c, temp_f, util, power = gpu
            print(f"GPU {idx}: {temp_f:.1f}°F ({temp_c:.1f}°C) | Util: {util:.0f}% | Power: {power:.1f} W")
        
        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopped.")
