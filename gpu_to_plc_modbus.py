#!/usr/bin/env python3
import subprocess
import time
from pymodbus.client import ModbusTcpClient

# ================= USER SETTINGS =================
PLC_IP = "192.168.1.20"   
PLC_PORT = 502
UNIT_ID = 1               # try 0 if 1 doesn't work
POLL_SEC = 1.0
NUM_GPUS = 3

# Holding register mapping (pymodbus uses 0-based addresses)
# 40001 -> address 0
TEMP_ADDR0 = 0    # 40001..40003
PWR_ADDR0  = 10   # 40011..40013
UTIL_ADDR0 = 20   # 40021..40023
# =================================================

def read_gpu_stats():
    """
    Reads GPU temperature, power, utilization from nvidia-smi
    Returns list of tuples: [(tempC, powerW, utilPct), ...]
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True).strip().splitlines()

    stats = []
    for line in output:
        parts = [p.strip() for p in line.split(",")]
        temp_c = float(parts[0])
        power_w = float(parts[1])
        util = int(float(parts[2]))
        stats.append((temp_c, power_w, util))

    return stats[:NUM_GPUS]

def main():
    print(f"Connecting to PLC at {PLC_IP}:{PLC_PORT} ...")
    client = ModbusTcpClient(PLC_IP, port=PLC_PORT)
    if not client.connect():
        raise SystemExit("ERROR: Could not connect to PLC. Check IP/cable/subnet and that Modbus TCP is enabled.")

    print("Connected. Writing GPU telemetry to PLC registers. Ctrl+C to stop.")
    try:
        while True:
            stats = read_gpu_stats()

            temps = [int(round(t * 10)) for (t, _, _) in stats]   # °C x10
            pwrs  = [int(round(p * 10)) for (_, p, _) in stats]   # W x10
            utils = [int(u) for (_, _, u) in stats]               # %

            r1 = client.write_registers(TEMP_ADDR0, temps, slave=UNIT_ID)
            r2 = client.write_registers(PWR_ADDR0,  pwrs,  slave=UNIT_ID)
            r3 = client.write_registers(UTIL_ADDR0, utils, slave=UNIT_ID)

            if r1.isError() or r2.isError() or r3.isError():
                print("Write error:", r1, r2, r3)
            else:
                print(f"Wrote temps={temps} pwrs={pwrs} utils={utils}")

            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.close()

if __name__ == "__main__":
    main()

