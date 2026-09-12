#!/usr/bin/env python3
"""
GPU Temperature + Power to PLC via Modbus TCP
----------------------------------------------
Reads GPU temperatures and power draw from all 3 GPUs using nvidia-smi
and writes them to the Do-More BRX PLC via Modbus TCP.

Register Mapping:
  MHR1 --> GPU1 Temp (C)
  MHR2 --> GPU2 Temp (C)
  MHR3 --> GPU3 Temp (C)
  MHR4 --> GPU1 Power (W)
  MHR5 --> GPU2 Power (W)
  MHR6 --> GPU3 Power (W)

NOTE: pymodbus address 0 = MHR1 on the Do-More BRX (confirmed by testing).
  Temps write to address 0 (lands MHR1-3)
  Powers write to address 3 (lands MHR4-6)

Requirements:
  pip install pymodbus --break-system-packages
  nvidia-smi must be available (NVIDIA drivers installed)

Manhattan University - 7x24 Exchange Student Chapter
"""

import subprocess
import time
import sys
from pymodbus.client import ModbusTcpClient

# ── Configuration ─────────────────────────────────────────────────────────────
PLC_IP        = "192.168.1.20"   # Do-More BRX PLC
PLC_PORT      = 502
TEMP_START    = 0                # pymodbus addr 0 = MHR1 on BRX
POWER_START   = 3                # pymodbus addr 3 = MHR4 on BRX
POLL_INTERVAL = 5                # Seconds between updates
NUM_GPUS      = 3
# ──────────────────────────────────────────────────────────────────────────────


def get_gpu_temps():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print(f"[ERROR] nvidia-smi (temp) failed: {result.stderr.strip()}")
            return [0] * NUM_GPUS
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        temps = [int(t) for t in lines]
        while len(temps) < NUM_GPUS:
            temps.append(0)
        return temps[:NUM_GPUS]
    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found. Are NVIDIA drivers installed?")
        return [0] * NUM_GPUS
    except Exception as e:
        print(f"[ERROR] Failed to read GPU temps: {e}")
        return [0] * NUM_GPUS


def get_gpu_power():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print(f"[ERROR] nvidia-smi (power) failed: {result.stderr.strip()}")
            return [0] * NUM_GPUS
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        powers = [round(float(p)) for p in lines]
        while len(powers) < NUM_GPUS:
            powers.append(0)
        return powers[:NUM_GPUS]
    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found. Are NVIDIA drivers installed?")
        return [0] * NUM_GPUS
    except Exception as e:
        print(f"[ERROR] Failed to read GPU power: {e}")
        return [0] * NUM_GPUS


def main():
    print("=" * 60)
    print("  GPU Temp + Power -> PLC Modbus TCP")
    print(f"  PLC IP : {PLC_IP}:{PLC_PORT}")
    print(f"  MHR1 = GPU1 Temp  | MHR2 = GPU2 Temp  | MHR3 = GPU3 Temp")
    print(f"  MHR4 = GPU1 Power | MHR5 = GPU2 Power | MHR6 = GPU3 Power")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print("=" * 60)

    client = ModbusTcpClient(host=PLC_IP, port=PLC_PORT)

    if not client.connect():
        print(f"[ERROR] Could not connect to PLC at {PLC_IP}:{PLC_PORT}")
        print("  - Check PLC IP address in this script")
        print("  - Check network cable / switch")
        print("  - Ping the PLC: ping " + PLC_IP)
        sys.exit(1)

    print(f"[OK] Connected to PLC at {PLC_IP}\n")

    try:
        while True:
            temps  = get_gpu_temps()
            powers = get_gpu_power()

            # Write temps to MHR1, MHR2, MHR3
            t_result = client.write_registers(TEMP_START, temps)
            # Write powers to MHR4, MHR5, MHR6
            p_result = client.write_registers(POWER_START, powers)

            if t_result.isError():
                print(f"[ERROR] Temp write failed: {t_result}")
            elif p_result.isError():
                print(f"[ERROR] Power write failed: {p_result}")
            else:
                print(
                    f"GPU1: {temps[0]:3}C  {powers[0]:4}W  |  "
                    f"GPU2: {temps[1]:3}C  {powers[1]:4}W  |  "
                    f"GPU3: {temps[2]:3}C  {powers[2]:4}W  "
                    f"--> MHR1-3 (temps)  MHR4-6 (power)"
                )

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOPPED] Script terminated by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        client.close()
        print("[CLOSED] Modbus connection closed.")


if __name__ == "__main__":
    main()
