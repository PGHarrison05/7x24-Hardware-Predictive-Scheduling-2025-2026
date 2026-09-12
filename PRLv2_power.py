#!/usr/bin/env python3
"""
GPU Temperature + Power to PLC via Modbus TCP
----------------------------------------------
Reads GPU temperatures and power draw from all 3 GPUs using nvidia-smi
and writes them to the Do-More BRX PLC via Modbus TCP.

Register Mapping:
  MHR0 (Modbus register 0) --> GPU1 Temp (C)
  MHR1 (Modbus register 1) --> GPU2 Temp (C)
  MHR2 (Modbus register 2) --> GPU3 Temp (C)
  MHR3 (Modbus register 3) --> GPU1 Power (W)
  MHR4 (Modbus register 4) --> GPU2 Power (W)
  MHR5 (Modbus register 5) --> GPU3 Power (W)

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
PLC_IP          = "192.168.1.20"   # Do-More BRX PLC
PLC_PORT        = 502
TEMP_REGISTER   = 0                # MHR0 = GPU1 Temp, MHR1 = GPU2 Temp, MHR2 = GPU3 Temp
POWER_REGISTER  = 3                # MHR3 = GPU1 Power, MHR4 = GPU2 Power, MHR5 = GPU3 Power
POLL_INTERVAL   = 5                # Seconds between updates
NUM_GPUS        = 3
# ──────────────────────────────────────────────────────────────────────────────


def get_gpu_temps():
    """
    Read GPU temperatures using nvidia-smi.
    Returns a list of integer temps in Celsius, one per GPU.
    Pads with 0 if fewer than NUM_GPUS are detected.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"[ERROR] nvidia-smi failed: {result.stderr.strip()}")
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
    """
    Read GPU power draw using nvidia-smi.
    Returns a list of integer power values in Watts, one per GPU.
    Pads with 0 if fewer than NUM_GPUS are detected.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"[ERROR] nvidia-smi (power) failed: {result.stderr.strip()}")
            return [0] * NUM_GPUS

        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        # power.draw returns floats like "150.23", round to int for Modbus register
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
    print(f"  Temp Registers : MHR{TEMP_REGISTER}  - MHR{TEMP_REGISTER  + NUM_GPUS - 1}")
    print(f"  Power Registers: MHR{POWER_REGISTER} - MHR{POWER_REGISTER + NUM_GPUS - 1}")
    print(f"  Poll interval  : {POLL_INTERVAL}s")
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

            # Write temps to MHR0-MHR2
            temp_result = client.write_registers(TEMP_REGISTER, temps)
            if temp_result.isError():
                print(f"[ERROR] Modbus temp write failed: {temp_result}")

            # Write power to MHR3-MHR5
            power_result = client.write_registers(POWER_REGISTER, powers)
            if power_result.isError():
                print(f"[ERROR] Modbus power write failed: {power_result}")

            if not temp_result.isError() and not power_result.isError():
                print(
                    f"GPU1: {temps[0]:3}C  {powers[0]:4}W  |  "
                    f"GPU2: {temps[1]:3}C  {powers[1]:4}W  |  "
                    f"GPU3: {temps[2]:3}C  {powers[2]:4}W  "
                    f"--> MHR0-2 (temps), MHR3-5 (power)"
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
