#!/usr/bin/env python3
"""
GPU Temperature + Power to PLC via Modbus TCP
----------------------------------------------
Reads GPU temperatures and power draw from all 3 GPUs using nvidia-smi
and writes them to the Do-More BRX PLC via Modbus TCP.

Register Mapping (interleaved per GPU):
  MHR0 --> GPU1 Temp (C)     [V2000]
  MHR1 --> GPU1 Power (W)    [V2001]
  MHR2 --> GPU2 Temp (C)     [V2002]
  MHR3 --> GPU2 Power (W)    [V4000]
  MHR4 --> GPU3 Temp (C)     [V4001]
  MHR5 --> GPU3 Power (W)    [V4002]

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
PLC_IP         = "192.168.1.20"   # Do-More BRX PLC
PLC_PORT       = 502
REGISTER_START = 0                # MHR0 = GPU1 Temp
POLL_INTERVAL  = 5                # Seconds between updates
NUM_GPUS       = 3
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
    """
    Read GPU power draw using nvidia-smi.
    Returns a list of integer power values in Watts, one per GPU.
    Pads with 0 if fewer than NUM_GPUS are detected.
    """
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
    print(f"  Register layout (interleaved):")
    for i in range(NUM_GPUS):
        print(f"    MHR{i*2}  --> GPU{i+1} Temp (C)")
        print(f"    MHR{i*2+1}  --> GPU{i+1} Power (W)")
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

            # Interleave: [temp1, pwr1, temp2, pwr2, temp3, pwr3]
            # This maps correctly to MHR0-MHR5
            interleaved = []
            for i in range(NUM_GPUS):
                interleaved.append(temps[i])
                interleaved.append(powers[i])

            # Single write of all 6 registers starting at MHR0
            result = client.write_registers(REGISTER_START, interleaved)

            if result.isError():
                print(f"[ERROR] Modbus write failed: {result}")
            else:
                print(
                    f"GPU1: {temps[0]:3}C  {powers[0]:4}W  |  "
                    f"GPU2: {temps[1]:3}C  {powers[1]:4}W  |  "
                    f"GPU3: {temps[2]:3}C  {powers[2]:4}W  "
                    f"--> MHR0-MHR5"
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
