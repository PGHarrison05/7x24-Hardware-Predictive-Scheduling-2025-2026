#!/usr/bin/env python3
"""
GPU Temperature to PLC via Modbus TCP
--------------------------------------
Reads GPU temperatures from all 3 GPUs using nvidia-smi
and writes them to the Do-More BRX PLC via Modbus TCP.

Register Mapping:
  V2000 (Modbus register 2000) --> GPU1 Temp (C)
  V2001 (Modbus register 2001) --> GPU2 Temp (C)
  V2002 (Modbus register 2002) --> GPU3 Temp (C)

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
REGISTER_START  = 0             # Maps to V2000 in Do-More
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

        # Pad to NUM_GPUS if fewer GPUs detected
        while len(temps) < NUM_GPUS:
            temps.append(0)

        return temps[:NUM_GPUS]

    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found. Are NVIDIA drivers installed?")
        return [0] * NUM_GPUS
    except Exception as e:
        print(f"[ERROR] Failed to read GPU temps: {e}")
        return [0] * NUM_GPUS


def main():
    print("=" * 50)
    print("  GPU Temp -> PLC Modbus TCP")
    print(f"  PLC IP : {PLC_IP}:{PLC_PORT}")
    print(f"  Registers: V{REGISTER_START} - V{REGISTER_START + NUM_GPUS - 1}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print("=" * 50)

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
            temps = get_gpu_temps()

            # Write all three temps in a single Modbus write
            result = client.write_registers(REGISTER_START, temps)

            if result.isError():
                print(f"[ERROR] Modbus write failed: {result}")
            else:
                print(f"GPU1: {temps[0]}C  |  GPU2: {temps[1]}C  |  GPU3: {temps[2]}C  --> Written to V2000-V2002")

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
