#!/usr/bin/env python3
"""
GPU Temps -> PLC via Modbus TCP (Do-more / BRX)
----------------------------------------------
Reads GPU temperatures using nvidia-smi and writes them as 16-bit holding registers.

IMPORTANT (Do-more mapping):
- Modbus holding registers are addressed as offsets in most libraries (0-based).
  Example: Holding Register 40001 == address 0.
- Do-more V-memory (e.g., V2000) is NOT always directly Modbus-addressable unless you
  explicitly map/publish it in the PLC program.
- If you publish temps to MH1..MH3 (common), you likely want --register-start 0.
- If you have a custom mapping where your target starts at some other holding register,
  set --register-start accordingly.

Default behavior:
- Writes 3 temps starting at REGISTER_START (default 2000, matching your original script)
- Reads back the same 3 registers and prints them so you can confirm what the PLC received.

Usage examples:
  python3 gpu_temps_to_plc_modbus.py --plc-ip 192.168.1.20 --register-start 0
  python3 gpu_temps_to_plc_modbus.py --plc-ip 192.168.1.20 --register-start 2000 --poll 2

Requirements:
  pip install pymodbus --break-system-packages
  nvidia-smi must be available (NVIDIA drivers installed)
"""

import argparse
import subprocess
import time
import sys
from typing import List

from pymodbus.client import ModbusTcpClient


def _write_registers(client: ModbusTcpClient, address: int, values: List[int], unit_id: int):
    """Compatibility wrapper for pymodbus 'unit' vs 'slave' keyword."""
    try:
        return client.write_registers(address, values, unit=unit_id)
    except TypeError:
        return client.write_registers(address, values, slave=unit_id)


def _read_holding_registers(client: ModbusTcpClient, address: int, count: int, unit_id: int):
    """Compatibility wrapper for pymodbus 'unit' vs 'slave' keyword."""
    try:
        return client.read_holding_registers(address, count, unit=unit_id)
    except TypeError:
        return client.read_holding_registers(address, count, slave=unit_id)


def get_gpu_temps(num_gpus: int) -> List[int]:
    """
    Read GPU temperatures using nvidia-smi.
    Returns a list of integer temps in Celsius, one per GPU.
    Pads with 0 if fewer than num_gpus are detected.
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
            return [0] * num_gpus

        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        temps = [int(t) for t in lines]

        while len(temps) < num_gpus:
            temps.append(0)

        temps = temps[:num_gpus]

        # Ensure values fit in unsigned 16-bit holding registers
        temps = [max(0, min(65535, int(t))) for t in temps]
        return temps

    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found. Are NVIDIA drivers installed?")
        return [0] * num_gpus
    except Exception as e:
        print(f"[ERROR] Failed to read GPU temps: {e}")
        return [0] * num_gpus


def parse_args():
    p = argparse.ArgumentParser(description="Send GPU temps to a Do-more/BRX PLC via Modbus TCP.")
    p.add_argument("--plc-ip", default="192.168.1.20", help="PLC IP address (default: 192.168.1.20)")
    p.add_argument("--plc-port", type=int, default=502, help="PLC Modbus TCP port (default: 502)")
    p.add_argument("--unit-id", type=int, default=1, help="Modbus Unit ID / Slave ID (default: 1)")
    p.add_argument("--register-start", type=int, default=2000,
                   help="Starting holding-register OFFSET to write (default: 2000). "
                        "Common alt: 0 for 40001 (often MH1).")
    p.add_argument("--num-gpus", type=int, default=3, help="Number of GPUs to read/write (default: 3)")
    p.add_argument("--poll", type=float, default=5.0, help="Seconds between updates (default: 5)")
    p.add_argument("--no-readback", action="store_true",
                   help="Disable readback verification after writing.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 68)
    print("GPU Temp -> PLC Modbus TCP")
    print(f"PLC: {args.plc_ip}:{args.plc_port}  |  Unit ID: {args.unit_id}")
    print(f"Write holding-register OFFSETS: {args.register_start} .. {args.register_start + args.num_gpus - 1}")
    print(f"Poll interval: {args.poll:.2f}s  |  Readback: {'OFF' if args.no_readback else 'ON'}")
    print("=" * 68)

    client = ModbusTcpClient(host=args.plc_ip, port=args.plc_port)

    if not client.connect():
        print(f"[ERROR] Could not connect to PLC at {args.plc_ip}:{args.plc_port}")
        print("  - Check PLC IP address")
        print("  - Check network cable / switch")
        print("  - Verify Modbus TCP server enabled on PLC (port 502)")
        print(f"  - Try: ping {args.plc_ip}")
        sys.exit(1)

    print(f"[OK] Connected to PLC at {args.plc_ip}\n")

    try:
        while True:
            temps = get_gpu_temps(args.num_gpus)

            # Write all temps in a single Modbus write
            wr = _write_registers(client, args.register_start, temps, args.unit_id)

            if wr.isError():
                print(f"[ERROR] Modbus write failed: {wr}")
            else:
                line = "  ".join([f"GPU{i+1}:{temps[i]}C" for i in range(args.num_gpus)])
                print(f"[WRITE OK] {line}  -> offsets {args.register_start}-{args.register_start + args.num_gpus - 1}", end="")

                if args.no_readback:
                    print("")
                else:
                    rb = _read_holding_registers(client, args.register_start, args.num_gpus, args.unit_id)
                    if rb.isError():
                        print(f"  |  [READBACK ERROR] {rb}")
                    else:
                        print(f"  |  [READBACK] {rb.registers}")

            time.sleep(args.poll)

    except KeyboardInterrupt:
        print("\n[STOPPED] Script terminated by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        client.close()
        print("[CLOSED] Modbus connection closed.")


if __name__ == "__main__":
    main()
