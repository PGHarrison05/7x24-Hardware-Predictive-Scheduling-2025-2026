#!/usr/bin/env python3
import subprocess
import time
from pymodbus.client import ModbusTcpClient

# ========= SETTINGS =========
PLC_IP = "192.168.1.20"
PLC_PORT = 502
UNIT_ID = 1          # try 0 if 1 doesn't work
POLL_SEC = 1.0
NUM_GPUS = 3

# 42001 corresponds to V2000 per your PLC mapping.
# pymodbus uses 0-based holding register addresses:
# 40001 -> 0, so 42001 -> 2000
GPU_TEMP_BASE_ADDR = 2000  # writes to 42001 (V2000)
SCALE_TEMP_X10 = True      # store temp as (degC * 10) integer
# ===========================

def read_gpu_temps():
    """
    Returns list of GPU temps in C as floats, one per GPU.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu",

        "--format=csv,noheader,nounits",
    ]
    lines = subprocess.check_output(cmd, text=True).strip().splitlines()
    temps = [float(x.strip()) for x in lines]
    return temps[:NUM_GPUS]

def main():
    print(f"Connecting to PLC Modbus TCP at {PLC_IP}:{PLC_PORT} ...")
    client = ModbusTcpClient(PLC_IP, port=PLC_PORT)
    if not client.connect():
        raise SystemExit("ERROR: Could not connect to PLC (IP/cable/subnet/Modbus server).")

    print("Connected. Writing GPU temps to holding registers starting at 42001 (V2000). Ctrl+C to stop.")
    try:
        while True:
            temps = read_gpu_temps()
            if SCALE_TEMP_X10:
                payload = [int(round(t * 10)) for t in temps]   # e.g., 40.2C -> 402
            else:
                payload = [int(round(t)) for t in temps]        # e.g., 40.2C -> 40

            # Write GPU0->42001, GPU1->42002, GPU2->42003
            wr = client.write_registers(GPU_TEMP_BASE_ADDR, payload, slave=UNIT_ID)

            # Read back immediately to confirm (great for debugging)
            rd = client.read_holding_registers(GPU_TEMP_BASE_ADDR, len(payload), slave=UNIT_ID)

            if wr.isError():
                print("WRITE ERROR:", wr)
            elif rd.isError():
                print("READBACK ERROR:", rd)
            else:
                print(f"Wrote temps={payload} (x10={SCALE_TEMP_X10}) | Readback={rd.registers}")

            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.close()

if __name__ == "__main__":
    main()
