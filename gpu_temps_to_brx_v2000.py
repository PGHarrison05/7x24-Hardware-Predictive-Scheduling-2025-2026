#!/usr/bin/env python3

"""

GPU Temps -> BRX PLC (Do-more) via Modbus TCP Holding Registers
 
EXPLICIT MAPPING:

  GPU0 Temp (°C)  --->  V2000

  GPU1 Temp (°C)  --->  V2001

  GPU2 Temp (°C)  --->  V2002
 
NOTE:

- Python cannot write to "V2000" by name.

- It writes to Modbus Holding Registers at an ADDRESS.

- PLC maps those registers into V-memory.
 
The only thing you may need to adjust is REGISTER_BASE.

"""
 
import subprocess

import time

from pymodbus.client import ModbusTcpClient
 
# ----------------------------

# USER SETTINGS

# ----------------------------

PLC_IP = "192.168.1.20"

PLC_PORT = 502

UNIT_ID = 255
 
POLL_SEC = 1.0
 
# REGISTER_BASE = Modbus register address corresponding to V2000

# Try 0 first. If V2000 doesn't change, try 2000.

REGISTER_BASE = 2000

# ----------------------------
 
 
def read_gpu_temps_c() -> list[int]:

    """

    Reads GPU temps from NVIDIA GPUs on Linux using nvidia-smi.

    Returns list like [gpu0_temp, gpu1_temp, gpu2_temp, ...]

    """

    cmd = [

        "nvidia-smi",

        "--query-gpu=temperature.gpu",

        "--format=csv,noheader,nounits",

    ]

    out = subprocess.check_output(cmd, text=True).strip()

    temps = [int(line.strip()) for line in out.splitlines() if line.strip()]

    return temps
 
 
def clamp_u16(x: int) -> int:

    """Ensure values fit inside a 16-bit unsigned Modbus register."""

    if x < 0:

        return 0

    if x > 65535:

        return 65535

    return x
 
 
def write_v2000_v2002(client: ModbusTcpClient, gpu0: int, gpu1: int, gpu2: int) -> None:

    """

    Writes 3 consecutive holding registers:
 
      REGISTER_BASE + 0  -> V2000  (GPU0)

      REGISTER_BASE + 1  -> V2001  (GPU1)

      REGISTER_BASE + 2  -> V2002  (GPU2)

    """

    regs = [clamp_u16(gpu0), clamp_u16(gpu1), clamp_u16(gpu2)]
 
    rr = client.write_registers(

        address=REGISTER_BASE,

        values=regs,

        slave=UNIT_ID

    )
    print("Write Response:",rr)
 
    if rr.isError():

        raise RuntimeError(f"Modbus write failed: {rr}")
 
 
def main():

    client = ModbusTcpClient(PLC_IP, port=PLC_PORT)
 
    if not client.connect():

        raise RuntimeError(f"Could not connect to PLC at {PLC_IP}:{PLC_PORT}")
 
    print(f"Connected to PLC at {PLC_IP}:{PLC_PORT}")

    print("EXPLICIT MAPPING:")

    print(f"  GPU0 -> V2000 (Modbus reg {REGISTER_BASE + 0})")

    print(f"  GPU1 -> V2001 (Modbus reg {REGISTER_BASE + 1})")

    print(f"  GPU2 -> V2002 (Modbus reg {REGISTER_BASE + 2})")

    print("Press Ctrl+C to stop.\n")
 
    try:

        while True:

            temps = read_gpu_temps_c()
 
            if len(temps) < 3:

                print(f"Only found {len(temps)} GPU(s): {temps}. Need at least 3.")

                time.sleep(POLL_SEC)

                continue
 
            gpu0, gpu1, gpu2 = temps[0], temps[1], temps[2]
 
            write_v2000_v2002(client, gpu0, gpu1, gpu2)
 
            print(f"Sent Temps: GPU0={gpu0}C -> V2000 | GPU1={gpu1}C -> V2001 | GPU2={gpu2}C -> V2002")

            time.sleep(POLL_SEC)
 
    except KeyboardInterrupt:

        print("\nStopping program...")
 
    finally:

        client.close()

        print("Disconnected from PLC.")
 
 
if __name__ == "__main__":

    main()

 
