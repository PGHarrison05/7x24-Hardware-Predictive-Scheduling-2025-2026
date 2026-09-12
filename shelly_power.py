#!/usr/bin/env python3

import requests
import sys

SHELLY_IP = "149.61.249.0"
URL = f"http://{SHELLY_IP}/rpc/Shelly.GetStatus"

def main():
    try:
        response = requests.get(URL, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Most Shelly smart plugs report active power her
        if "switch:0" in data and "apower" in data["switch:0"]:
            power_watts = data["switch:0"]["apower"]
            print(f"Total power: {power_watts} W")
        else:
            print("Could not find power field in response.")
            print("Full response:")
            print(data)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Invalid JSON response: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
