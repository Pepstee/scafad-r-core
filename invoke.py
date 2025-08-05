# invoke.py
import json
import random
import subprocess
import time
import string
import os
import argparse
from datetime import datetime

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Invoke SCAFAD Lambda with random payloads.")
parser.add_argument("--n", type=int, default=10, help="Number of invocations to simulate")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

N = args.n
random.seed(args.seed)

# --- Directory Setup ---
os.makedirs("telemetry/payloads", exist_ok=True)
master_log_path = "telemetry/payloads/invocation_master_log.jsonl"

# --- Test Configs ---
anomalies = ["cold_start", "cpu_burst", "benign"]
profiles = ["func_A", "func_B", "func_C"]
phases = ["init", "invoke", "shutdown"]

start_time = time.time()
print(f"📡 Starting {N} invocations...\n")

for i in range(N):
    payload = {
        "anomaly": random.choice(anomalies),
        "function_profile_id": random.choice(profiles),
        "execution_phase": random.choice(phases),
        "concurrency_id": ''.join(random.choices(string.ascii_uppercase, k=3)),
        "force_starvation": random.choice([True, False, False, False]),
        "invocation_timestamp": time.time()
    }

    # Save current payload to timestamped file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    payload_filename = f"telemetry/payloads/payload_{i+1:03d}_{ts}.json"
    with open(payload_filename, "w") as f:
        json.dump(payload, f, indent=2)

    # Also update shared payload.json for SAM to use
    with open("payload.json", "w") as f:
        json.dump(payload, f)

    # Append to master JSONL log
    with open(master_log_path, "a") as logf:
        logf.write(json.dumps(payload) + "\n")

    print(f"▶️  [{i+1}/{N}] Invoking with payload: {payload}")
    subprocess.run(
        "sam local invoke HelloWorldFunction --event payload.json",
        shell=True
    )
    time.sleep(0.2)

elapsed = round(time.time() - start_time, 2)
print(f"\n✅ All {N} invocations completed in {elapsed} seconds.")
print(f"📁 Payloads saved in: telemetry/payloads/")
print(f"📃 Master payload log: {master_log_path}")
