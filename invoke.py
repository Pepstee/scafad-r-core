# invoke.py
import json
import random
import subprocess
import time
import string

anomalies = ["cold_start", "cpu_burst", "benign"]
N = 10  # number of invocations

for i in range(N):
    payload = {
        "anomaly": random.choice(anomalies),
        "function_profile_id": random.choice(["func_A", "func_B", "func_C"]),
        "execution_phase": random.choice(["init", "invoke", "shutdown"]),
        "concurrency_id": ''.join(random.choices(string.ascii_uppercase, k=3)),
        "force_starvation": random.choice([True, False, False, False])  # ~25% chance
    }

    with open("payload.json", "w") as f:
        json.dump(payload, f)

    print(f"Invoking with: {payload}")
    subprocess.run(
        "sam local invoke HelloWorldFunction --event payload.json",
        shell=True  # ✅ required on Windows
    )
    time.sleep(1)
