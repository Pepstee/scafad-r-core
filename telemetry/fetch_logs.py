# telemetry/fetch_logs.py

import json
import csv
import subprocess
import os
import time
from datetime import datetime
from collections import Counter

# --- AWS Lambda Log Group ---
log_group = "/aws/lambda/scafad-test-stack-HelloWorldFunction-k79tX3iBcK74"  # ✅ Change if needed

# --- Ensure Archive Folder Exists ---
os.makedirs("telemetry/archive", exist_ok=True)

# --- Fetch Logs from CloudWatch ---
print("📡 Fetching logs from CloudWatch...")
result = subprocess.run(
    [
        "aws", "logs", "filter-log-events",
        "--log-group-name", log_group,
        "--limit", "100",
        "--start-time", str(int((time.time() - 3600) * 1000))  # ✅ Last 60 minutes
    ],
    capture_output=True,
    text=True
)

# --- Handle CLI Errors ---
if not result.stdout:
    print("❌ No output received. Check AWS credentials, region, or log group name.")
    print(f"🔎 stderr: {result.stderr.strip()}")
    exit(1)

try:
    log_data = json.loads(result.stdout)
except json.JSONDecodeError:
    print("❌ Failed to parse AWS CLI output as JSON.")
    exit(1)

events = log_data.get("events", [])

# --- Log Containers ---
side_channel_logs = []
primary_logs = []
malformed_count = 0
fieldnames = set()

# --- Parse Logs ---
for event in events:
    message = event.get("message", "")
    if message.startswith("[SCAFAD_TRACE]"):
        side_channel_logs.append({"raw_trace": message})
    else:
        try:
            log_entry = json.loads(message)
            primary_logs.append(log_entry)
            fieldnames.update(log_entry.keys())
        except json.JSONDecodeError:
            malformed_count += 1

# --- Filenames for Archive ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"telemetry/archive/lambda_telemetry_{timestamp}.csv"
trace_filename = f"telemetry/archive/side_channel_trace_{timestamp}.log"

# --- Save Structured Logs to CSV ---
if primary_logs:
    with open("telemetry/lambda_telemetry.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(primary_logs)

    with open(csv_filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(primary_logs)

    print(f"✅ Extracted {len(primary_logs)} structured logs to:")
    print(f"   → telemetry/lambda_telemetry.csv")
    print(f"   → {csv_filename}")
else:
    print("⚠️ No structured logs found.")

# --- Save Side Channel Logs ---
if side_channel_logs:
    with open("telemetry/side_channel_trace.log", "w", encoding="utf-8") as f:
        for entry in side_channel_logs:
            f.write(entry["raw_trace"] + "\n")

    with open(trace_filename, "w", encoding="utf-8") as f:
        for entry in side_channel_logs:
            f.write(entry["raw_trace"] + "\n")

    print(f"✅ Extracted {len(side_channel_logs)} side traces to:")
    print(f"   → telemetry/side_channel_trace.log")
    print(f"   → {trace_filename}")
else:
    print("⚠️ No side-channel traces found.")

# --- Summary Stats ---
if primary_logs:
    counter = Counter()
    fallback_count = 0

    for log in primary_logs:
        anomaly = log.get("anomaly_type", "undefined")
        fallback = log.get("fallback_mode", False)
        counter[anomaly] += 1
        if fallback:
            fallback_count += 1

    print("\n📊 Log Summary:")
    for anomaly, count in counter.items():
        print(f"   • {anomaly:20} → {count} logs")
    print(f"   • fallback_mode=True       → {fallback_count} logs")
    print(f"   • malformed logs skipped   → {malformed_count}")
