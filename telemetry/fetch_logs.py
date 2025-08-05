# telemetry/fetch_logs.py
import json
import csv
import subprocess

log_group = "/aws/lambda/scafad-test-stack-HelloWorldFunction-k79tX3iBcK74"  # Update if needed

# Step 1: Pull logs via AWS CLI
result = subprocess.run(
    ["aws", "logs", "filter-log-events", "--log-group-name", log_group, "--limit", "100"],
    capture_output=True, text=True
)

log_data = json.loads(result.stdout)
events = log_data.get("events", [])

parsed_logs = []
fieldnames = set()

side_channel_logs = []
primary_logs = []
fieldnames = set()

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
            continue


# Step 3: Save as CSV
if not parsed_logs:
    print("No structured logs found.")
else:
    with open("lambda_telemetry.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(parsed_logs)

    print(f"✅ Extracted {len(parsed_logs)} structured logs to lambda_telemetry.csv")

# Save main structured logs
if primary_logs:
    with open("lambda_telemetry.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(primary_logs)
    print(f"✅ Extracted {len(primary_logs)} structured logs to lambda_telemetry.csv")

# Save side-channel traces
if side_channel_logs:
    with open("side_channel_trace.log", "w") as f:
        for entry in side_channel_logs:
            f.write(entry["raw_trace"] + "\n")
    print(f"✅ Extracted {len(side_channel_logs)} side traces to side_channel_trace.log")

