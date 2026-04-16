# Makefile for SCAFAD-L0: AWS SAM Deployment & Testing

# --- Configurable Variables ---
FUNCTION_NAME=HelloWorldFunction
EVENT_FILE=payload.json
LOG_GROUP=/aws/lambda/scafad-test-stack-$(FUNCTION_NAME)
ARCHIVE_DIR=telemetry/archive

# --- Build Lambda Locally ---
build:
	sam build

# --- Deploy Lambda to AWS ---
deploy:
	sam deploy --guided

# --- Invoke Lambda Locally with Test Payload ---
invoke:
	sam local invoke $(FUNCTION_NAME) --event $(EVENT_FILE)

# --- Fetch Telemetry Logs from CloudWatch ---
logs:
	python telemetry/fetch_logs.py

# --- Simulate N Anomaly Events ---
invoke-n:
	python invoke.py --n=10

# --- Run Pytest Test Suite ---
test:
	pytest tests/unit/test_lambda.py -v

# --- Clean SAM Build Artifacts ---
clean:
	rm -rf .aws-sam __pycache__ .pytest_cache telemetry/lambda_telemetry.csv telemetry/side_channel_trace.log $(ARCHIVE_DIR)/*.csv $(ARCHIVE_DIR)/*.log telemetry/payloads/*.json telemetry/payloads/invocation_master_log.jsonl

# --- Full Workflow: Build → Deploy → Simulate → Fetch Logs ---
full:
	make build && make deploy && make invoke-n && make logs

# --- Quick Re-invoke from Last Payload ---
reinvoke:
	ls -1 telemetry/payloads/*.json | sort | tail -n 1 | xargs -I {} cp {} payload.json && make invoke

# --- View Raw Logs in Terminal ---
logs-raw:
	aws logs filter-log-events --log-group-name $(LOG_GROUP) --limit 20 --output text

# --- View Archived CSV Log Summary ---
summarise:
	python3 -c "import pandas as pd; df = pd.read_csv('telemetry/lambda_telemetry.csv'); print(df['anomaly_type'].value_counts())"
