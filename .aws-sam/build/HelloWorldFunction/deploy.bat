@echo off
REM deploy.bat — Windows-friendly wrapper for SCAFAD Makefile tasks

REM --- Build Lambda Locally ---
echo [1/5] Building Lambda...
sam build

REM --- Deploy Lambda to AWS ---
echo [2/5] Deploying Lambda...
sam deploy --guided

REM --- Simulate 10 Invocations ---
echo [3/5] Simulating anomaly invocations...
python invoke.py --n=10

REM --- Fetch Logs from CloudWatch ---
echo [4/5] Fetching telemetry logs...
python telemetry/fetch_logs.py

REM --- Run Test Suite ---
echo [5/5] Running unit tests...
pytest tests/unit/test_lambda.py -v

REM --- Done ---
echo.
echo ✅ Deployment + simulation complete. Logs saved to telemetry/archive.
pause
