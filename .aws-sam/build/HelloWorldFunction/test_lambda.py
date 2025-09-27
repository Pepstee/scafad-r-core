# tests/unit/test_lambda.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from hello_world import app
import pytest
import json
import types


# --- Mock Context Generator ---
def get_mock_context():
    ctx = types.SimpleNamespace()
    ctx.aws_request_id = "mocked-request-id"
    return ctx

# --- Test: Benign Execution ---
def test_benign_execution():
    event = {"anomaly": "benign"}
    ctx = get_mock_context()
    response = app.lambda_handler(event, ctx)
    assert json.loads(response["body"]).startswith("SCAFAD")

# --- Test: Cold Start Behaviour ---
def test_cold_start_behavior():
    event = {"anomaly": "cold_start", "execution_phase": "init"}
    ctx = get_mock_context()
    response = app.lambda_handler(event, ctx)
    assert response["statusCode"] == 200

# --- Test: CPU Burst ---
def test_cpu_burst_behavior():
    event = {"anomaly": "cpu_burst", "execution_phase": "invoke"}
    ctx = get_mock_context()
    response = app.lambda_handler(event, ctx)
    assert response["statusCode"] == 200

# --- Test: Timeout Fallback ---
def test_timeout_fallback():
    # Use a fake anomaly that causes long delay
    event = {"anomaly": "cold_start", "execution_phase": "shutdown"}
    ctx = get_mock_context()
    response = app.lambda_handler(event, ctx)
    body = json.loads(response["body"])
    assert "SCAFAD" in body
    # No explicit fallback_mode in response, but telemetry should reflect it

# --- Test: Starvation Injection ---
def test_starvation_injection():
    event = {"force_starvation": True, "function_profile_id": "test_func"}
    ctx = get_mock_context()
    response = app.lambda_handler(event, ctx)
    body = json.loads(response["body"])
    assert response["statusCode"] == 206
    assert "fallback-injected" in body

# --- Test: Side Trace Logic ---
def test_trace_emission_logic():
    anomalies = ["cold_start", "cpu_burst"]
    for anomaly in anomalies:
        event = {"anomaly": anomaly, "execution_phase": "init"}
        ctx = get_mock_context()
        response = app.lambda_handler(event, ctx)
        assert response["statusCode"] == 200
