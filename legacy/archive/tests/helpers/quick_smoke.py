# quick_fail.py
from app_silent_failure import assess_silent_failure
from types import SimpleNamespace

cfg = SimpleNamespace(deployment_stage="DEV")
record = {
    "function_id": "ml_inference",
    "input": {"correlation_id": "abc", "x": 1.0},
    # Break two invariants + corruption:
    "output": {
        "correlation_id": "ZZZ",            # mismatched echo
        "prediction": "cat",
        "confidence": 1.7,                  # out of [0,1]
        "input_hash": "deadbeef",           # won’t match
        "output_hash": "deadbeef"           # won’t match
    },
    "telemetry": SimpleNamespace(duration=95.0),  # above max_duration_s default
    "trace": {"phases": ["INVOKE","INIT"], "timestamps": [2,1]},  # misordered + non-monotonic
}
print(assess_silent_failure(record, cfg, return_report=True))
