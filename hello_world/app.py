import time
import random
import json
import string

def lambda_handler(event, context):
    # --- Log schema version control ---
    LOG_VERSION = {
        "version": "v4.0",
        "stage": "dev",  # or: "release", "beta", "hotfix"
        "notes": "Execution-aware, fallback-aware, dual-channel telemetry (Layer 0)"
    }

    # --- Incoming Parameters ---
    anomaly = event.get("anomaly", "benign")
    function_profile_id = event.get("function_profile_id", "func_default")
    execution_phase = event.get("execution_phase", "invoke")
    concurrency_id = event.get("concurrency_id", ''.join(random.choices(string.ascii_uppercase, k=3)))

        # --- Simulate upstream starvation ---
    simulate_starvation = event.get("force_starvation", False)
    if simulate_starvation:
        log_entry = {
            "event_id": context.aws_request_id,
            "telemetry_status": "fallback_injected",
            "reason": "telemetry_starvation",
            "function_profile_id": function_profile_id,
            "execution_phase": execution_phase,
            "concurrency_id": concurrency_id,
            "source": "scafad-lambda",
            "timestamp": time.time(),
            "log_version": LOG_VERSION
        }
        print(json.dumps(log_entry))
        return {
            "statusCode": 206,  # Partial content
            "body": json.dumps("SCAFAD fallback-injected due to starvation.")
        }


    print(">>> EXECUTION REACHED <<<")
    start = time.time()

    # --- Configurable Timeout Threshold (in seconds) ---
    TIMEOUT_THRESHOLD = 0.6
    fallback_mode = False

    # --- Anomaly Simulation ---
    try:
        if anomaly == "cold_start":
            time.sleep(0.5)
            memory_spike = bytearray(20 * 1024 * 1024)
        elif anomaly == "cpu_burst":
            [x ** 0.5 for x in range(1000000)]
            memory_spike = bytearray(10 * 1024 * 1024)
        else:
            memory_spike = bytearray(8 * 1024 * 1024)
    except Exception:
        fallback_mode = True
        anomaly = "execution_failure"
        memory_spike = bytearray(1 * 1024 * 1024)

    duration = time.time() - start

    # --- Timeout Fallback Logic ---
    if duration > TIMEOUT_THRESHOLD:
        fallback_mode = True
        anomaly = "timeout_fallback"

    # --- Log Entry ---
    log_entry = {
        "event_id": context.aws_request_id,
        "duration": round(duration, 3),
        "memory_spike_kb": len(memory_spike) // 1024,
        "anomaly_type": anomaly,
        "function_profile_id": function_profile_id,
        "execution_phase": execution_phase,
        "concurrency_id": concurrency_id,
        "fallback_mode": fallback_mode,
        "source": "scafad-lambda",
        "timestamp": time.time(),
        "log_version": LOG_VERSION
    }


    # --- Execution-aware sampling trigger ---
    should_emit_side_trace = (
        execution_phase == "init"
        or anomaly in ["cold_start", "cpu_burst"]
        or duration > TIMEOUT_THRESHOLD
    )

    if should_emit_side_trace:
        side_trace = f"[SCAFAD_TRACE] phase='{execution_phase}' profile='{function_profile_id}' ts={round(time.time(), 3)}"
        print(side_trace)


    print(json.dumps(log_entry))

    return {
        "statusCode": 200,
        "body": json.dumps("SCAFAD online. Fallback-aware telemetry complete.")
    }
