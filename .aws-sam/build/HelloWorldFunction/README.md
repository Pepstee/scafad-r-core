# SCAFAD-L0: Adaptive Telemetry Controller for Serverless Environments

**Version:** v4.0-dev  
**Module:** Layer 0 — SCAFAD (Serverless Context-Aware Fusion Anomaly Detection)  
**Status:** Active Development / Research-grade Prototype  
**Institution:** Birmingham Newman University  
**Author:** [Your Name], BSc Computer Science (2025)

---

## Overview

This module constitutes **Layer 0** of the [SCAFAD](https://github.com/your-org/scafad) architecture: an adaptive, resilient telemetry controller tailored for anomaly detection in AWS Lambda environments.

Designed with both experimental trace fidelity and production feasibility in mind, the system simulates invocation-level behavioural anomalies, emits structured and redundant telemetry, and implements fail-safe mechanisms for data starvation and cold-start variance.

---

## Architectural Purpose

> **Layer 0 — Adaptive Telemetry Controller**  
> *"Resolves upstream data starvation and guards against telemetry gaps via execution-aware sampling, redundant trace channels, and null-model fallback injection."*

This layer serves as a **pre-intake behavioural profiler**, enriching raw telemetry with structural identifiers and selectively escalating its verbosity under uncertainty.

---

## Key Capabilities

- ✅ **Anomaly Simulation:**  
  Supports `cold_start`, `cpu_burst`, and `benign` execution modes

- ✅ **Dual Telemetry Channels:**  
  - JSON-structured log entries  
  - Human-readable side-channel traces for critical phases

- ✅ **Execution-Aware Sampling:**  
  Logs increase in verbosity during `init` phase, cold starts, or timeouts

- ✅ **Starvation Resilience:**  
  Fallback telemetry injected under simulated upstream data loss

- ✅ **Telemetry Tagging:**  
  Includes `function_profile_id`, `execution_phase`, `concurrency_id`

- ✅ **Schema Versioning:**  
  All logs include a semantic `log_version` dictionary (e.g. `v4.0-dev`)

---

## Output Format (Structured Log)

```json
{
  "event_id": "uuid...",
  "duration": 0.531,
  "memory_spike_kb": 20480,
  "anomaly_type": "cold_start",
  "function_profile_id": "func_B",
  "execution_phase": "init",
  "concurrency_id": "DUO",
  "fallback_mode": false,
  "source": "scafad-lambda",
  "timestamp": 1754353364.729,
  "log_version": {
    "version": "v4.0",
    "stage": "dev",
    "notes": "Execution-aware, fallback-aware, dual-channel telemetry (Layer 0)"
  }
}
