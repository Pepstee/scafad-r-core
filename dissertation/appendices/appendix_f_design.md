# Appendix F — Detailed Design

## F.1 Overview

This appendix provides detailed design documentation for SCAFAD supplementing the high-level design presented in Chapter 6. It covers the interface contracts between each layer, the internal module structure, and representative sequence diagrams for the two primary runtime paths: the standard evaluation path and the threshold-calibration path.

## F.2 Seven-Layer Architecture Summary

SCAFAD is structured as a strict seven-layer pipeline. Each layer exposes a well-defined interface consumed by the layer above it; no layer may access another layer's internals directly.

| Layer | Name | Primary responsibility |
|-------|------|----------------------|
| Layer 0 | Telemetry Ingestion | Parse raw AWS Lambda telemetry; validate against JSON schema; enforce field-level type contracts |
| Layer 1 | Preservation-Aware Conditioning | Hash PII fields (SHA-256); redact non-critical fields; preserve anomaly-critical fields at ≥99.95% fidelity |
| Layer 2 | Statistical Detectors | Run statistical anomaly detectors (IQR, Z-Score, MAD, DBSCAN, KDE); emit per-record score vectors |
| Layer 3 | ML Detectors + Trust-Weighted Fusion | Run ML detectors (Isolation Forest, OCSVM, LOF, AutoEncoder, et al.); aggregate scores via trust-weighted fusion engine; compute fused anomaly score |
| Layer 4 | Explainability + Decision | Apply decision threshold (≥0.09 → anomalous); generate per-record MITRE ATT&CK attribution; produce explainability report |
| Layer 5 | Audit + Reporting | Generate audit trail entries; persist results to `evaluation/results/`; support structured result queries |
| Layer 6 | Feedback Loop | Ingest operator corrections; update trust weights; schedule re-calibration runs |

## F.3 Layer Interface Contracts

### F.3.1 Layer 0 → Layer 1 Interface

- **Input type:** `TelemetryRecord` (Python dataclass; fields defined in `scafad/layer0/models.py`)
- **Contract:** All required fields (`request_id`, `function_name`, `execution_time_ms`, `memory_used_mb`, `error_flag`) must be non-null.
- **Error behaviour:** Records failing schema validation are rejected with a `ValidationError`; the pipeline does not silently drop them.

### F.3.2 Layer 1 → Layer 2/3 Interface

- **Output type:** `ConditionedRecord` — a copy of `TelemetryRecord` with PII fields replaced by their SHA-256 digests and redacted fields replaced with sentinel values.
- **Contract:** `anomaly_signal` fields (execution time, memory, error flag, cold-start indicator) must pass through unchanged. `retention_rate` must be ≥0.9995 over any batch of 100+ records.

### F.3.3 Layer 3 → Layer 4 Interface

- **Output type:** `FusedScore` — a float in [0, 1] representing the trust-weighted anomaly probability.
- **Contract:** `FusedScore` = ∑(wᵢ × sᵢ) / ∑(wᵢ) where wᵢ is the trust weight for detector i and sᵢ is its normalised score.
- **Threshold:** Records with `fused_score ≥ 0.09` are labelled anomalous; records with `fused_score < 0.09` are labelled benign.

## F.4 Module Structure

```
scafad/
├── layer0/
│   ├── __init__.py
│   ├── models.py          # TelemetryRecord, Layer0Config
│   ├── ingestion.py       # parse_telemetry(), validate_record()
│   └── compression.py     # optional compression for large batches
├── layer1/
│   ├── __init__.py
│   ├── conditioning.py    # ConditioningPipeline.apply()
│   ├── hashing.py         # sha256_field(), deferred_hash_batch()
│   └── redaction.py       # redact_field(), RedactionPolicy
├── layer2/
│   ├── __init__.py
│   └── detectors/         # IQRDetector, ZScoreDetector, MADDetector, ...
├── layer3/
│   ├── __init__.py
│   ├── detectors/         # IsolationForestDetector, OCSVMDetector, LOFDetector, ...
│   └── fusion.py          # TrustWeightedFusionEngine
├── layer4/
│   ├── __init__.py
│   └── explainability.py  # ExplainabilityEngine, apply_threshold()
├── layer5/
│   ├── __init__.py
│   └── audit.py           # AuditTrailGenerator
└── layer6/
    ├── __init__.py
    └── feedback.py        # FeedbackLoop, update_trust_weights()
```

## F.5 Sequence Diagram — Standard Evaluation Path

The following sequence describes the flow for a single telemetry record through the full SCAFAD pipeline:

```
Caller          Layer0           Layer1           Layer2+3          Layer4
  |                |                |                  |               |
  |-- parse() ---> |                |                  |               |
  |                |-- validate() ->|                  |               |
  |                |                |-- hash_pii() --->|               |
  |                |                |-- retain_anomaly_fields() -->    |
  |                |                |                  |               |
  |                |                |<-- ConditionedRecord ----------  |
  |                |                |                  |               |
  |                |                |-- detect() ----> |               |
  |                |                |                  |-- fuse() ---->|
  |                |                |                  |               |
  |                |                |                  |<-- FusedScore |
  |                |                |                  |               |
  |                |                |                  |-- threshold --|
  |                |                |                  |   >= 0.09?    |
  |                |                |                  |               |
  |<--------------------------------------------- AnomalyLabel -------|
```

## F.6 Sequence Diagram — Threshold Calibration Path

The threshold calibration script (`evaluation/find_optimal_threshold.py`) runs a grid search:

```
GridSearch       EvaluationCache       ExplainabilityEngine
    |                  |                       |
    |-- load_cache()-->|                       |
    |<-- 6300 records--|                       |
    |                  |                       |
    |  for t in 0.05..0.95 step 0.01:          |
    |-- apply_threshold(t) ----------------->  |
    |<-- labels --------------------------------|
    |-- compute_f1(labels, ground_truth) -->   |
    |<-- F1 score -------------------------    |
    |                                          |
    |-- select_optimal_t (max F1) -----------> output: 0.09
    |-- save optimal_threshold.json -------->  done
```

## F.7 Design Decisions

**DD-1 — Strict layer isolation.** No layer imports from a higher-numbered layer. This is enforced by static analysis (`tests/unit/` import-safety tests) and by the architecture contract.

**DD-2 — Immutable records.** `TelemetryRecord` and `ConditionedRecord` are Python dataclasses with `frozen=True`. This eliminates accidental mutation between layers.

**DD-3 — Trust weights are runtime-configurable.** Initial trust weights are set to equal values; the feedback loop (Layer 6) updates them after operator corrections. This allows the system to adapt to deployment-specific anomaly distributions without retraining.

**DD-4 — Threshold externalised to JSON.** The decision threshold (0.09) is stored in `evaluation/results/optimal_threshold.json`, not hard-coded. The explainability engine reads it at startup, making recalibration a configuration-only change.
