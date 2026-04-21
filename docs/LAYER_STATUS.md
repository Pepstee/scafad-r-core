# Layer Status

Last updated: 2026-04-21

This document records the current implementation status of each SCAFAD-R
layer.  All canonical paths are under `scafad/`.  References to `layers/`
or `core/` reflect archival or legacy material only.

## Canonical Architecture Summary

The authoritative implementation surface is the `scafad/` package:

```
scafad/
  layer0/   — Adaptive Telemetry Controller
  layer1/   — Behavioural Intake Zone
  layer2/   — Multi-Vector Detection Matrix
  layer3/   — Trust-Weighted Fusion
  layer4/   — Explainability and Decision Trace
  layer5/   — Threat Alignment
  layer6/   — Feedback and Learning
  runtime/  — Lambda handler, canonical runtime, multilayer pipeline
```

`layers/` is archived migration residue (gitignored, not in HEAD).
`core/` contains legacy compatibility shims — do not add new logic there.

## Layer 0 — Adaptive Telemetry Controller

- Purpose: telemetry capture, adaptive control, anomaly detection, fallback,
  runtime coordination; produces `TelemetryRecord` v4.2
- Status: complete — all 27 modules import cleanly; 100 tests green
- Canonical location: `scafad/layer0/`
- Key module: `scafad/layer0/app_telemetry.py` (`TelemetryRecord`)
- Adapter: `scafad/layer0/adapter.py` (`RCoreToLayer1Adapter`) — the only
  authorised L0→L1 translation point (I-2)
- Test suites: T-019 (34), T-026 (50), T-027 (16)

## Layer 1 — Behavioural Intake Zone

- Purpose: validation, sanitisation, privacy filtering, deferred hashing,
  preservation assessment, quality audit; produces `Layer1ProcessedRecord`
- Status: complete — canonical pipeline implemented and validated
- Canonical location: `scafad/layer1/`
- Key module: `scafad/layer1/pipeline.py` (`Layer1CanonicalPipeline`)
- Test suites: T-013 (36), T-014 (39), T-015 (25), T-016 (36), T-017 (28),
  T-020 (17)
- Invariant satisfied: I-5 (L1 never silently drops fields)

## Layer 2 — Multi-Vector Detection Matrix

- Purpose: 26 parallel detectors over the canonical L1 record
- Status: complete — all 26 detectors implemented and tested
- Canonical location: `scafad/layer2/`
- Key module: `scafad/layer2/detection_matrix.py` (`MultiVectorDetectionMatrix`)
- Test suite: T-018 (37)

## Layer 3 — Trust-Weighted Fusion

- Purpose: combine detection signals with learned trust weights; produces
  fused anomaly score (C-2)
- Status: complete
- Canonical location: `scafad/layer3/`
- Key module: `scafad/layer3/trust_fusion.py` (`TrustWeightedFusionEngine`)
- Test suite: T-021 (37)

## Layer 4 — Explainability and Decision Trace

- Purpose: tiered rationale generation with redaction budget; terse /
  standard / verbose modes (C-4)
- Status: complete
- Canonical location: `scafad/layer4/`
- Key module: `scafad/layer4/explainability.py` (`ExplainabilityDecisionEngine`)
- Test suite: T-022 (36)

## Layer 5 — Threat Alignment

- Purpose: MITRE ATT&CK mapping and campaign clustering
- Status: complete
- Canonical location: `scafad/layer5/`
- Key module: `scafad/layer5/threat_alignment.py` (`ThreatAlignmentEngine`)
- Test suite: T-023 (28)

## Layer 6 — Feedback and Learning

- Purpose: analyst label ingestion, trust modulation, contrastive replay
- Status: complete
- Canonical location: `scafad/layer6/`
- Key module: `scafad/layer6/feedback_learning.py` (`FeedbackLearningEngine`)
- Test suite: T-024 (29)

## Runtime

- Purpose: Lambda handler, canonical runtime orchestration (L0→L6),
  multilayer pipeline coordination
- Status: complete — I-1 satisfied (single entrypoint)
- Canonical location: `scafad/runtime/`
- Lambda handler: `scafad/runtime/lambda_handler.py`
  (`scafad.runtime.lambda_handler.lambda_handler`)
- Canonical runtime: `scafad/runtime/runtime.py` (`SCAFADCanonicalRuntime`)
- Test suite: T-025 (37 E2E tests)

## Evaluation Matrix

| Claim | Evidence | Status |
|---|---|---|
| L0 produces valid TelemetryRecord | T-026, T-027 | ✓ verified |
| L0→L1 adapter is the only translation point | T-019, I-2 | ✓ verified |
| L1 validates and sanitises without silent drops | T-013..T-017, T-020, I-5 | ✓ verified |
| L2 runs 26 detectors | T-018 | ✓ verified |
| L3 trust-weighted fusion produces fused score | T-021 | ✓ verified |
| L4 tiered explainability with budget control | T-022 | ✓ verified |
| L5 MITRE ATT&CK alignment | T-023 | ✓ verified |
| L6 feedback modulates trust | T-024 | ✓ verified |
| Full L0→L6 E2E pipeline works | T-025 | ✓ verified |
| Single Lambda entrypoint | I-1, DL-039 | ✓ verified |
| `python -m pytest scafad -q` green | 485 passed, 1 warning | ✓ Johann-verified |

## Non-Canonical Surfaces (do not import from these)

| Path | Status | Notes |
|---|---|---|
| `layers/` | Gitignored NTFS residue | Archived to `legacy/layers-migration-snapshot/` |
| `core/` | Legacy shims | Modified in working tree; do not add logic |
| `legacy/root-flat-files/` | Archival | Flat-root era |
