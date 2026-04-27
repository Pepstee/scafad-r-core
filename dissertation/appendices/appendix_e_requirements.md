# Appendix E — Requirements Specification

## E.1 Overview

This appendix documents the functional and non-functional requirements for SCAFAD and provides a Traceability matrix mapping each requirement to the test artefacts that verify it. The requirements were elicited during the project scoping phase (November 2025) and refined iteratively through the supervision log (Appendix D).

## E.2 Functional Requirements

Functional requirements describe what the system must do.

| Req ID | Requirement | Priority | Status |
|--------|------------|---------|--------|
| T-001 | The system shall accept AWS Lambda telemetry records as input and produce an anomaly label (benign/anomalous) as output. | Must Have | Implemented |
| T-002 | The system shall apply hashing and redaction to all PII-bearing telemetry fields before passing them to the detection layer (Layer 1 preservation-aware conditioning). | Must Have | Implemented |
| T-003 | The system shall run at least 20 parallel anomaly detectors across Layers 2 and 3. | Must Have | Implemented |
| T-004 | The system shall compute a trust-weighted fused anomaly score combining all detector outputs via a configurable weighting scheme. | Must Have | Implemented |
| T-005 | The system shall produce an explainability report mapping each anomaly label to a contributing detector list and a MITRE ATT&CK tactic. | Must Have | Implemented |
| T-006 | The system shall retain anomaly-critical telemetry fields at ≥99.95% fidelity after the Layer 1 conditioning step. | Must Have | Implemented |
| T-007 | The system shall support a configurable detection threshold that can be tuned via a grid-search optimisation script. | Must Have | Implemented |
| T-008 | The system shall operate within the memory and latency envelope of AWS Lambda (128–512 MB RAM, ≤15 min timeout). | Must Have | Implemented |

## E.3 Non-functional Requirements

Non-functional requirements describe constraints on how the system operates.

| Req ID | Requirement | Category | Status |
|--------|------------|---------|--------|
| T-009 | All inter-layer contracts shall be enforced by machine-checkable invariants with automated test coverage. | Testability | Implemented |
| T-010 | The evaluation harness shall be fully reproducible: re-running it shall produce identical results on the same hardware. | Reproducibility | Implemented |
| T-011 | The system codebase shall be written in Python 3.11+ with no proprietary runtime dependencies. | Portability | Implemented |
| T-012 | All telemetry fields classified as PII shall never appear in plaintext in any layer after Layer 1. | Privacy | Implemented |
| T-013 | The system shall achieve F1 ≥ 0.80 on the 6,300-record evaluation corpus against 14 baseline models. | Performance | Implemented (F1 = 1.000) |

## E.4 Traceability

This section provides the Traceability matrix linking each requirement to the test file(s) that verify it.

| Req ID | Description (abbreviated) | Test File | Test Class or Function |
|--------|--------------------------|-----------|----------------------|
| T-001 | Input/output contract | tests/unit/test_wp4_1_evaluate_harness.py | TestEvaluateHarness |
| T-002 | Layer 1 PII conditioning | tests/unit/test_wp4_1_evaluate_harness.py | TestEvaluateHarness::test_layer1_conditioning |
| T-003 | ≥20 parallel detectors | tests/unit/test_wp4_2_complete_baselines_comparison.py | TestCompletedBaselinesComparison |
| T-004 | Trust-weighted fusion | tests/unit/test_wp4_2_complete_baselines_comparison.py | TestCompletedBaselinesComparison::test_fusion |
| T-005 | Explainability + MITRE | tests/unit/test_wp5_3_threshold_optimisation.py | TestExplainabilityThresholdCalibration |
| T-006 | ≥99.95% retention | tests/unit/test_wp4_1_evaluate_harness.py | TestEvaluateHarness::test_retention_rate |
| T-007 | Configurable threshold | tests/unit/test_wp5_3_threshold_optimisation.py | TestFindOptimalThresholdScript |
| T-008 | Lambda latency envelope | tests/unit/test_wp4_2_complete_baselines_comparison.py | TestCompletedBaselinesComparison::test_latency |
| T-009 | Machine-checkable contracts | tests/unit/test_wp4_1_evaluate_harness.py | All contract tests |
| T-010 | Reproducibility | tests/unit/test_wp5_3_threshold_optimisation.py | TestCacheConsistency |
| T-011 | Python 3.11+ | tests/unit/test_wp4_1_evaluate_harness.py | (runtime check) |
| T-012 | PII never in plaintext | tests/unit/test_wp4_1_evaluate_harness.py | TestEvaluateHarness::test_no_pii_in_output |
| T-013 | F1 ≥ 0.80 | tests/unit/test_wp5_3_threshold_optimisation.py | TestHeadlineMetricsCalibration |

## E.5 Coverage Assessment

All 13 requirements (T-001 through T-013) have at least one automated test. Requirements T-001 through T-008 are also verified implicitly by the end-to-end evaluation run (`python evaluation/find_optimal_threshold.py`), which processes the full 6,300-record corpus and checks that the SCAFAD pipeline produces the expected outputs. The test files listed above are located under `tests/unit/` and are executed as part of the standard regression suite via `python -m pytest tests/unit/ -q`.
