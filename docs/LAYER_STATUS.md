# Layer Status

This document records the current implementation status of each SCAFAD layer in the primary repository.

## Layer 0

- Purpose: telemetry capture, adaptive control, anomaly detection, redundancy, fallback, and runtime coordination
- Status: implemented and strongly validated
- Entry point: the Layer 0 family and application modules in the repository root
- Evidence: the established Layer 0 and Layer 0 to Layer 1 test suites

## Layer 1

- Purpose: validation, sanitisation, privacy filtering, hashing, preservation, quality checks, and audit generation
- Status: implemented in minimal canonical form inside `r-core`
- Canonical contract:
  - input: adapter output from `core/r_core_to_layer1_adapter.py`
  - output: `Layer1ProcessedRecord` from `core/layer1_pipeline.py`
- Evidence:
  - Tests `#001` to `#005`
  - archived payload pipeline testing
  - the canonical L1 to L6 integration test added in `#008`

## Layer 2

- Purpose: multi-vector detection over the canonical Layer 1 record
- Status: minimal working implementation
- Contract: `Layer2DetectionResult`
- Evidence: scaffold tests and the integrated L1 to L6 flow

## Layer 3

- Purpose: trust-weighted fusion and volatility adjustment
- Status: minimal working implementation
- Contract: `Layer3FusionResult`
- Evidence: scaffold tests and the integrated L1 to L6 flow

## Layer 4

- Purpose: explanation summary and analyst-facing decision trace
- Status: minimal working implementation
- Contract: `Layer4DecisionTrace`
- Evidence: scaffold tests and the integrated L1 to L6 flow

## Layer 5

- Purpose: MITRE-style threat alignment and campaign grouping
- Status: minimal working implementation
- Contract: `ThreatAlignmentResult`
- Evidence: scaffold tests and the integrated L1 to L6 flow

## Layer 6

- Purpose: analyst feedback ingestion and trust modulation
- Status: minimal working implementation
- Contract: `Layer6FeedbackState`
- Evidence: scaffold tests and the integrated L1 to L6 flow

## Evaluation Matrix

| Claim | Current Evidence |
|---|---|
| Layer 0 to Layer 1 handoff works | Tests `#001`, contract tests, archived payload path |
| Layer 1 canonical path works | Tests `#002` to `#005`, canonical pipeline in `core/layer1_pipeline.py` |
| Higher layers are concrete and testable | `tests/test_l2_l6_scaffold.py` |
| Full minimal L1 to L6 path works | `tests/test_008_l1_to_l6_pipeline.py` |

## Future Work

- replace remaining lightweight heuristics with stronger research implementations
- consolidate any duplicate Layer 1 logic still living only in tests
- reduce legacy clutter in the repository root and archive area
- align the root README more fully with the final dissertation narrative
