# Known Good Baseline

This file records the current practical checkpoints used to measure progress in the primary dissertation repository.

## Primary Repository

- Primary implementation repo: `scafad-r-core`
- Reference Layer 1 donor repo: `scafad-delta`

## Stable Reference Checkpoints

- `scafad-r-core` clean committed baseline before the current working changes:
  - `3928680` — `fix(tests): Phase 1.2 — 121/121 tests green baseline`
- `scafad-delta` latest Codex compatibility checkpoint:
  - `7ef9433` — `fix(delta): expand core compatibility baseline for tests`

## Canonical Active Test Baseline

The active baseline is defined as:

- Layer 0 to Layer 1 contract and smoke tests
- Layer 1 milestone tests `#001` to `#005`
- E2E and benchmark suites `#006` and `#007`
- the broader repo-wide regression baseline that was previously confirmed green

## Current Repository Reality

The current `scafad-r-core` working tree contains in-progress file churn around:

- `tests/test_005_archived_dataset_pipeline.py`
- `tests/test_006_e2e_integration.py`
- `tests/test_007_performance_benchmarks.py`
- archived payload files
- `app_adversarial.py`

Those files should be treated as active work rather than discarded clutter.

## Canonical Architectural Rule

The only allowed Layer 0 to Layer 1 handoff boundary is:

- `core/r_core_to_layer1_adapter.py`

The canonical Layer 1 processed-record contract and higher-layer path now live in:

- `core/layer1_pipeline.py`
- `core/multilayer_pipeline.py`
