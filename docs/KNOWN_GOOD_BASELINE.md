# Known Good Baseline

Last updated: 2026-04-21

This file records the verified, trustworthy baseline for the SCAFAD-R
dissertation repository.  Claims here have been confirmed against live
repo state — not inferred from session summaries.

## Primary Repository

- Repo: `scafad-r-core`
  (`C:\Users\Gutua\OneDrive\Documents\SCAFAD\project\scafad-r-core`)
- Reference donor repo: `scafad-delta` (Layer 1 source, not a runtime trunk)
- Governing documents: `SCAFAD_MASTER_BLUEPRINT.md` (strategic),
  `SCAFAD_DISSERTATION_BIBLE.md` (operational) — both in `SCAFAD/`

## Verified Test Baseline

Command:
```
python -m pytest scafad -q
```

Verified result (2026-04-21, Johann-direct):
```
485 passed, 1 warning
```

This is the single authoritative baseline.  Do not quote historical
shell estimates in its place.

## Canonical Architecture

| Surface | Role |
|---|---|
| `scafad/` | **Canonical implementation** — Layers 0–6 and runtime |
| `scafad/layer0/` – `scafad/layer6/` | Layer implementations |
| `scafad/runtime/` | Lambda handler, canonical runtime, multilayer pipeline |
| `legacy/root-flat-files/` | Archival — flat-root era, not active runtime |
| `legacy/layers-migration-snapshot/` | Archival — migration intermediate, not active |
| `layers/` | NTFS working-tree residue, gitignored, not in HEAD after DL-038 |
| `core/` | Legacy compatibility shims — do not add new logic here |

## Lambda Deployment Target

```
Handler: scafad.runtime.lambda_handler.lambda_handler
File:    scafad/runtime/lambda_handler.py
Runtime: SCAFADCanonicalRuntime (scafad/runtime/runtime.py)
```

## Canonical Test Set (T-013..T-028)

All 485 tests live under `scafad/`:

| Suite | File | Tests |
|---|---|---|
| T-013 | `scafad/layer1/tests/test_layer1_validation.py` | 36 |
| T-014 | `scafad/layer1/tests/test_layer1_preservation.py` | 39 |
| T-015 | `scafad/layer1/tests/test_layer1_privacy.py` | 25 |
| T-016 | `scafad/layer1/tests/test_layer1_sanitisation.py` | 36 |
| T-017 | `scafad/layer1/tests/test_layer1_hashing.py` | 28 |
| T-018 | `scafad/layer2/tests/test_layer2_detection.py` | 37 |
| T-019 | `scafad/layer0/tests/test_layer0_adapter.py` | 34 |
| T-020 | `scafad/layer1/tests/test_layer1_extended.py` | 17 |
| T-021 | `scafad/layer3/tests/test_layer3_fusion.py` | 37 |
| T-022 | `scafad/layer4/tests/test_layer4_explainability.py` | 36 |
| T-023 | `scafad/layer5/tests/test_layer5_threat.py` | 28 |
| T-024 | `scafad/layer6/tests/test_layer6_feedback.py` | 29 |
| T-025 | `scafad/runtime/tests/test_runtime_e2e.py` | 37 |
| T-026 | `scafad/layer0/tests/test_layer0_core.py` | 50 |
| T-027 | `scafad/layer0/tests/test_layer0_detectors.py` | 16 |
| T-028 | `scafad/tests/formal/test_completeness_score.py` | 17 |

Separate canonical runtime-entry tests (under `tests/`):
- `tests/test_009_canonical_runtime.py`
- `tests/test_010_entrypoint_convergence.py`
- `tests/test_lambda_handler_routes_canonical.py`

## Key Commits (Phase 3–4)

| Commit | Label | Summary |
|---|---|---|
| `43e5294` | DL-039 | Single Lambda entrypoint — I-1 satisfied |
| `ec350fa` | DL-040 | Namespace-alias hook; T-012 ghost imports fixed |
| `9a9c025` | DL-041 | `scafad/conftest.py` + pythonpath — 15 bare-import errors fixed |
| `8af8016` | DL-042 | Remove spurious async from SemanticPreservationOptimizer |
| `9020dd5` | DL-043 | Archive `layers/` snapshot; `.gitignore` updated |
| `4a8b06c` | DL-044 | Remove all tracked `__pycache__` entries from tree |

Note: `4de8542` is also labelled DL-040 (archival commit) — numbering drift,
recorded as a known issue.

## Outstanding Before Phase 4

- Windows-side: delete `layers\`, untracked scratch files (`orchestrator.py`,
  `repo_tree.txt`, `tests\conftest_probe.py`, etc.)
- Step 3 docs reconciliation (this file + `LAYER_STATUS.md` + `README.md`)
- Step 4 root clutter triage
- WP-4.4: KMS key manager
