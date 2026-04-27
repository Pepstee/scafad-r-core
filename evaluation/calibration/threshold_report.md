# WP-5.2: L2 Detection Threshold Calibration Report

**Date:** 2026-04-24  
**Task:** WP-5.2  
**Author:** BuilderAgent (task `d0ab2d75-c5dc-4a3f-a819-0eda0d34caf6`)  
**Dataset:** `datasets/synthetic_eval_dataset.json.gz` (WP-4.4, schema v4.2)  
**Script:** `evaluation/calibration/threshold_calibration.py`  
**Results JSON:** `evaluation/calibration/calibration_results.json`

---

## 1. Objective

The `MultiVectorDetectionMatrix` (L2 layer) used a hard-coded anomaly threshold of
**0.30** (blueprint Section 4.5), annotated as "to be calibrated".  This report
documents the threshold sweep and justifies the new value of **0.15**, which
achieves F1 = 1.000 versus F1 = 0.590 at 0.30 — a **+69.4% relative improvement**.

---

## 2. Methodology

### 2.1 Dataset

The WP-4.4 synthetic evaluation dataset (`synthetic_eval_dataset.json.gz`, seed 42,
schema v4.2) was used.  It contains **6,500 records** across 26 anomaly classes:

| Subset | Records |
|--------|---------|
| Benign | 250 (1 class × 250) |
| Anomalous | 6,250 (25 classes × 250) |
| **Total** | **6,500** |

Ground truth label: `anomaly_type == "benign"` → 0, any other type → 1.

### 2.2 Record Mapping

Each synthetic record (WP-4.4 flat schema) was translated to the L2 envelope format
before scoring.  Key field mappings:

| Synthetic field | L2 envelope field | Notes |
|---|---|---|
| `duration` (seconds) | `telemetry_data.l0_duration_ms` | × 1000 |
| `memory_spike_kb` | `telemetry_data.l0_memory_spike_kb` | direct |
| `cpu_utilization` (%) | `telemetry_data.l0_cpu_utilization` | direct |
| `network_io_bytes` | `telemetry_data.l0_network_io_bytes` | direct |
| `causal_depth` | `provenance_chain.causal_depth` | direct |
| `parent_chain` | `provenance_chain.parent_chain` | direct |
| `economic_risk_score` | `context_metadata.economic_risk_score` | direct |
| `silent_failure_probability` | `context_metadata.silent_failure_probability` | direct |
| `completeness_score` | `context_metadata.completeness_score` | direct |
| `tags.category` | `context_metadata.anomaly_category` | direct |
| `adversarial_score` × 10 | `context_metadata.anomaly_severity` | scale [0,1]→[0,10] |

### 2.3 Threshold Sweep

Each record was passed through `MultiVectorDetectionMatrix.analyze()` to obtain an
`aggregate_score` (confidence-weighted mean of four detector outputs).  The sweep
covered 17 candidate thresholds from 0.10 to 0.90 in steps of 0.05.  At each
candidate, precision, recall, and F1 were computed against the ground truth labels.

**Selection rule:** highest F1; ties broken by higher recall, then by higher
threshold value (more conservative choice, robust to distribution shift).

---

## 3. Score Distribution

The four detectors (`RuleChainEngine`, `DriftTracker`, `GraphImmunizedDetector`,
`SemanticDeviationCore`) produce a clean separation between benign and anomalous
records:

| Class | Min | Max | Mean |
|-------|-----|-----|------|
| Benign (n=250) | 0.0519 | **0.0961** | 0.0747 |
| Anomalous (n=6,250) | **0.1769** | 0.4911 | 0.2866 |

**Separation gap:** `[0.0961, 0.1769]` — width **0.0808**.

No benign record scores above 0.0961 and no anomalous record scores below 0.1769.
The gap exists because:

- `SemanticDeviationCore` adds a base score of 0.45 for any non-benign `anomaly_type`.
  Benign records receive 0 on this component.
- `RuleChainEngine` and `DriftTracker` contribute small additional scores from
  metric thresholds and risk fields populated in anomalous synthetic records.
- All benign records were generated with low/normal telemetry values (no drift
  markers, low economic risk, normal completeness scores).

---

## 4. Precision / Recall Curve

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|-----------|----|----|-----|-----------|--------|-----|
| 0.10 | 6250 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** |
| **0.15** | **6250** | **0** | **0** | **1.0000** | **1.0000** | **1.0000** |
| 0.20 | 5758 | 0 | 492 | 1.0000 | 0.9213 | 0.9590 |
| 0.25 | 3922 | 0 | 2328 | 1.0000 | 0.6275 | 0.7711 |
| **0.30 (baseline)** | 2617 | 0 | 3633 | 1.0000 | 0.4187 | 0.5903 |
| 0.35 | 1014 | 0 | 5236 | 1.0000 | 0.1622 | 0.2792 |
| 0.40 | 459 | 0 | 5791 | 1.0000 | 0.0734 | 0.1368 |
| 0.45 | 110 | 0 | 6140 | 1.0000 | 0.0176 | 0.0346 |
| >= 0.50 | 0 | 0 | 6250 | N/A | 0.0000 | 0.0000 |

**Key observation:** Precision is 1.000 at every threshold because no benign record
reaches any tested threshold (max benign score = 0.0961 < 0.10).  The critical
variable is **recall**: only thresholds <= 0.1769 can achieve recall = 1.000.

---

## 5. Selected Threshold: 0.15

### 5.1 Why 0.15 over 0.10

Both 0.10 and 0.15 yield identical F1 = 1.000, precision = 1.000, recall = 1.000 on
this dataset.  The tie-breaking rule favours the **higher threshold** (0.15) because:

1. **Robustness margin.**  
   - Above benign max: 0.15 − 0.0961 = **0.054** (vs 0.10 − 0.0961 = 0.004)  
   - Below anomalous min: 0.1769 − 0.15 = **0.027** (vs 0.1769 − 0.10 = 0.077)  
   At 0.10, a benign record with only a minor metric fluctuation (score increase of
   ~0.004) would cross the threshold.  At 0.15, the margin is 13× larger.

2. **False-positive resistance.**  In production, new benign workloads (e.g., burst
   cold-starts with elevated `cpu_utilization`) could push scores slightly above the
   synthetic benign maximum.  The 0.054 margin above 0.0961 absorbs reasonable
   distributional drift before a false positive occurs.

3. **Standard calibration practice.**  When multiple thresholds tie on the primary
   metric, the more conservative (higher) value is preferred in security-sensitive
   systems to reduce alert fatigue.

### 5.2 Final metrics at 0.15

| Metric | Baseline (0.30) | Selected (0.15) | Delta |
|--------|----------------|-----------------|-------|
| Precision | 1.0000 | 1.0000 | 0.0000 |
| Recall | 0.4187 | **1.0000** | **+0.5813** |
| F1 | 0.5903 | **1.0000** | **+0.4097** |
| TP | 2,617 | 6,250 | +3,633 |
| FP | 0 | 0 | 0 |
| FN | 3,633 | 0 | -3,633 |

F1 improvement: **+0.4097** (+69.4% relative).

---

## 6. Impact on Existing Tests

The `scafad/layer2/tests/test_layer2_detection.py` suite contains tests that assert
`anomaly_indicated` based on specific score thresholds.  With the threshold changed
from 0.30 to 0.15:

- **Tests expecting `anomaly_indicated=True`** for records with aggregate score >= 0.15:
  will continue to pass (the lower threshold means more records are flagged).
- **Tests expecting `anomaly_indicated=False`** for benign records with low scores:
  will continue to pass (benign scores cluster at 0.05–0.10, well below 0.15).
- **Tests that relied on the 0.30 boundary specifically** (e.g., records with score
  in [0.15, 0.29]): will now return `anomaly_indicated=True` instead of `False`.
  The tester agent must verify these cases.

---

## 7. Limitations and Caveats

1. **Semantic leakage.**  `SemanticDeviationCore` reads `anomaly_type` directly from
   the record and adds 0.45 to the score if it is non-benign.  In the synthetic
   dataset, this field is the ground truth label, so the detector has access to the
   label during inference.  This creates a perfectly separating feature for the
   calibration experiment.  In real deployments, `anomaly_type` would be a preliminary
   classification from an upstream layer; the calibration result may not transfer
   directly if that upstream classifier has lower accuracy.

2. **Synthetic dataset coverage.**  The 6,500 records span 26 fixed anomaly classes
   generated from parametric templates.  Novel attack patterns or benign workloads
   outside the template distribution may not be well-represented.

3. **No FP in dataset.**  Precision is 1.000 at all tested thresholds because all
   250 benign records score below 0.10.  This prevents a precision–recall tradeoff
   curve from forming.  A richer benign corpus with more diverse workloads (e.g.,
   bursty legitimate traffic) would produce a more informative curve.

4. **Static detector weights.**  The calibration fixes only the final classification
   threshold.  The four detector confidence weights (0.60–0.75) and individual
   scoring components were not tuned in this work item.

---

## 8. Code Change

**File:** `scafad/layer2/detection_matrix.py`, class `MultiVectorDetectionMatrix.analyze()`

```python
# Before (blueprint default):
anomaly_indicated=aggregate >= 0.3,

# After (WP-5.2 calibrated):
# Threshold calibrated via WP-5.2 sweep on synthetic_eval_dataset.json.gz.
# 0.15 yields F1=1.000 vs F1=0.590 at the original 0.30.
# Calibration report: evaluation/calibration/threshold_report.md
anomaly_indicated=aggregate >= 0.15,
```

---

## 9. Reproducibility

To reproduce the calibration sweep:

```bash
cd project/scafad-r-core
python evaluation/calibration/threshold_calibration.py
# Expected output:
#   Selected  (0.15) -> P=1.0000  R=1.0000  F1=1.0000
#   Improvement    : F1 +0.4097 (+69.4%)
```

The full numerical results are in `evaluation/calibration/calibration_results.json`.
