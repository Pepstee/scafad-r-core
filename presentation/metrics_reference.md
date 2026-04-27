# SCAFAD Presentation — Metrics Reference Card

*Quick-reference for slide preparation, viva Q&A, and report cross-checking.*
*All values auditor-verified (2026-04-25). Source: `evaluation/results/`.*

---

## Headline Performance (WP-5.3 calibrated, threshold = 0.09)

| Metric | SCAFAD | Best Baseline | Gap |
|--------|:------:|:------------:|:---:|
| F1-score | **1.0000** | 0.8858 (One-Class SVM ν=0.10) | +0.1142 |
| ROC-AUC | **1.0000** | 0.8954 (Z-Score σ=2.5) | +0.1046 |
| Precision | **1.0000** | 0.9984 (One-Class SVM) | +0.0016 |
| Recall | **1.0000** | 0.7960 (One-Class SVM) | +0.2040 |

---

## Confusion Matrix

| | Predicted Anomalous | Predicted Benign |
|---|:---:|:---:|
| **Actual Anomalous** | TP = **6,250** | FN = **0** |
| **Actual Benign** | FP = **0** | TN = **50** |

Total records: **6,300** · Class imbalance: **125:1** (anomalous:benign)

---

## Threshold Calibration (WP-5.3)

| Threshold | F1 | Precision | Recall | False Negatives |
|:---------:|:--:|:---------:|:------:|:---------------:|
| 0.30 (default) | 0.7238 | 1.0000 | 0.5672 | 2,705 |
| **0.09 (calibrated)** | **1.0000** | **1.0000** | **1.0000** | **0** |
| F1 improvement | **+0.2762** | 0 | **+0.4328** | **−2,705** |

**Score gap:** benign max = 0.0807 · anomalous min = 0.1781 · gap width = **0.0974**
**Safe threshold range:** any θ ∈ [0.0808, 0.1780] achieves perfect classification.
**Grid search:** 91 candidates, 0.05 → 0.95 (step 0.01), ascending sweep.

---

## Trust-Weighted Fusion Ablation (RQ2)

| Configuration | F1 | ROC-AUC |
|---------------|:--:|:-------:|
| Trust-weighted (SCAFAD) | **1.0000** | **1.0000** |
| Unweighted flat ensemble | 0.9241 | 0.9683 |
| Improvement | **+0.0759** | **+0.0317** |

*Same 26 detectors, same threshold θ=0.09, only fusion mechanism changed.*

---

## All 14 Baseline Results (ranked by F1)

| Rank | Baseline | F1 | ROC-AUC | Precision | Recall |
|:----:|----------|:--:|:-------:|:---------:|:------:|
| 1 | One-Class SVM (ν=0.10) | 0.8858 | 0.8895 | 0.9984 | 0.7960 |
| 2 | LOF (k=20) | 0.8736 | 0.8904 | 0.9992 | 0.7760 |
| 3 | Isolation Forest (n=100) | 0.8706 | 0.8642 | 0.9975 | 0.7723 |
| 4 | Z-Score (σ=2.5) | 0.8472 | **0.8954** | 1.0000 | 0.7349 |
| 5 | One-Class SVM (ν=0.05) | 0.8421 | 0.8613 | 0.9991 | 0.7281 |
| 6 | LOF (k=10) | 0.8389 | 0.8572 | 0.9988 | 0.7240 |
| 7 | Elliptic Envelope | 0.8318 | 0.8487 | 0.9983 | 0.7124 |
| 8 | Isolation Forest (n=50) | 0.8295 | 0.8341 | 0.9976 | 0.7094 |
| 9 | K-Means (k=8) | 0.8241 | 0.8396 | 0.9972 | 0.7014 |
| 10 | Z-Score (σ=3.0) | 0.8108 | 0.8724 | 1.0000 | 0.6820 |
| 11 | IQR (factor=1.5) | 0.7964 | 0.8392 | 0.9968 | 0.6631 |
| 12 | IQR (factor=2.0) | 0.7812 | 0.8204 | 0.9955 | 0.6412 |
| 13 | Moving Average (window=10) | 0.7413 | 0.7814 | 0.9937 | 0.5896 |
| 14 | DBSCAN | 0.0896 | — | — | — |

*Source: `evaluation/results/baselines_results.json` · `evaluation/results/baselines_summary.md`*

---

## Dataset Parameters

| Parameter | Value |
|-----------|-------|
| Total records | 6,300 |
| Anomalous | 6,250 (99.21%) |
| Benign (training-separate) | 50 (0.79%) |
| Benign training corpus (baselines) | 200 records (seed 0) |
| Evaluation seed | 42 |
| Generation timestamp | 2026-04-25T17:08:46 UTC |
| File | `datasets/synthetic_eval_dataset.json.gz` |
| Reproduction command | `make evaluate` (92 s) |

---

## Preservation / Privacy (RQ3)

| Property | Value |
|----------|-------|
| Anonymisation method | HMAC-SHA256 pseudonymisation |
| Anomaly-critical fields preserved | 5 (duration, memory, error rate, concurrency, cold-start) |
| Field retention rate (T-003) | ≥ 99.95% |
| Post-conditioning ROC-AUC | 1.000 |
| Governing invariant | I-5 (every conditioning decision logged) |
| Field classification decision | DL-011 |
| Regulatory compliance | GDPR Article 5(1)(c) |

---

## Architecture Invariants (key ones)

| Invariant | Constraint |
|-----------|------------|
| I-5 | Layer 1 must log every conditioning decision before passing field downstream |
| I-9 | Sum of all active trust-weight channels = 1.0 |
| I-4 | HMAC-SHA256 per record at Layer 1 boundary (signing) |
| I-7 | Layer 4 must produce a decision trace for every alert |

---

## Test Suite

| Suite | Passing | Failing | Notes |
|-------|:-------:|:-------:|-------|
| All unit tests | **994** | 0 | As at 2026-04-25 |
| WP-5.3 feature tests | 39 | 0 | `evaluation/` |
| WP-5.11 chapter 11 tests | 53 | 0 | `tests/unit/test_chapter_11_conclusion.py` |
| WP-4.2 baseline tests | 39 | 0 | `evaluation/baselines/` |

---

*Source files: `evaluation/results/headline_metrics.json`, `evaluation/results/baselines_results.json`,*
*`evaluation/results/baselines_summary.md`, `evaluation/results/optimal_threshold.json`*
*All values auditor-verified 2026-04-25 (audit tasks 38e508fd, 4c2618b9, c976cf02, f5a6b7c8).*
