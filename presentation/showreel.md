# SCAFAD Showreel: Script, Shot List, and Capture Checklist

*WP-5.12 — Supplementary demonstration evidence for dissertation Chapters 7 and 9.*

---

## Purpose

This document specifies every repo-editable step needed to produce a compelling
demonstration recording of the SCAFAD pipeline. Section 2 (Demo Script) provides
a narrated walk-through aligned with the dissertation narrative; Section 3 (Shot
List) lists exactly what should be visible on screen at each moment; Section 4
(Capture Checklist) gives the operator a practical pre-recording and post-recording
procedure.

The finished recording constitutes supplementary empirical evidence for the
implementation and evaluation claims in Chapters 7 and 9. The primary
reproducibility evidence (evaluation artefacts in `evaluation/results/`) remains
unchanged; the showreel demonstrates the same results interactively to examiners.

All metrics cited below are canonical values verified by the reproducible
evaluation harness (`make evaluate`):

| Metric | SCAFAD | Best baseline (OneClassSVM ν=0.10) | Gap |
|--------|--------|-----------------------------------|-----|
| F1 | **1.0000** | 0.8858 | +0.1142 |
| Precision | **1.0000** | 0.9984 | +0.0016 |
| Recall | **1.0000** | 0.7960 | +0.2040 |
| ROC-AUC | **1.0000** | 0.8954 | +0.1046 |

Source: `evaluation/results/headline_metrics.json` (seed 42, 6,300 records).

---

## Demo Script

### Opening — 0:00–0:25

*Narrate:*

"SCAFAD is a seven-layer, privacy-preserving anomaly-detection pipeline for AWS
Lambda serverless telemetry. This recording walks through the pipeline end-to-end:
from telemetry ingestion at Layer 0 through trust-weighted fusion at Layer 3,
explainability at Layer 4, MITRE ATT&CK alignment at Layer 5, and the reproducible
evaluation that achieved F1 = 1.000 and ROC-AUC = 1.000 against fourteen classical
baselines on a corpus of 6,300 synthetic Lambda records."

---

### Shot 1 — Repository structure — 0:25–0:55

*Narrate:*

"The project root contains the full seven-layer implementation. The `scafad/`
package hosts the detection layers; `lambda_handler.py` is the sole Lambda
entry-point; and `evaluation/` holds the reproducible evaluation harness and
all cached results."

*Show:* run `ls` (or `dir`) at the repository root; then `ls scafad/` to reveal
the `layer0_core.py`, `layers/layer1/` … `layers/layer6/` structure. Scroll
slowly so every layer directory is visible.

---

### Shot 2 — Lambda handler and Canonical Runtime — 0:55–1:35

*Narrate:*

"Every production invocation enters through `lambda_handler.lambda_handler` and
delegates immediately to `SCAFADCanonicalRuntime.process_event`. Architectural
Invariant I-1 prohibits any alternative entry-point from bypassing the canonical
runtime, ensuring that no telemetry record can skip the seven-layer processing
chain."

*Show:* open `lambda_handler.py` in an editor; scroll to the single
`SCAFADCanonicalRuntime.process_event(event)` call; zoom in so the line is
clearly readable. The `CanonicalRuntimeResult.to_dict()` return line should also
be visible.

---

### Shot 3 — Layer 0: detection engine and HMAC signing — 1:35–2:20

*Narrate:*

"Layer 0 is the observation layer. The `AnomalyDetectionEngine` runs twenty-six
detection algorithms concurrently across four families: statistical, resource-based,
execution-pattern, and advanced/economic. Each invocation record is assembled into
a twenty-eight-field `TelemetryRecord` schema v4.2 and signed with HMAC-SHA256,
satisfying Invariant I-6 (cryptographic integrity of every emitted record)."

*Show:* open `layer0_core.py`; scroll to the `AnomalyDetectionEngine` class
definition, highlighting the list of twenty-six detector names. Then scroll to the
HMAC-SHA256 signing block so both elements are shown.

---

### Shot 4 — Layer 1: privacy-preserving conditioning — 2:20–3:05

*Narrate:*

"Layer 1 applies GDPR Article 5(1)(c)-compliant data minimisation. Personally
identifiable fields are pseudonymised via HMAC-SHA256 deferred hashing. Critically,
anomaly-critical features — invocation duration, memory utilisation, CPU
utilisation, network I/O, and cold-start frequency — are explicitly excluded from
any transformation. The T-003 test suite verified ≥ 99.95 % field retention on
an adversarial edge-case corpus, confirming that privacy conditioning does not
degrade detection fidelity."

*Show:* open `layers/layer1/` — navigate to the hashing pipeline module; highlight
the preservation-aware conditioning logic that enumerates the excluded
anomaly-critical fields. The `quality_report` and `audit_record` fields in the
output dataclass should be visible.

---

### Shot 5 — Layer 3: trust-weighted fusion and score gap — 3:05–3:45

*Narrate:*

"Layer 3 fuses the twenty-six detector signals using per-source trust weights
derived from the Layer-1 quality report. The resulting score distribution shows
a clear separation gap of 0.097: benign records score at most 0.081 whilst
anomalous records score at least 0.178. The WP-5.3 calibrated threshold of 0.09
sits in the middle of this gap, yielding zero false positives and zero false
negatives on the evaluation corpus."

*Show:* open `evaluation/results/optimal_threshold.json`; scroll to highlight
the `score_distribution` block — `benign.max: 0.0807`, `anomaly.min: 0.1781`,
`gap: 0.0974` — and then the `optimal_threshold` block: `value: 0.09`,
`f1: 1.0`, `precision: 1.0`, `recall: 1.0`.

---

### Shot 6 — `make evaluate`: reproducing headline metrics — 3:45–4:50

*Narrate:*

"The full evaluation is reproducible from a single command. Running `make evaluate`
executes the pipeline against all 6,300 records — 6,250 anomalous and 50 benign —
and writes the headline metrics file."

*Show:* in a terminal at the project root, type and run:

```bash
make evaluate
```

Allow the command to run to completion. Then open the results file:

```bash
cat evaluation/results/headline_metrics.json
```

Scroll to highlight `"f1": 1.0`, `"roc_auc": 1.0`, `"confusion_matrix"` with
`"tp": 6250, "fp": 0, "fn": 0, "tn": 50`, and `"all_pass": true`.

*Narrate while scrolling:*

"F1 = 1.000. ROC-AUC = 1.000. TP = 6,250, FP = 0, FN = 0, TN = 50.
`dissertation_claims.all_pass` is true — every claimed metric matches the
reproduced result within the 0.02 tolerance."

---

### Shot 7 — Baselines comparison table — 4:50–5:20

*Narrate:*

"SCAFAD outperforms all fourteen baselines. The best single-model baseline —
OneClassSVM (ν = 0.10) — achieves F1 = 0.8858. SCAFAD's seven-layer
trust-weighted pipeline reaches F1 = 1.000, a gap of +0.1142."

*Show:* in the terminal:

```bash
cat evaluation/results/baselines_summary.md
```

Scroll to the results table; zoom in so the SCAFAD final row (bold,
F1 = **1.0000**, ROC-AUC = **1.0000**) is clearly visible alongside the top
baseline.

---

### Closing — 5:20–5:35

*Narrate:*

"All source code, evaluation artefacts, and test results are available in this
repository. Full reproduction is accomplished by executing `make evaluate` from
the project root. Thank you."

---

## Shot List

The table below is the authoritative ordering for the recording operator. The
time column is the target timestamp for the *start* of each shot.

| # | Target time | Screen content | File or command | Key element to show |
|---|-------------|----------------|-----------------|---------------------|
| 1 | 0:00 | Opening narration | — | Spoken only (or title card) |
| 2 | 0:25 | Terminal | `ls` repo root; `ls scafad/` | Layer-directory names L0–L6 |
| 3 | 0:55 | Editor | `lambda_handler.py` | `SCAFADCanonicalRuntime.process_event` call |
| 4 | 1:35 | Editor | `layer0_core.py` | `AnomalyDetectionEngine`, 26 detector names, HMAC-SHA256 signing |
| 5 | 2:20 | Editor | `layers/layer1/` hashing module | Anomaly-critical field exclusion list; `audit_record` |
| 6 | 3:05 | Editor | `evaluation/results/optimal_threshold.json` | `gap: 0.0974`, `benign.max: 0.0807`, `anomaly.min: 0.1781`, `value: 0.09` |
| 7 | 3:45 | Terminal | `make evaluate` run | Command output completing without errors |
| 8 | 4:15 | Terminal | `cat evaluation/results/headline_metrics.json` | `f1: 1.0`, `roc_auc: 1.0`, `all_pass: true` |
| 9 | 4:50 | Terminal | `cat evaluation/results/baselines_summary.md` | SCAFAD row F1=**1.0000**; best baseline F1=0.8858 |
| 10 | 5:20 | Terminal | — | Closing narration |

---

## Capture Checklist

### Environment setup (do this before starting the recording software)

- [ ] Working directory is the repository root:
      `C:/Projects/SCAFAD/project/scafad-r-core/` (Windows) or equivalent on the
      deployment machine.
- [ ] Python environment is activated and all dependencies installed
      (`pip install -r requirements.txt` or equivalent).
- [ ] `evaluation/results/scafad_results.json` is present (required by
      `python evaluation/run_all.py --use-cache`). If absent, run
      `python evaluation/run_scafad_pipeline.py` first.
- [ ] Dry-run `make evaluate` once before recording to confirm exit code 0 and
      that `evaluation/results/headline_metrics.json` is written with `"all_pass": true`.
- [ ] Terminal font size ≥ 16pt; editor font size ≥ 14pt.
- [ ] Syntax highlighting enabled in the editor.
- [ ] Screen resolution 1920 × 1080 minimum (2560 × 1440 preferred for HiDPI
      exports).
- [ ] Browser tabs, desktop notifications, and unrelated windows closed.
- [ ] Microphone tested: no background noise; input level −12 to −6 dBFS when
      speaking normally.
- [ ] Screen-recording software launched and configured (OBS Studio, Camtasia, or
      QuickTime Screen Recording are all suitable).

### Recording procedure

1. Start the screen-recording software.
2. Open a fresh terminal at the repository root.
3. Follow the Script (Section 2) and Shot List (Section 3) in order — do not
   skip shots or reorder them.
4. **Shot 6 (`make evaluate`) is mandatory** — it is the primary live
   reproducibility evidence and must not be pre-recorded or substituted with a
   static screenshot.
5. After `make evaluate`, open `headline_metrics.json` and scroll slowly so
   `"all_pass": true` is clearly visible for at least three seconds.
6. After the baselines table (Shot 9), deliver the closing narration.
7. Stop the recording.

### Post-recording

- [ ] Trim any silence or mistakes from the head and tail of the recording.
- [ ] Export as **MP4 (H.264)**, target bitrate ≥ 4 Mbps, stereo audio 44.1 kHz.
- [ ] Name the file `showreel.mp4` and place it in
      `project/scafad-r-core/presentation/` alongside this document.
- [ ] Verify that the exported video plays without artefacts and that all terminal
      text is readable at 100 % zoom.
- [ ] Commit with message:
      `docs(presentation): add SCAFAD showreel recording (WP-5.12)`
- [ ] If the video file exceeds 50 MB, store it via Git LFS or upload it to the
      agreed external host and update `presentation/showreel_url.txt` with the
      direct link.

---

## Narrative Consistency Reference

The table below maps each shot to the dissertation section that makes the
corresponding claim, so the recording operator can verify alignment before
pressing record.

| Shot | Key claim shown | Dissertation section |
|------|-----------------|----------------------|
| 2 | Seven-layer architecture | Ch. 7 §7.1 |
| 3 | Invariant I-1; canonical runtime | Ch. 7 §7.1 |
| 4 | 26 detectors; HMAC-SHA256 (I-6) | Ch. 7 §7.2 (L0) |
| 5 | Privacy conditioning; ≥ 99.95 % retention | Ch. 7 §7.2 (L1); Ch. 11 RQ3 |
| 6 | Threshold = 0.09; gap = 0.097 | Ch. 9 §9.6; Ch. 11 RQ1 |
| 7–8 | `make evaluate`; F1 = 1.000; ROC-AUC = 1.000 | Ch. 9 §9.4–§9.5; Ch. 11 §11.2 |
| 9 | SCAFAD +0.1142 F1 over best baseline | Ch. 9 §9.5; Ch. 10 §10.2 |
