# SCAFAD Presentation Package

**Version:** 1.0  
**Created:** 2026-04-26  
**Purpose:** MSc dissertation viva / board submission presentation  
**Target audience:** Academic examination board, viva supervisors

---

## Package Contents

| File | Purpose |
|------|---------|
| `slides.md` | Main slide deck — 19 slides in Marp-compatible Markdown |
| `speaker_notes.md` | Detailed per-slide speaker notes, Q&A guidance, and pacing |
| `metrics_reference.md` | Quick-reference card: all headline metrics, baselines table, dataset parameters |
| `showreel.md` | Narrated demo script, ordered shot list, and full capture checklist |
| `SHOWREEL_STATUS.md` | Single source of truth for the recording lifecycle (PENDING / RECORDED_LOCAL / RECORDED_EXTERNAL) |
| `showreel.mp4` | *(when produced)* Screen-recording walkthrough of the seven-layer pipeline |
| `README.md` | This file — usage instructions |

---

## Rendering the Slides

### Option 1 — Marp CLI (recommended)

```bash
# Install Marp CLI once
npm install -g @marp-team/marp-cli

# Render to PDF
marp slides.md --pdf --output scafad-presentation.pdf

# Render to HTML (self-contained)
marp slides.md --html --output scafad-presentation.html

# Render to PowerPoint
marp slides.md --pptx --output scafad-presentation.pptx

# Preview in browser with live reload
marp slides.md --preview
```

### Option 2 — VS Code with Marp extension

1. Install the **Marp for VS Code** extension (ID: `marp-team.marp-vscode`).
2. Open `slides.md`.
3. Click the Marp preview icon in the top-right toolbar.
4. Use **File → Export Slide Deck** to export to PDF/HTML/PPTX.

### Option 3 — View raw Markdown

The slides are structured plain Markdown — legible without any renderer.  
`---` separates slides. YAML frontmatter at the top controls Marp theming.

---

## Presentation Structure

| Slide | Title | Duration |
|-------|-------|----------|
| 1 | Title | 0:30 |
| 2 | The Detection Gap in Serverless Computing | 1:30 |
| 3 | Three Research Questions | 1:00 |
| 4 | SCAFAD — Seven-Layer Architecture | 1:15 |
| 5 | Layer 0: 26 Heterogeneous Detectors | 1:15 |
| 6 | Layer 1: Preservation-Aware Privacy Conditioning | 1:30 |
| 7 | Layer 3: Trust-Weighted Fusion Engine | 1:00 |
| 8 | Evaluation Setup | 0:45 |
| 9 | Key Results: SCAFAD vs 14 Baselines | 1:00 |
| 10 | Threshold Calibration (WP-5.3) | 1:15 |
| 11 | RQ1: Layered Pipeline Answer | 0:45 |
| 12 | RQ2: Trust-Weighted Fusion Answer | 0:45 |
| 13 | RQ3: Preservation-Aware Conditioning Answer | 0:45 |
| 14 | Five Principal Contributions | 0:45 |
| 15 | Limitations | 1:15 |
| 16 | Future Work and Recommendations | 0:45 |
| 17 | Comparison with Related Systems | 0:45 |
| 18 | Conclusion | 0:45 |
| 19 | Thank You / Questions | — |
| **Total** | | **~15 min** |

---

## Before the Viva — Checklist

- [ ] Replace `[Author Name]` on slide 1 and 19 with your name
- [ ] Replace `[Supervisor Name]` on slide 19 with your supervisor's name
- [ ] Replace `[Institution Name]` on slide 19 with your institution
- [ ] Render and proof-read PDF output
- [ ] Run `make evaluate` to confirm all metrics reproduce (seed 42, expect 92 s)
- [ ] Read `speaker_notes.md` — especially the Q&A section on slide 19
- [ ] Have `metrics_reference.md` open as a second screen or printed reference during Q&A

---

## Metric Consistency — Key Numbers

These numbers appear across slides and must stay consistent with the dissertation:

| Value | Context |
|-------|---------|
| **F1 = 1.0000** | SCAFAD at threshold 0.09 |
| **ROC-AUC = 1.0000** | SCAFAD threshold-independent |
| **F1 = 0.8858** | Best baseline (One-Class SVM ν=0.10) |
| **+11.42 pp** | F1 gap over best baseline |
| **+10.46 pp** | ROC-AUC gap over best baseline |
| **TP=6,250 / TN=50 / FP=0 / FN=0** | Confusion matrix |
| **6,300 records** | Total evaluation corpus |
| **14 baselines** | Compared systems |
| **26 detectors** | Layer 0 detector count |
| **θ = 0.09** | Calibrated decision threshold (WP-5.3) |
| **F1 = 0.9241** | Flat ensemble ablation (unweighted) |
| **≥ 99.95%** | Anomaly-critical field retention (T-003) |
| **seed 42** | Evaluation reproducibility seed |

All values are sourced from:
- `evaluation/results/headline_metrics.json`
- `evaluation/results/baselines_results.json`
- `evaluation/results/optimal_threshold.json`
- Auditor verification reports in `context/reports/`

---

## Customisation Notes

**To add a diagram slide:** Insert a new `---` separated section in `slides.md` and add an SVG/PNG image with `![alt](path)`.

**To adjust timing:** Edit the `## Pacing guidance` table in `speaker_notes.md`.

**To export just the notes:** `speaker_notes.md` is self-contained and suitable for printing as a cue-card.

**Theme customisation:** Edit the `style:` block in the YAML frontmatter of `slides.md`. Marp supports standard CSS within the `<style>` block.

---

## Alignment with Dissertation Chapters

| Slide(s) | Dissertation source |
|----------|-------------------|
| 2 (Problem) | Chapter 1, §1.1–1.2 |
| 3 (RQs) | Chapter 1, §1.3 |
| 4–7 (Architecture) | Chapter 6–7 |
| 8 (Evaluation setup) | Chapter 9, §9.1–9.2 |
| 9–10 (Results) | Chapter 9, §9.3–9.5 |
| 11–13 (RQ answers) | Chapter 10, §10.2–10.4 |
| 14 (Contributions) | Chapter 11, §11.2 |
| 15 (Limitations) | Chapter 10, §10.7 |
| 16 (Future work) | Chapter 11, §11.4 |
| 17 (Related systems) | Chapter 10, §10.5 |
| 18 (Conclusion) | Chapter 11, §11.1 + §11.3 |

---

## Demonstration Video

The `showreel.md` file in this directory contains a complete narrated script, an ordered
shot list, and a capture checklist for a screen-recording walkthrough of the SCAFAD pipeline.

**Current recording status:** see [`SHOWREEL_STATUS.md`](SHOWREEL_STATUS.md).

- **If `showreel.mp4` is present locally:** open it directly in any media player.
- **If the recording is hosted externally:** the URL is in `showreel_url.txt` (generated from
  `showreel_url.txt.template`).
- **If recording is still PENDING:** follow the instructions in `showreel.md` to record it,
  then update `SHOWREEL_STATUS.md` accordingly.

To verify recording presence programmatically:

```bash
python scripts/verify_showreel.py
```

---

*Generated by BuilderAgent · Task 4d8b2e73 · 2026-04-26*
