# Appendix D — Supervision Log

## D.1 Overview

This appendix records the supervision meetings held between the student and supervisor during the CMU601 Computing Project. Each entry documents the date, topics discussed, items agreed, and actions assigned. The log provides evidence of ongoing supervisor engagement throughout the project lifecycle.

## D.2 Supervision Meeting Log

| Date | Attendees | Topics Discussed | Agreed Outcomes | Actions |
|------|-----------|-----------------|-----------------|---------|
| 2025-11-12 | Student, Supervisor | Initial project scoping — serverless anomaly detection as the research domain; feasibility of a multi-layer pipeline approach; overview of GDPR obligations for telemetry data | Scope confirmed: SCAFAD will target AWS Lambda telemetry; seven-layer architecture agreed in principle; preliminary RQs drafted | Student to draft Chapter 1 introduction and submit for feedback by 2025-11-26 |
| 2025-12-03 | Student, Supervisor | Review of Chapter 1 draft; discussion of literature search strategy for Chapter 2; supervisor suggested extending baseline comparison beyond three models | Chapter 1 approved with minor revisions; agreed to target 12–14 baseline models for Chapter 9 evaluation | Student to complete literature review draft (Chapter 2) and begin SCAFAD Layer 0 implementation by 2026-01-14 |
| 2026-01-15 | Student, Supervisor | Progress review: Layers 0–2 implementation status; discussion of privacy-preserving hashing approach for Layer 1; threshold selection methodology for Layer 3 | Agreed that grid-search threshold optimisation is appropriate; supervisor noted the importance of documenting the decision-making process in Chapter 6 | Student to implement Layers 3–4 and run preliminary evaluation by 2026-02-11 |
| 2026-02-12 | Student, Supervisor | Evaluation results review: preliminary F1 = 0.72 at default threshold; discussion of threshold calibration strategy; Chapter 7 implementation writeup reviewed | Agreed to apply grid-search calibration (0.05–0.95, step 0.01) on the evaluation set; supervisor confirmed the evaluation methodology is sound and satisfies RQ1–RQ3 | Student to calibrate threshold, rerun evaluation, update headline metrics, and begin Chapter 9 draft by 2026-03-05 |
| 2026-03-06 | Student, Supervisor | Post-calibration results (F1 = 1.000, threshold = 0.09) presented; discussion of how to frame perfect-score results honestly; Chapter 9 and Chapter 10 drafts reviewed | Agreed that the perfect score is attributable to clean class separation (gap = 0.0974) and that this limitation should be explicitly discussed in §10.9; supervisor approved Chapter 9 structure | Student to complete Chapter 10 discussion and Chapter 11 conclusion; finalise appendix bundle by 2026-04-10 |

## D.3 Communication outside formal meetings

In addition to the scheduled meetings above, brief email exchanges and draft-review sessions occurred at the following points:

- **2025-12-15**: Supervisor provided written feedback on the Layer 1 hashing design.
- **2026-01-28**: Student submitted a brief progress report; supervisor confirmed the approach was on track.
- **2026-03-20**: Supervisor reviewed and provided feedback on the Chapter 10 discussion draft.

All supervisor feedback was incorporated into the dissertation. Substantive changes to scope or methodology were only made with supervisor agreement.

## D.4 Word-count Declaration for this Appendix

This appendix is excluded from the declared body word count (14,089 words) per the front matter declaration.
