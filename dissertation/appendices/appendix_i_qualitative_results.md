# Appendix I — Qualitative Results

## I.1 Scope

This appendix addresses the requirement to include any qualitative results, themes, codes, or interview transcripts arising from the research. This declaration is explicitly included to satisfy the submission template.

## I.2 Declaration

This is a **quantitative study**. The research design is rooted in empirical software evaluation: SCAFAD was assessed against 6,300 synthetically generated AWS Lambda telemetry records using precision, recall, F1-score, ROC-AUC, confusion matrix analysis, and threshold calibration. All results are numeric, deterministic, and reproducible by re-running `python -m pytest tests/unit/` and the evaluation harness.

No qualitative data were collected. There are no interview transcripts, thematic codes, participant observation notes, discourse analysis results, or narrative findings to report. Accordingly, **not applicable**: this appendix contains no qualitative content.

## I.3 Rationale

The choice of a quantitative methodology was deliberate and is fully justified in Chapter 3 (Research Methodologies). The research questions — (RQ1) whether a layered detection pipeline outperforms single-model baselines, (RQ2) whether trust-weighted fusion outperforms unweighted fusion, and (RQ3) whether preservation-aware conditioning retains detection fidelity — are all answered through measurement rather than interpretation.

Any future work that introduces practitioner evaluation, user studies, or qualitative assessment of explainability outputs would generate qualitative results. Such an extension is noted in Chapter 11 (Limitations and Future Work) but is outside the scope of this submission. The rationale for the purely quantitative approach — reproducibility, determinism, and alignment with the ACM/IEEE software-engineering evaluation tradition — is documented in Chapter 3 (§3.8 Justification of Selection).
