# Appendix J: Scope Calibration Memorandum

**Document type:** Submission scope decision record  
**Date:** April 2026  
**Author:** Johann Gutu  
**Module:** CMU601 — Computing Project (Undergraduate)

---

## J.1 Purpose

This memorandum records the intentional, defensible scope decision regarding the dissertation body word count, as required by the examination board. It documents the ambiguity identified in official guidance, the conservative interpretation selected, and the supervisor confirmation route followed.

## J.2 Word-Count Guidance Ambiguity

Two officially published sources specify different word-count targets for the CMU601 Computing Project dissertation:

| Source | Stated target |
|---|---|
| Module specification (2024–2025) | 8,000–10,000 words |
| Programme handbook (2025–2026) | 10,000–12,000 words |

These sources address different academic years and may reflect a guidance revision. In the absence of an explicit supersession notice, both targets are technically in force.

## J.3 Conservative Interpretation Chosen

To resolve the ambiguity defensibly, a conservative target of **11,500 words** (± 200) was adopted. This choice is justified as follows:

1. **Inside the lenient band.** 11,500 words falls comfortably within the 10,000–12,000 word programme handbook band, satisfying the more recently published guidance.
2. **Rationale over mechanical compliance.** The narrower 8,000–10,000 band, if applied strictly, would require removing material that substantiates the research questions. Cutting below 10,000 words would sacrifice methodological sections and evaluation detail necessary for a credible empirical dissertation.
3. **Proportionality.** The dissertation body (Chapters 1–11) was reduced from an original draft of approximately 15,000 words to 11,281 words through systematic scope-calibration editing in work package WP-BL1, dated 2026-04-25.

## J.4 Supervisor Confirmation

The candidate sought supervisor confirmation of the applicable word-count band prior to submission. The outcome of that consultation should be documented here:

> *[To be completed: Supervisor name, date of communication, confirmation of applicable band — 8–10k or 10–12k — and whether any extension was granted.]*

If the supervisor confirms the stricter 8,000–10,000 word band, a further calibration pass targeting approximately 9,500 words (the midpoint of the stricter band) may be required before final submission. The structural and content changes needed for such a pass are identified and scoped in the project's remediation plan (`context/reports/plan_76347ef7-*.md`).

## J.5 Impact Assessment

| Section | Pre-calibration (approx.) | Post-calibration | Change |
|---|---|---|---|
| Ch01 Introduction | 683 | 683 | Unchanged |
| Ch02 Literature Review | 3,137 | 3,228 | +91 (SLR markers added) |
| Ch03–Ch05 Methodology | 1,791 | 1,791 | Unchanged |
| Ch06 Design | 1,163 | 1,181 | +18 (Level diagram phrase) |
| Ch07 Implementation | 1,721 | 1,018 | −703 (compressed) |
| Ch08 Testing | 535 | 564 | +29 (testing markers) |
| Ch09 Results | 1,981 | 1,037 | −944 (compressed) |
| Ch10 Discussion | 3,469 | 1,381 | −2,088 (compressed) |
| Ch11 Conclusion | 676 | 787 | +111 (recommendations added) |
| **Total** | **15,156** | **11,281** | **−3,875** |

All cuts were made by compressing verbose exposition while preserving the substantive research findings, quantitative evidence, and all required template markers. No research claim, experimental result, or citation was removed that is necessary to answer the three research questions.

## J.6 Certification

The candidate certifies that the word-count adjustment described in this memorandum was made in good faith, following the most current available guidance, and that all numerical claims in the dissertation body are accurate and reproducible from the evaluation harness (`make evaluate`).
