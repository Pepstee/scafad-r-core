# Appendix G — Literature Review Evidence

## G.1 Overview

This appendix documents the systematic evidence-gathering process underpinning Chapter 2 (Literature Review). It records the database sources consulted, the search strings applied, and the inclusion and exclusion criteria used to select papers for detailed review. A PRISMA-style screening summary is included to make the selection process auditable.

## G.2 Databases Consulted

The following academic database sources were searched:

| Database | Access method | Date searched |
|----------|--------------|---------------|
| ACM Digital Library | Web | November 2025 |
| IEEE Xplore | Web | November 2025 |
| Google Scholar | Web | November–December 2025 |
| Springer Link | Web | December 2025 |
| arXiv (cs.CR, cs.DC) | Web | December 2025 |

## G.3 Search Strings

The following search string templates were applied across all databases. Terms in square brackets were substituted to produce variant queries:

**Primary search string:**
> ("serverless" OR "AWS Lambda" OR "FaaS") AND ("anomaly detection" OR "intrusion detection" OR "fraud detection")

**Secondary search string (privacy):**
> ("privacy-preserving" OR "GDPR" OR "PII") AND ("anomaly detection" OR "telemetry" OR "log analysis")

**Tertiary search string (fusion/ensemble):**
> ("ensemble" OR "fusion" OR "trust-weighted") AND ("anomaly detection" OR "outlier detection") AND ("cloud" OR "serverless")

**Quaternary search string (baselines):**
> ("Isolation Forest" OR "One-Class SVM" OR "LOF" OR "Z-Score") AND ("benchmark" OR "baseline" OR "evaluation") AND ("anomaly detection")

Boolean operators, phrase-matching quotation marks, and field-restricted searches (title, abstract, keywords) were used where supported by the database interface.

## G.4 Inclusion Criteria

A paper was included in the detailed review pool if it met **all** of the following criteria:

1. Published in a peer-reviewed venue (conference proceedings, journal, or workshop paper).
2. Published between 2015 and 2026 (inclusive) — to ensure contemporary relevance.
3. The primary topic is anomaly detection, intrusion detection, or fraud detection in a cloud, serverless, or related distributed-systems context.
4. The paper provides quantitative evaluation results (precision, recall, F1, AUC, or equivalent).
5. Full text is accessible via institutional access or open access.

## G.5 Exclusion Criteria

A paper was excluded if it met **any** of the following criteria:

1. The paper is a grey-literature source (blog post, white paper, vendor documentation) without peer review.
2. The primary domain is unrelated (e.g., medical anomaly detection, financial fraud without cloud context).
3. The paper does not present an implemented system — purely theoretical proposals without evaluation were excluded.
4. Published before 2015, unless it introduced a canonical method used as a baseline (e.g., the original LOF paper by Breunig et al., 2000).
5. Duplicate of a paper already included via a different database or search string.

## G.6 PRISMA-Style Screening Summary

The PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) framework was adapted for this literature search, which is a structured review rather than a formal systematic review.

| Stage | Count |
|-------|-------|
| Records identified via database searches | 312 |
| Duplicate records removed | 47 |
| Records screened on title and abstract | 265 |
| Records excluded at title/abstract stage | 198 |
| Full-text articles assessed for eligibility | 67 |
| Full-text articles excluded (with reasons) | 41 |
| Studies included in the detailed review (Chapter 2) | 26 |

**Reasons for full-text exclusion (41 papers):**
- Insufficient quantitative evaluation: 14
- Domain outside scope (non-cloud): 11
- Purely theoretical (no implementation): 8
- Inaccessible full text: 5
- Superseded by a newer version included: 3

## G.7 Key Papers Selected

The 26 papers included in the detailed review cover the following thematic clusters discussed in Chapter 2:
- Serverless-specific anomaly and intrusion detection (§2.1, §2.3).
- Log-based anomaly detection methods (§2.2).
- Privacy-preserving machine learning (§2.4).
- Multi-model fusion and trust-weighting techniques (§2.5).

Full bibliographic details for all 26 papers are provided in the Bibliography (bibliography.md).
