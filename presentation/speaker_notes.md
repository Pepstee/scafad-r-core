# SCAFAD Presentation — Speaker Notes

**Version:** 1.0 · **Date:** 2026-04-26
**Estimated duration:** 12–15 minutes (18 content slides + Q&A allowance)
**Audience:** MSc dissertation viva board / academic panel

---

## Slide 1 — Title

*Open with confidence. Do not apologise for the research scope.*

"Good [morning/afternoon]. My dissertation presents SCAFAD — a seven-layer, privacy-preserving anomaly-detection framework for AWS Lambda. The work addresses a concrete gap in the literature: no published framework integrates multi-vector detection, trust-weighted signal fusion, and preservation-aware privacy conditioning in a single reproducible pipeline. Today I will walk you through the problem, the architecture, the experimental results, and what I believe are the four most important limitations."

*Timing: ~45 seconds.*

---

## Slide 2 — The Problem

*Do not rush this. The problem motivates everything else.*

"Serverless computing — and AWS Lambda in particular — executes code in containers that live and die within milliseconds. Traditional intrusion-detection systems depend on persistent session records and long-lived process tables that simply do not exist in this model. The anomalies I care about — Denial-of-Wallet billing exploitation, cryptomining, data exfiltration, lateral movement across privileged Lambda execution roles — all manifest within a single function invocation and vanish before a conventional detector can observe them.

The second tension is privacy. If you redact all personally identifiable fields to satisfy GDPR, you destroy the anomaly signal. If you preserve all fields for detection, you violate data-minimisation requirements. No published framework resolves this tension with principled field-level classification."

*Timing: ~90 seconds.*

---

## Slide 3 — Research Questions

*Emphasise that RQs yield numerical answers — this is what separates them from vague objectives.*

"Three research questions drive the study. I want to be explicit that each yields a numerical answer from controlled experimentation — not a qualitative claim. RQ1 asks whether the layered pipeline beats single-model baselines; RQ2 asks whether trust-weighted fusion beats unweighted fusion; RQ3 asks whether preservation-aware conditioning maintains detection fidelity whilst satisfying privacy. I will answer all three affirmatively with evidence."

*Likely board question:* "How did you operationalise 'sufficient' in RQ3?"
*Answer:* "The T-003 test suite defines sufficient as ≥ 99.95% retention of the five anomaly-critical fields on an adversarial edge-case corpus. The threshold was chosen to permit at most one missed field per 2,000 invocations — a level at which the downstream ROC-AUC degrades by less than 0.001."

*Timing: ~60 seconds.*

---

## Slide 4 — Architecture Overview

*Point to each layer. Do not read the diagram — narrate it.*

"The pipeline has seven layers. Telemetry enters at the top: four scalars per Lambda invocation — execution duration, memory utilisation, error rate, and concurrency burst. Layer 0 fans that signal out across 26 heterogeneous detectors. Layer 1 conditions the telemetry for privacy before it reaches Layer 2's Byzantine-fault cross-validator. Layer 3 fuses the 26 detector scores with a trust-weighted vector. Layer 4 produces an explainability trace. Layer 5 maps confirmed anomalies to MITRE ATT&CK tactics. Layer 6 receives analyst feedback and recalibrates the trust weights. The key architectural novelty is that every layer boundary is enforced by a machine-checkable contract schema — inter-layer data contracts that are verified at test time, not just assumed."

*Timing: ~75 seconds.*

---

## Slide 5 — Layer 0: 26 Detectors

*Be prepared for: "Why 26? Why not 3?"*

"Layer 0 runs 26 detectors in three families. Statistical detectors — Z-Score, IQR, Moving Average — catch distribution outliers. Classical machine-learning detectors — Isolation Forest, One-Class SVM, LOF, and others — catch feature-space anomalies. Fifteen specialised detectors address specific threat semantics: economic abuse, adversarial evasion, graph-based lateral movement, cold-start exploitation.

The number 26 is not arbitrary. The design principle is complementarity: no single detector sees the full threat landscape, but the union of 26 detectors leaves no blind spot in the four-feature space. The trust-weighted fusion layer then suppresses the noisier signals, so the cost of adding a low-quality detector is bounded — it cannot dominate the fused score."

*Board question:* "Could you achieve the same result with fewer detectors?"
*Answer:* "Possibly. The ablation in Section 10.3 shows that removing the trust-weighting degrades F1 by 0.0759. A detector-selection ablation study — removing each detector family in turn — is an identified direction for future work. The current architecture errs towards completeness because the trust weight vector penalises noisy detectors anyway."

*Timing: ~75 seconds.*

---

## Slide 6 — Layer 1: Privacy Conditioning

*This is the academic differentiator — spend time here.*

"The privacy conditioning layer is where SCAFAD makes its principal privacy-engineering contribution. The insight is that 'data minimisation' does not mean 'redact everything' — it means 'do not retain data beyond what is necessary for the specified purpose.' The specified purpose is anomaly detection. The five anomaly-critical fields — duration, memory, error rate, concurrency burst, and cold-start frequency — carry no personal data and are necessary for detection. Decision Log DL-011 provides the individual field-level justification for each.

The PII fields — IP addresses, ARNs, request paths — are pseudonymised with HMAC-SHA256, not simply deleted. Pseudonymisation preserves the ability to correlate records within a session for multi-invocation attack pattern detection whilst preventing re-identification. Invariant I-5 mandates that every conditioning decision is logged, making the pipeline fully auditable."

*Board question:* "How do you handle schema drift — new PII fields introduced by AWS?"
*Answer:* "The field classifier is a configurable allowlist: fields not explicitly classified as anomaly-critical are sent through conditioning by default. New fields default to the 'redact' path unless explicitly promoted via Decision Log update. This is a conservative default that errs on the side of privacy."

*Timing: ~90 seconds.*

---

## Slide 7 — Layer 3: Trust-Weighted Fusion

*Keep the maths minimal — focus on the intuition.*

"The fusion layer receives a 26-dimensional score vector for each record. The naive approach — simple average — gives equal weight to every detector, including DBSCAN, whose F1 on the evaluation set is 0.0896. Giving DBSCAN a 1/26 vote alongside the temporal-pattern detector degrades the fused score boundary.

Instead, the trust weight vector is maintained via exponential moving-average of each detector's observed validation-set QoS. Detectors that are right more often get more weight. The sum of all weights is constrained to exactly 1.0 — Invariant I-9 — so the fused score remains interpretable as a probability-like measure between 0 and 1. The ablation result — flat ensemble degrades F1 to 0.9241 — is direct evidence that the weight vector is the mechanism generating the performance advantage."

*Timing: ~60 seconds.*

---

## Slide 8 — Evaluation Setup

*Anticipate the synthetic data criticism — acknowledge it before they ask.*

"Before I show the results, let me be transparent about the evaluation setup. The corpus is entirely synthetic — 6,300 records generated by a parametric harness calibrated to AWS Lambda telemetry statistics from the literature, with seed 42 for bit-identical reproduction. The 125:1 class imbalance reflects published attack-traffic studies. All 14 baselines are evaluated on the same corpus under the same conditions.

The synthetic nature of the corpus is the primary limitation of this work — I will return to it explicitly in the limitations slides. For now: the evaluation is rigorous and reproducible within its stated scope."

*Timing: ~45 seconds.*

---

## Slide 9 — Key Results

*Let the numbers speak. Do not apologise for them.*

"SCAFAD achieves F1 = 1.0000, ROC-AUC = 1.0000, precision = 1.0000, recall = 1.0000, with a perfect confusion matrix: 6,250 true positives, 50 true negatives, zero false positives, zero false negatives. The best single-model baseline — One-Class SVM — achieves F1 = 0.8858. The improvement is 11.42 percentage points on F1 and 10.46 percentage points on ROC-AUC.

I want to be precise: a ROC-AUC of 1.0 on a synthetic corpus is a strong result, but it is also a flag. I will address the caveat in the limitations section."

*Timing: ~60 seconds.*

---

## Slide 10 — Threshold Calibration

*Explain why the calibration is principled, not overfitting.*

"The default threshold of 0.30 produced F1 = 0.7238 — respectable but leaving 2,705 anomalies undetected. WP-5.3 calibrated the threshold by grid search across 91 candidates from 0.05 to 0.95 in steps of 0.01. The optimal threshold is 0.09 — the lowest threshold achieving maximum F1.

The critical point is that the calibration exploits a natural score gap between the benign score distribution, whose maximum is 0.0807, and the anomalous distribution, whose minimum is 0.1781. The gap is 0.0974 wide. Any threshold within this gap achieves perfect classification — the result is not sensitive to the specific value of 0.09. This is not overfitting to a single boundary point; it is exploiting a real distributional separation."

*Board question:* "Is this gap an artefact of the synthetic generator?"
*Answer:* "Possibly. If the synthetic generator produces anomalies with fixed statistical signatures that are too distinctive, the gap may be narrower on production data. This is precisely limitation L1. The gap existence is a structural property of the layered architecture, but its width on production data is unknown."

*Timing: ~75 seconds.*

---

## Slide 11 — RQ1 Answer

*State the answer clearly. Do not hedge.*

"RQ1 is answered affirmatively. The layered, 26-detector architecture with trust-weighted fusion substantially outperforms all 14 single-model baselines on both score-level and threshold-level measures. The 10.46 pp ROC-AUC advantage is threshold-independent — it reflects the quality of the score function, not a threshold choice. The maximum single-model recall of 0.7960 implies partial overlap between baseline score distributions that SCAFAD's layered architecture eliminates entirely."

*Timing: ~45 seconds.*

---

## Slide 12 — RQ2 Answer

*The ablation is the key evidence. Explain it carefully.*

"RQ2 is answered affirmatively by a controlled ablation. I took the full SCAFAD pipeline and replaced only the trust-weighted fusion engine with a uniform average, keeping all 26 detectors and the calibrated threshold of 0.09 unchanged. F1 dropped from 1.0000 to 0.9241 and ROC-AUC dropped from 1.0000 to 0.9683.

The mechanism is clear: Moving Average and DBSCAN produce noisy scores. Under uniform weighting their noise narrows the score gap at the 125:1 imbalanced boundary, introducing misclassifications at the margin. Trust weighting suppresses them. The weight vector is not decorative — it is the mechanism."

*Timing: ~45 seconds.*

---

## Slide 13 — RQ3 Answer

*Three lines of evidence — walk through each one.*

"RQ3 is answered affirmatively by three independent lines of evidence. First: GDPR compliance — all PII fields pseudonymised with HMAC-SHA256, every conditioning decision logged. Second: field-level classification in Decision Log DL-011 — the five anomaly-critical fields were individually assessed and found to contain no personal data; they are therefore exempt from transformation under the data-minimisation principle. Third: the T-003 test suite verified ≥ 99.95% retention of these five fields on an adversarial edge-case corpus. The post-conditioning ROC-AUC of 1.0 confirms that fidelity is fully preserved."

*Timing: ~45 seconds.*

---

## Slide 14 — Five Principal Contributions

*Be able to speak to each contribution in one sentence. Board members may ask about any of them.*

"The five contributions are: a contract-enforced layered architecture with machine-checkable inter-layer schemas; preservation-aware privacy conditioning with field-level PII classification; a trust-weighted fusion engine with adaptive QoS-driven weights; budgeted tiered explainability with MITRE ATT&CK alignment and campaign tracking; and a fully reproducible evaluation harness. Each is independently evidenced in the test suite."

*Board question:* "Which of the five contributions do you consider the most original?"
*Answer:* "The preservation-aware privacy conditioning — Contribution C-2 — because it resolves a genuine tension in the literature. Every other system either applies blanket redaction (sacrificing detection fidelity) or no conditioning (sacrificing privacy compliance). The field-level decision log — DL-011 — is the mechanism that makes the resolution principled rather than ad hoc."

*Timing: ~45 seconds.*

---

## Slide 15 — Limitations

*Do not be defensive. Own the limitations — they demonstrate academic rigour.*

"I want to be explicit about the four limitations, because they define the boundary conditions within which the results should be interpreted.

L1 — synthetic data — is the most significant. The score gap may be narrower on production data. L2 — AWS Lambda only — means the calibration is not transferable to Azure or GCP without recalibration. L3 — threshold specificity — means θ = 0.09 is a hypothesis in a cold-start production deployment, not a constant. L4 — no red-team exercise — means adversarial robustness claims rest on simulated rather than real adversary behaviour.

These are not flaws in the methodology. They are the natural frontier for future work, and I address each one in the recommendations."

*Timing: ~75 seconds.*

---

## Slide 16 — Future Work

*Show you have thought beyond the dissertation.*

"The recommendations fall into three horizons. Short-term: validate the score gap on live Lambda telemetry using synthetic-anomaly injection, and build a guided threshold-calibration protocol for cold-start deployments. Medium-term: port to Azure Functions using the RCoreToLayer1Adapter breakpoint mechanism — the extension point is already in the architecture — and integrate GAN-based adversarial training into the Layer-6 feedback loop. Long-term: combine SCAFAD with ALASTOR for a full detection-to-forensics pipeline, and consider weighted F1 as the deployment metric where FP operational cost dominates FN security risk."

*Timing: ~45 seconds.*

---

## Slide 17 — Related Systems

*Keep this crisp — the board may want to test your literature depth.*

"SCAFAD occupies a broader architectural scope than any single related system. LogGD requires dense HPC logs; SCAFAD needs only four scalars. ALASTOR does forensics post-detection; SCAFAD provides the detection signal ALASTOR needs — the two systems are architecturally complementary. ARES uses GAN adversarial training; SCAFAD uses rule-based simulation — closing that gap is future work. FaaSRCA does latency root cause; SCAFAD detects security anomalies earlier and could trigger FaaSRCA. DoWNet specialises in Denial-of-Wallet; SCAFAD embeds that as one of 26 algorithms in a unified multi-threat pipeline."

*Timing: ~45 seconds.*

---

## Slide 18 — Conclusion

*End confidently. Repeat the headline numbers once more.*

"SCAFAD answers all three research questions affirmatively with direct empirical evidence. F1 = 1.000, ROC-AUC = 1.000, 11.42 percentage points above the best single-model baseline, ≥ 99.95% anomaly-critical field retention, and full GDPR compliance through HMAC-SHA256 pseudonymisation and auditable field-level conditioning decisions. The four limitations — synthetic data, single cloud provider, dataset-specific threshold, and simulated adversarial evaluation — define the frontier for production validation. The framework is fully reproducible: make evaluate reproduces every reported metric bit-identically in 92 seconds."

*Timing: ~45 seconds.*

---

## Slide 19 — Questions

*During Q&A:*
- **Listen to the full question** before answering.
- **Distinguish between**: "I know the answer," "I have a reasoned hypothesis," and "That is outside the scope of this study."
- **Defend with evidence**, not assertion: "The data shows..." not "I believe..."
- **Do not over-hedge**: if a result is strong, say so.

### Common board questions and suggested answers

**Q: "A perfect ROC-AUC on synthetic data — isn't that just telling you the synthetic generator is too easy?"**
A: "That concern is valid and is explicitly stated in Limitation L1 and Interesting Point 1 in §10.6. The score gap may be a product of the synthetic generator's fixed anomaly signatures. The test of the claim is production validation with synthetic-anomaly injection — exactly what I recommend as the first priority in future work."

**Q: "How does SCAFAD handle concept drift?"**
A: "Layer 6 implements a feedback-and-recalibration mechanism driven by analyst labels. The trust weight vector is updated via exponential moving-average of observed QoS, so detectors whose accuracy degrades under distributional shift lose weight automatically. The limitation is that this requires labelled feedback from a deployed system — it cannot act pre-deployment."

**Q: "Your test suite has 994 tests — does a passing test suite guarantee the implementation is correct?"**
A: "No. The tests verify that the implementation satisfies its specifications at the level of abstraction the tests are written at. They provide evidence, not proof. The formal verification artefacts in the formal_verification/ directory provide additional confidence for the invariant properties (I-1 through I-9), but a complete formal proof of correctness is out of scope for this project."

**Q: "The 125:1 class imbalance seems extreme — does this inflate your metrics?"**
A: "F1-score is appropriate for imbalanced datasets because it is the harmonic mean of precision and recall — it penalises false negatives and false positives equally. ROC-AUC is threshold-independent and measures score-level discrimination quality; it is not affected by class imbalance at the metric level. The confusion matrix — TP=6,250, TN=50, FP=0, FN=0 — provides the most interpretable construct."

**Q: "Why not use a deep learning approach — an LSTM or transformer on the telemetry sequence?"**
A: "Three reasons. First, SCAFAD operates on four scalars per invocation — there is insufficient feature dimensionality to justify the complexity of a deep sequence model. Second, deep models require large amounts of labelled training data; SCAFAD is designed for environments where labelled attack data is scarce. Third, explainability: Layer 4 produces a per-alert attribution trace that is straightforward with 26 interpretable detectors and becomes opaque with end-to-end neural models. The classical ensemble approach is the appropriate choice for the problem's feature dimensionality and explainability requirements."

---

## Pacing guidance

| Slide | Target time |
|-------|-------------|
| 1 (Title) | 0:30 |
| 2 (Problem) | 1:30 |
| 3 (RQs) | 1:00 |
| 4 (Architecture) | 1:15 |
| 5 (Layer 0) | 1:15 |
| 6 (Layer 1) | 1:30 |
| 7 (Layer 3) | 1:00 |
| 8 (Evaluation setup) | 0:45 |
| 9 (Key results) | 1:00 |
| 10 (Calibration) | 1:15 |
| 11 (RQ1) | 0:45 |
| 12 (RQ2) | 0:45 |
| 13 (RQ3) | 0:45 |
| 14 (Contributions) | 0:45 |
| 15 (Limitations) | 1:15 |
| 16 (Future work) | 0:45 |
| 17 (Related systems) | 0:45 |
| 18 (Conclusion) | 0:45 |
| **Total** | **~15 min** |

---

*End of speaker notes.*
