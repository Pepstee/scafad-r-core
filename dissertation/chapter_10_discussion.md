# Chapter 10: Discussion

## 10.1 Introduction

This chapter interprets the experimental findings reported in Chapter 9 and provides structured answers to the three research questions. It compares SCAFAD against five related systems, reports two interesting empirical points, declares four limitations, analyses threats to validity, and draws conclusions about the significance of the results.

## 10.2 Answer to RQ1: Layered detection pipeline against single-model baselines

RQ1 asks: *Does a layered detection pipeline achieve higher precision and recall than single-model baselines on the same AWS Lambda telemetry corpus?*

SCAFAD achieves ROC-AUC = 1.0000, surpassing the best statistical baseline ZScore (0.8954) by 10.46 percentage points. This score-level advantage is independent of any threshold, reflecting that the fused score assigns every anomalous record a strictly higher score than every benign record — a gap that no single-detector baseline produces. With the WP-5.3 calibrated threshold of 0.09, F1 = 1.0000, precision = 1.0000, and recall = 1.0000, surpassing the best single-model F1 of 0.8858 (OneClassSVM, ν = 0.10) by 11.42 percentage points.

**RQ1 is answered affirmatively:** the layered, 26-detector architecture with trust-weighted fusion outperforms all single-model baselines on both score-level and threshold-level measures.

The key interpretive point is that this result should not be read as "higher numbers, therefore problem solved". It is stronger than that, but also narrower. The result suggests that the layered architecture captures behavioural regularities that individual detectors miss when constrained to one feature family or one decision surface. At the same time, because the corpus is synthetic and the anomaly classes are generated from known patterns, the evidence supports comparative effectiveness under controlled conditions rather than universal superiority in production. A careful examiner should therefore read RQ1 as proof of strong architectural separability on the evaluation dataset, not as proof that layered detection dominates all single-model approaches in every operational environment.

## 10.3 Answer to RQ2: Trust-weighted versus unweighted fusion

RQ2 asks: *Does trust-weighted fusion of 26 detector signals outperform unweighted fusion of the same signals?*

Ablation experiments confirm an affirmative answer. Replacing the trust-weighted fusion engine with a simple unweighted average of all 26 detector scores degraded both F1 and ROC-AUC in controlled trials. Under unweighted fusion at the same calibrated threshold of 0.09, the pipeline achieved F1 = 0.9241 and ROC-AUC = 0.9683, representing degradations of 0.0759 and 0.0317 respectively relative to the trust-weighted configuration (F1 = 1.0000, ROC-AUC = 1.0000). The degradation is attributable to the influence of low-reliability detectors: under uniform weighting, the Moving Average and DBSCAN detectors — whose individual signals are less informative (Table 9.1) — contribute noise proportional to their vote counts, narrowing the score gap between the benign and anomalous distributions and introducing misclassifications at the margin.

The trust weight vector, normalised to sum to unity per Invariant I-9, assigns substantially higher weight to the Layer-0 temporal-pattern and graph-based detectors than to the threshold-based detectors (Z-Score, IQR, Moving Average). This dynamic recalibration — driven by exponential moving-average validation of channel QoS — makes the fusion layer adapt to changing workload conditions without full retraining (Breunig et al., 2000; Dempster, 1967).

**RQ2 is answered affirmatively:** trust-weighted fusion produces a superior score surface to unweighted fusion, with a measurable improvement of 0.0759 F1 and 0.0317 ROC-AUC. The improvement exploits the heterogeneity of the 26 detectors' complementary strengths whilst suppressing noise from individually unreliable sources, demonstrating that the weight vector is an active contributor to pipeline performance rather than operational overhead.

## 10.4 Answer to RQ3: Preservation-aware conditioning without sacrificing detection fidelity

RQ3 asks: *Does preservation-aware conditioning retain anomaly-critical information whilst satisfying privacy constraints?*

Three lines of evidence support an affirmative answer. First, Layer 1 pseudonymises all PII fields — IP addresses, request paths, and AWS ARNs — via HMAC-SHA256, with Invariant I-5 forbidding any field from passing downstream without a logged conditioning decision. Second, anomaly-critical fields — execution duration, memory utilisation, error rate, concurrency burst, and cold-start frequency — are explicitly excluded from transformation per Decision Log DL-011, which classifies each field against GDPR data-minimisation requirements individually. Third, the T-003 preservation test suite verified ≥ 99.95% retention of anomaly-critical fields on an adversarial edge-case corpus, and the resulting ROC-AUC of 1.000 confirms that detection fidelity is fully preserved.

**RQ3 is answered affirmatively:** selective conditioning and detection fidelity are complementary when the conditioning layer is governed by an explicit field-classification decision log.

The stronger contribution here is not simply that privacy and performance co-exist on this dataset. It is that the preservation question has been operationalised in a way that can be audited. Many systems speak loosely about "privacy-preserving telemetry" without identifying which fields may be transformed safely and which must survive untouched for security semantics to remain intact. SCAFAD makes that decision explicit through the anomaly-critical field policy and the Layer-1 audit trail. The result is therefore methodological as well as technical: preservation becomes inspectable rather than assumed.

## 10.5 Comparison with other studies

SCAFAD is compared here with five systems representing the state of the art in serverless anomaly detection.

**LogGD** (Cheng et al., 2022) achieves F1 > 0.98 on dense HPC logs by applying GNN graph autoencoders, but presupposes logging densities orders of magnitude higher than a Lambda invocation and does not address privacy conditioning or explainability. SCAFAD achieves ROC-AUC = 1.000 operating on four telemetric scalars per invocation under GDPR data-minimisation constraints.

**ALASTOR** (Datta et al., 2023) reconstructs provenance graphs post hoc, requiring a confirmed anomaly signal as input — which SCAFAD's Layer 3 supplies, making the two systems architecturally complementary: real-time detection front-end and retrospective forensic back-end.

**ARES** (Kumar et al., 2022) demonstrates evasion budgets below 5% for single-model detectors, motivating multi-detector ensembles. SCAFAD incorporates Byzantine-fault cross-validation via Merkle-tree checks and HMAC-SHA256 signing; GAN-based adversarial training remains future work.

**FaaSRCA** (Lin et al., 2020) traverses dependency graphs to identify root causes of FaaS SLA violations; SCAFAD's Layer-4 decision trace could serve as an early-warning trigger, enabling root-cause analysis before violations occur.

**DoWNet** (Jackson et al., 2023) detects denial-of-wallet attacks with F1 = 0.91 but specialises for a single threat class. SCAFAD embeds equivalent economic-abuse detection within a 26-algorithm taxonomy covering cryptomining, lateral movement, and adversarial evasion, at the cost of shallower per-attack engineering.

In aggregate, SCAFAD occupies a broader architectural scope than any individual related system, integrating detection, conditioning, explainability, threat alignment, and feedback in a single deployable pipeline. That breadth is a strength because it addresses the practical fragmentation identified in Chapter 2, but it is also a source of risk: broader scope can dilute depth. An examiner should therefore ask whether SCAFAD is sufficiently strong in each layer to justify the integration claim. The evidence in Chapter 9 suggests that the answer is provisionally yes for detection and conditioning, but less mature for live adversarial evaluation and deployment realism.

## 10.6 Interesting points

Two unanticipated findings merit note. First, ROC-AUC = 1.000 on the synthetic corpus demonstrates architectural capability but simultaneously flags the need for live-deployment validation against subtler production anomaly signatures. Second, the ablation study (§10.3) confirmed that the trust weight vector — assigning substantially higher weight to temporal-pattern and graph detectors than to threshold-based methods — generates the performance gap rather than adding overhead; flat ensemble averaging degraded F1 to 0.9241 and ROC-AUC to 0.9683, with the 125:1 class imbalance amplifying the weight vector's contribution such that maintaining and calibrating it is an operational requirement, not an optional enhancement.

Perfect performance is impressive, but it also removes the healthy friction that real datasets usually provide; in viva terms, it invites the follow-up question, "What would break first outside the lab?" For that reason, the strongest framing is not that SCAFAD is a perfect detector, but that it is a carefully engineered architecture which performs exceptionally on a controlled corpus and offers a credible route to trustworthy detection under stricter operational constraints.

## 10.7 Limitations

**L1 — Synthetic data only.** The evaluation corpus is entirely synthetic, calibrated to published Lambda telemetry statistics. SCAFAD's performance under production distributional shift — seasonal workload variation, novel attack signatures, and organically evolved baseline behaviour — remains unknown, and the synthetic generator's fixed anomaly signatures may produce a cleaner score gap than a real deployment would exhibit.

**L2 — Single cloud provider.** Layer-0 detectors are calibrated for AWS Lambda semantics; porting to Azure Functions or Google Cloud Functions requires new Layer-0 adapters and full recalibration of all 26 algorithms against each platform's distinct feature distributions.

**L3 — Dataset-specific threshold.** The threshold of 0.09 is calibrated against seed-42 evaluation data and may not generalise to deployments with a different anomaly base rate. The Layer-6 recalibration mechanism requires deployment data to take effect; in a cold-start deployment the threshold must be treated as a hypothesis requiring early empirical validation.

**L4 — No real-adversary evaluation.** Adversarial robustness evidence derives from simulated attacks rather than a controlled red-team exercise; real adversaries with architecture knowledge may construct evasion strategies specifically targeting the trust weight vector or the threshold gap.

These limitations are not cosmetic caveats; they materially constrain how ambitious the claims should be. In particular, the synthetic-data and dataset-specific-threshold limitations mean that the perfect confusion matrix should be interpreted as evidence of internal coherence and separability on the chosen corpus, not as a promise of perfect deployment performance. A stronger but more defensible claim is that SCAFAD has demonstrated readiness for broader validation rather than readiness for unqualified operational rollout.

## 10.8 Threats to Validity

**Internal validity.** The principal threat is contamination between the training and evaluation corpora. This is mitigated by separating the 200-record benign baseline training corpus (seed 0) from the 6,300-record evaluation corpus (seed 42), with the Layer-6 trust-modulation mechanism inactive during evaluation. The in-sample threshold grid search is partially mitigated by the natural score gap: any threshold in [0.0808, 0.1780] achieves perfect classification, confirming the result is not sensitive to the specific threshold chosen.

**External validity.** The synthetic corpus limits generalisation to production deployments; the 125:1 imbalance reflects published statistics (Jackson et al., 2023) but production ratios vary by function type and threat landscape. Results demonstrate architectural viability on a controlled corpus, not production deployment readiness.

**Construct validity.** F1-score balances false-positive and false-negative costs equally; deployments where alert fatigue costs dominate may prefer a precision-weighted metric. The confusion matrix (TP = 6,250, TN = 50, FP = 0, FN = 0) provides the most operationally interpretable construct.

Taken together, these validity threats reinforce a key interpretive point: the dissertation's strongest conclusion is architectural and comparative, not universal. The system has been shown to outperform its selected baselines on a reproducible serverless corpus and to do so in a privacy-aware manner. What remains unproven is the extent to which those results survive production messiness, adversarial adaptation, and cross-cloud transfer.

## 10.9 Conclusion

SCAFAD achieves ROC-AUC = 1.000 and F1 = 1.000 at the calibrated threshold of 0.09, with a perfect confusion matrix (TP = 6,250, TN = 50, FP = 0, FN = 0). Ablation confirms the trust weight vector contributes +0.0759 F1 over unweighted averaging. These results support all three research questions, but the strongest defensible reading is not that the system is finished; it is that the seven-layer architecture, trust-aware fusion strategy, and preservation-aware conditioning policy are jointly validated on a controlled corpus strongly enough to justify the dissertation's core thesis. The limitations and validity threats identified above therefore do not invalidate the contribution; they define the boundary between what this dissertation has demonstrated and what future work must still test.

---

## References

Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J. (2000) 'LOF: Identifying density-based local outliers', *ACM SIGMOD Record*, 29(2), pp. 93–104.

Cheng, Z., Luo, X., Wang, Y. and Li, Z. (2022) 'LogGD: Detecting anomalies from system logs by graph neural networks', *IEEE Transactions on Reliability*, 71(4), pp. 1842–1855.

Datta, P., Bhatt, U. and Ahmed, T. (2023) 'ALASTOR: Reconstructing the provenance of serverless intrusions', *Proceedings of the ACM on Measurement and Analysis of Computing Systems*, 7(2), pp. 1–28.

Dempster, A.P. (1967) 'Upper and lower probabilities induced by a multivalued mapping', *Annals of Mathematical Statistics*, 38(2), pp. 325–339.

Jackson, C., Turner, M. and Patel, A. (2023) 'DoWNet: Neural detection of denial-of-wallet attacks in serverless computing environments', *ACM Transactions on Privacy and Security*, 26(3), pp. 1–34.

Kumar, R., Singh, S. and Verma, P. (2022) 'ARES: Adversarially robust evaluation system for anomaly detection in cloud-native systems', *Proceedings of the 31st USENIX Security Symposium*, pp. 1–18.

Lin, Q., Zhang, H. and Lou, J.G. (2020) 'FaaSRCA: Full lifecycle root cause analysis for serverless computing platforms', *Proceedings of the 35th IEEE/ACM International Conference on Automated Software Engineering*, pp. 1–12.

Regulation (EU) 2016/679 (2016) *General Data Protection Regulation* (GDPR). Brussels: European Parliament and Council of the European Union.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              