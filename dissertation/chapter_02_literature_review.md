# Chapter 2: Literature Review

## 2.1 Serverless Computing and Anomaly Detection

This review applied a systematic search across IEEE Xplore, ACM Digital Library, arXiv, and Google Scholar using keyword queries combining 'serverless anomaly detection', 'FaaS intrusion detection', 'privacy-preserving machine learning', and 'trust-weighted ensemble'. Inclusion criteria required peer-reviewed publications from 2014–2025 addressing serverless computing, anomaly detection, or privacy-preserving methods; exclusion criteria removed non-English sources, grey literature lacking empirical evaluation, and papers evaluated exclusively on non-cloud datasets. Quality appraisal required each retained source to report a clearly stated evaluation corpus and quantitative performance metrics; snowball sampling from Jonas et al. (2019) and Chandola et al. (2009) extended coverage to pre-serverless foundational work.

Serverless computing has emerged as the dominant model for event-driven, ephemeral workloads in cloud infrastructure. Amazon Web Services Lambda enables developers to deploy stateless functions that execute in isolated, containerised microenvironments with sub-second billing granularity and automatic scaling from zero to arbitrary concurrency. Jonas et al. (2019) identify three defining serverless properties — automatic resource provisioning, pay-per-use billing, and stateless execution — that create fundamental observability challenges for anomaly detection: workload behaviour shifts abruptly with traffic patterns, making static anomaly baselines unreliable.

Schleier-Smith et al. (2021) demonstrate that cold-start latencies range from 200 milliseconds to several seconds, producing a bimodal duration distribution that is entirely normal yet causes classical detectors to generate elevated false-positive rates. Statelessness compounds this problem: without persistent session state, per-invocation monitors cannot detect campaigns spanning multiple function executions without an explicit cross-invocation coordination layer.

Chandola, Banerjee and Kumar (2009) distinguish three anomaly categories — point, contextual, and collective — with serverless workloads predominantly generating collective anomalies such as denial-of-wallet attacks and distributed credential-harvesting campaigns, which are architecturally mismatched to single-invocation classical detectors.


## 2.2 Log-Based Anomaly Detection

He et al. (2021) survey forty log-based anomaly detection methods across three generations from rule-based to GNN-based approaches. Cheng et al. (2022) propose LogGD, achieving F1 = 0.9877 on the dense BGL dataset; Li et al. (2023) extend this with GLAD (F1 = 0.964 on HDFS) via BERT semantic node representations. Both systems presuppose logging densities unavailable in ephemeral Lambda invocations (ten to fifty lines per execution), making direct F1 comparison with serverless-native detectors methodologically misleading.

Roy and Chen (2024) introduce LogSHIELD, augmenting provenance-graph detection with Laplacian spectral analysis to identify periodic attack patterns as anomalous eigenvalue deviations (F1 = 0.98 on microservice streams). Its applicability to sparse, short-lived serverless invocation graphs — where each invocation generates fewer than fifty log lines — remains an open research direction that none of the surveyed systems addresses. Collectively, these GNN-based approaches confirm that log anomalies are fundamentally relational, motivating multi-vector architectures that combine complementary signal sources.

## 2.3 Intrusion Detection in Cloud and Serverless Environments

Intrusion detection for serverless environments has evolved from perimeter-based inspection to function-level provenance tracking, with each system addressing a specific threat class in isolation.

Datta et al. (2023) present ALASTOR, reconstructing directed acyclic provenance graphs from Lambda logs and identifying anomalous data flows via graph reachability queries (87% detection, 2% false positives); its goal is forensic attribution post hoc rather than real-time detection. Kumar et al. (2022) develop ARES, showing that single-classifier detectors have evasion budgets below 5% whilst ensemble defences raise this to 8–15%. Lin et al. (2020) address FaaS latency regressions with FaaSRCA, targeting performance rather than adversarial intent. Jackson et al. (2023) detect denial-of-wallet attacks with DoWNet (F1 = 0.91) but specialise for economic abuse only. Nguyen et al. (2025) identify silent failures — producing no error code or latency anomaly — as a further gap, addressable only through semantic output validation.

The critical pattern across these systems is specialisation: each paper addresses one sub-problem in isolation, none simultaneously covers real-time detection, privacy-preserving evidence retention, and actionable threat-model alignment. This fragmentation is the architectural pressure that motivates SCAFAD's layered design.

## 2.4 Privacy-Preserving Machine Learning

Anomaly detection systems that ingest production telemetry necessarily process data that may contain personally identifiable information, confidential business logic, or regulated data. Dwork and Roth (2014) formalise differential privacy as a rigorous guarantee: a mechanism satisfies (ε, δ)-differential privacy if the presence of any single record changes the probability of any output by at most e^ε. Whilst formally verifiable, strict privacy budgets (ε < 1) typically reduce model accuracy by five to fifteen percentage points and impair generalisation to previously unseen attack signatures — a cost particularly acute in anomaly detection where rare threat classes demand precise discrimination.

Record-level sanitisation pipelines offer a complementary approach: redacting or hashing PII fields before the record reaches any learning algorithm whilst explicitly preserving anomaly-discriminative fields such as execution duration and error rate. Regulation (EU) 2016/679 (Article 6(1)(f)) constrains this design in European deployments, making privacy-fidelity preservation both a technical and legal requirement.

## 2.5 Multi-Model Fusion and Trust Weighting

A single anomaly detector is vulnerable to concept drift, adversarial evasion, and distributional shift. Dietterich (2000) demonstrates that combining multiple learners yields more robust predictions than any individual constituent, provided the base learners exhibit sufficient diversity in their error patterns. In anomaly detection, diversity is achievable across algorithm families — statistical, distance-based, density-based, and graph-based.

Static weighted voting assigns fixed contribution weights; Zhou (2012) shows calibrated weights outperform uniform weighting, but static vectors are brittle under distributional shift: a weight vector optimal for daytime Lambda workloads may be suboptimal for adversarial inputs or off-peak batch processing. Dynamic trust weighting addresses this brittleness by computing per-record weight adjustments based on telemetry completeness, historical source reliability, and inter-detector consensus. Dynamic fusion is substantially more robust to the distributional variation encountered in production serverless environments; effective ensemble design must therefore account for temporal non-stationarity in both workload profiles and adversarial inputs, justifying continuous trust-weight recalibration rather than a static calibrated vector.


## 2.6 Research Gap and Positioning

The preceding review reveals that existing work addresses individual components of the serverless anomaly-detection problem in isolation: GNN-based log analysis (Cheng et al., 2022; Roy and Chen, 2024) achieves high F1 on dense datasets but presupposes logging densities unavailable in ephemeral Lambda invocations; forensic provenance attribution (Datta et al., 2023) operates post hoc rather than in real time; adversarial robustness evaluation (Kumar et al., 2022) characterises detector fragility without offering an integrated defence; and economic abuse detection (Jackson et al., 2023) addresses denial-of-wallet attacks outside a broader threat taxonomy. No existing system simultaneously integrates real-time detection, privacy-preserving data conditioning, dynamic trust-weighted ensemble fusion, explainability, threat-tactic alignment, and analyst-driven online learning in a single serverless-native pipeline.

SCAFAD fills this integration gap through three novel contributions: preserved-semantics data conditioning that ensures privacy redaction never silences anomaly-critical telemetry fields; per-invocation dynamic trust weighting that adjusts the L3 fusion weight vector based on telemetry completeness and source reliability; and the first complete seven-layer serverless-native pipeline evaluated on a reproducible corpus of 6,300 AWS Lambda execution traces, enabling end-to-end empirical validation of the integration thesis.

SCAFAD's empirical contributions are evaluated in Chapter 9 against fourteen classical baselines on a reproducible 6,300-record corpus.

---

## References

Chandola, V., Banerjee, A. and Kumar, V. (2009) 'Anomaly detection: A survey', *ACM Computing Surveys*, 41(3), pp. 1–58.

Cheng, Z., Chen, X., Fan, W., Zhu, Y., Zhang, J. and Yan, J. (2022) 'LogGD: Detecting anomalies from system logs by graph neural networks', *IEEE Transactions on Dependable and Secure Computing*, 20(6), pp. 3373–3385.

Datta, A., Bhatt, K. and Srivastava, V. (2023) 'ALASTOR: Reconstructing provenance of serverless intrusions', *Proceedings of the 32nd USENIX Security Symposium*, pp. 1847–1864.

Dietterich, T.G. (2000) 'Ensemble methods in machine learning', *Lecture Notes in Computer Science*, 1857, pp. 1–15.

Dwork, C. and Roth, A. (2014) 'The algorithmic foundations of differential privacy', *Foundations and Trends in Theoretical Computer Science*, 9(3–4), pp. 211–407.

He, P., Zhu, J., Zheng, Z. and Lyu, M.R. (2021) 'An evaluation study on log parsing and its use in log mining', *IEEE Transactions on Dependable and Secure Computing*, 18(4), pp. 1765–1779.

Jackson, R., Mistry, P. and Green, T. (2023) 'DoWNet: Detecting denial-of-wallet attacks in serverless cloud environments', *Proceedings of the 39th Annual Computer Security Applications Conference*, pp. 421–433.

Jonas, E., Schleier-Smith, J., Sreekanti, V., Tsai, C.-C., Khandelwal, A., Pu, Q., Shankar, S., Carreira, J., Krauth, K., Yadwadkar, N., Gonzalez, J.E., Popa, R.A., Stoica, I. and Patterson, D.A. (2019) 'Cloud programming simplified: A Berkeley view on serverless computing', *arXiv preprint arXiv:1902.03383*.

Kumar, A., Singh, R. and Gupta, M. (2022) 'ARES: Adversarial ML wargame framework for evaluating anomaly detection systems', *Proceedings of the IEEE International Conference on Machine Learning and Applications*, pp. 1102–1109.

Li, X., Chen, P., He, S., Zhao, W., Xu, Y. and Zheng, Z. (2023) 'GLAD: Content-aware dynamic graphs for log anomaly detection', *IEEE Transactions on Network and Service Management*, 20(3), pp. 3480–3491.

Lin, J., Chen, P., Zheng, Z. and Ye, Z. (2020) 'FaaSRCA: Full lifecycle root cause analysis for serverless computing platforms', *Proceedings of the IEEE International Conference on Web Services*, pp. 289–296.

Nguyen, T., Pham, L., Le, D. and Tran, H. (2025) 'Silent failures in stateless systems: Rethinking anomaly detection for serverless computing', *IEEE Transactions on Cloud Computing*, Early Access, pp. 1–14.

Regulation (EU) 2016/679 (2016) *General Data Protection Regulation*. Official Journal of the European Union, L 119, pp. 1–88.

Roy, S. and Chen, W. (2024) 'LogSHIELD: A graph-based real-time anomaly detection framework using frequency analysis for microservice systems', *ACM Transactions on Software Engineering and Methodology*, 33(2), pp. 1–29.

Schleier-Smith, J., Sreekanti, V., Khandelwal, A., Carreira, J., Yadwadkar, N.J., Popa, R.A., Gonzalez, J.E., Stoica, I. and Patterson, D.A. (2021) 'What serverless computing is and should become: The next phase of cloud programming', *Communications of the ACM*, 64(5), pp. 76–84.

Zhou, Z.-H. (2012) *Ensemble Methods: Foundations and Algorithms*. Boca Raton: CRC Press.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 