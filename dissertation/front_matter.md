# SCAFAD: A Privacy-Preserving Multi-Layer Anomaly Detection Framework for Serverless Computing

---

**Student:** Johann Gutu  
**Student Number:** [Withheld for blind marking]  
**Module:** CMU601 — Computing Project (Undergraduate)  
**Institution:** Birmingham Newman University  
**Academic Year:** 2025–2026  
**Submission Date:** April 2026  
**Word Count:** 9,994 words (excluding front matter, code listings, table rows, appendices, and bibliography; including chapter body prose, headings, and in-chapter reference lists)

---

## Abstract

Serverless computing platforms, typified by AWS Lambda, introduce distinct anomaly detection challenges — ephemeral execution environments, stateless invocations, and heterogeneous telemetry streams — that render conventional intrusion detection inadequate. Regulatory obligations under the General Data Protection Regulation require privacy-preserving treatment of telemetry data that may contain personally identifiable information.

This dissertation presents SCAFAD (Serverless Computing Anomaly and Fraud Anomaly Detection), a seven-layer anomaly detection framework for AWS Lambda. SCAFAD's preservation-aware conditioning layer employs hashing and redaction to retain anomaly-critical telemetry fields at ≥99.95% fidelity whilst satisfying configurable privacy constraints. Layers 2–3 implement a multi-vector detection matrix of 26 parallel detectors aggregated via a trust-weighted fusion engine. Layers 4–6 provide explainability, MITRE ATT&CK alignment, and analyst-driven feedback. All inter-layer contracts are enforced through machine-checkable invariants.

SCAFAD was evaluated on a synthetic corpus of 6,300 AWS Lambda telemetry records (6,250 anomalous, 50 benign) spanning four attack families. Against 14 single-model baselines, SCAFAD achieved a ROC-AUC of 1.0000, surpassing the strongest baseline (ZScore, ROC-AUC = 0.8954) by 10.46 percentage points and eliminating all false positives (Precision = 1.0000). With threshold calibration at 0.09, SCAFAD achieves F1 = 1.0000 and Recall = 1.0000, surpassing the best baseline F1 of 0.8858 by 11.42 percentage points.

The principal contributions are: a contract-enforced seven-layer architecture with machine-checkable invariants; a preservation-aware conditioning layer that protects personally identifiable information without degrading anomaly fidelity; a trust-weighted fusion engine validated by ablation; a reproducible evaluation harness; and a MITRE ATT&CK–aligned explainability layer.

---

## Table of Contents

- [Chapter 1 — Introduction](#chapter-1--introduction)
  - 1.1 Background and Motivation
  - 1.2 Problem Statement
  - 1.3 Research Questions
  - 1.4 Aims and Objectives
  - 1.5 Dissertation Structure

- [Chapter 2 — Literature Review](#chapter-2--literature-review)
  - 2.1 Serverless Computing and Anomaly Detection
  - 2.2 Log-Based Anomaly Detection
  - 2.3 Intrusion Detection in Cloud and Serverless Environments
  - 2.4 Privacy-Preserving Machine Learning
  - 2.5 Multi-Model Fusion and Trust Weighting
  - 2.6 Research Gap and Positioning
  - 2.7 Chapter Summary

- [Chapter 3 — Research Methodologies](#chapter-3--research-methodologies)
  - 3.1 Introduction
  - 3.2 What is a Methodology?
  - 3.3 Quantitative and Qualitative Approaches
  - 3.4 Research Questions
  - 3.5 Justification of Selection
  - 3.6 Ethical Considerations
  - 3.7 Conclusion

- [Chapter 4 — Software Development Lifecycle Methodology](#chapter-4--software-development-lifecycle-methodology)
  - 4.1 Overview of SDLC Methodologies
  - 4.2 Waterfall
  - 4.3 Spiral
  - 4.4 Agile/Iterative Selection and Justification
  - 4.5 Mapping to the SCAFAD Phase Structure
  - 4.6 Conclusion

- [Chapter 5 — Design Methodology](#chapter-5--design-methodology)
  - 5.1 Introduction
  - 5.2 What is a Design Methodology?
  - 5.3 SSADM
  - 5.4 UML
  - 5.5 EER
  - 5.6 Selection and Justification
  - 5.7 Conclusion

- [Chapter 6 — Design](#chapter-6--design)
  - 6.1 Overview
  - 6.2 System Architecture
  - 6.3 Component Design Decisions
  - 6.4 Data Flow
  - 6.5 Security Analyst Persona

- [Chapter 7 — Implementation](#chapter-7--implementation)
  - 7.1 Architecture Overview
  - 7.2 Basic Functionality Walkthrough
  - 7.3 Advanced Functionality
  - 7.4 Version Control Evidence
  - 7.5 Test-Driven Development Evidence
  - 7.6 Showreel Video Reference
  - 7.7 Conclusion

- [Chapter 8 — Testing Methodology](#chapter-8--testing-methodology)
  - 8.1 Testing Philosophy
  - 8.2 Test Categories
  - 8.3 Evaluation Test Suite
  - 8.4 Limitations of Testing

- [Chapter 9 — Results and Evaluation](#chapter-9--results-and-evaluation)
  - 9.1 Introduction
  - 9.2 Evaluation Design
  - 9.3 Materials
  - 9.4 Procedure
  - 9.5 Baseline Comparison
  - 9.6 Headline SCAFAD Results
  - 9.7 Confusion Matrix
  - 9.8 Calibrated versus Uncalibrated F1
  - 9.9 Discussion of ROC-AUC = 1.0000
  - 9.10 Conclusion

- [Chapter 10 — Discussion](#chapter-10--discussion)
  - 10.1 Introduction
  - 10.2 Answer to RQ1: Layered detection pipeline against single-model baselines
  - 10.3 Answer to RQ2: Trust-weighted versus unweighted fusion
  - 10.4 Answer to RQ3: Preservation-aware conditioning without sacrificing detection fidelity
  - 10.5 Comparison with Related Systems
  - 10.6 Interesting Points
  - 10.7 Limitations
  - 10.8 Threats to Validity
  - 10.9 Conclusion

- [Chapter 11 — Conclusion](#chapter-11--conclusion)
  - 11.1 Summary of Work
  - 11.2 Answers to Research Questions
  - 11.3 Contributions
  - 11.4 Limitations and Future Work

---

## List of Figures

| Figure | Title | Chapter |
|--------|-------|---------|
| Figure 6.1 | End-to-End Pipeline Diagram (L0–L6) | Chapter 6 |
| Figure 6.2 | Class Diagram: Layers 0–2 | Chapter 6 |
| Figure 6.3 | Class Diagram: Layer 3 (Trust-Weighted Fusion Engine) | Chapter 6 |
| Figure 6.4 | Class Diagram: Layers 4–6 | Chapter 6 |

## List of Tables

| Table | Title | Chapter |
|-------|-------|---------|
| Table 6.1 | SCAFAD Seven-Layer Pipeline Summary | Chapter 6 |
| Table 6.2 | Security Analyst Persona | Chapter 6 |
| Table 9.1 | Evaluation Dataset Summary | Chapter 9 |
| Table 9.2 | Baseline Comparison (14 Detectors) | Chapter 9 |
| Table 9.3 | Headline SCAFAD Results | Chapter 9 |
| Table 9.4 | Confusion Matrix | Chapter 9 |

## List of Appendices

| Appendix | Title |
|----------|-------|
| Appendix A | Ethics Form |
| Appendix B | Ethics Approval / Exemption |
| Appendix C | AI Usage Declaration |
| Appendix D | Supervision Log |
| Appendix E | Requirements Specification |
| Appendix F | Detailed Design |
| Appendix G | Literature Review Evidence |
| Appendix H | Instruments |
| Appendix I | Qualitative Results |
| Appendix J | Scope Calibration Memorandum |

---

## List of Abbreviations

| Abbreviation | Expansion |
|---|---|
| ALASTOR | Audit Log and Serverless Threat Origin Reconstruction |
| ARES | Adversarial Robustness Evaluation System |
| ATT&CK | Adversarial Tactics, Techniques, and Common Knowledge (MITRE framework) |
| AUC | Area Under the Curve |
| AWS | Amazon Web Services |
| CPU | Central Processing Unit |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise |
| DL | Decision Log |
| DoWNet | Denial-of-Wallet Network (detection system) |
| EER | Enhanced Entity-Relationship (modelling notation) |
| F1 | F1-Score (harmonic mean of precision and recall) |
| FaaS | Function-as-a-Service |
| FaaSRCA | Function-as-a-Service Root Cause Analysis |
| FN | False Negative |
| FP | False Positive |
| GAN | Generative Adversarial Network |
| GDPR | General Data Protection Regulation |
| GLAD | Content-Aware Dynamic Graphs for Log Anomaly Detection |
| GNN | Graph Neural Network |
| IQR | Interquartile Range |
| IS | Information Systems |
| JSON | JavaScript Object Notation |
| L0–L6 | Layer 0 through Layer 6 (SCAFAD pipeline layers) |
| Lambda | AWS Lambda (serverless compute service) |
| LOF | Local Outlier Factor |
| LogGD | Log Graph Detector |
| LogSHIELD | Log-Based System-Level Hierarchical Intrusion and Event-Level Detection |
| ML | Machine Learning |
| MITRE | MITRE Corporation (ATT&CK framework maintainer) |
| OCSVM | One-Class Support Vector Machine |
| PII | Personally Identifiable Information |
| RQ | Research Question |
| ROC | Receiver Operating Characteristic |
| SCAFAD | Serverless Computing Anomaly and Fraud Anomaly Detection |
| SDLC | Software Development Life Cycle |
| SSADM | Structured Systems Analysis and Design Method |
| SSLA | Semi-Supervised Learning for Injection Attack Detection |
| SVM | Support Vector Machine |
| TN | True Negative |
| TP | True Positive |
| UML | Unified Modelling Language |
| WP | Work Package |
                                                                                                                                                                                                                                                       