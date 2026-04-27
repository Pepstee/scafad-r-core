# Chapter 7: Implementation

## 7.1 Architecture Overview

SCAFAD is implemented as a seven-layer serverless pipeline, deployed on AWS Lambda and orchestrated through a single canonical runtime, `SCAFADCanonicalRuntime`. The canonical data path is:

```
AWS Lambda invocation
  → lambda_handler.lambda_handler(event, context)
    → SCAFADCanonicalRuntime.process_event(event)
      → Layer 0: TelemetryRecord v4.2 capture and signing
      → RCoreToLayer1Adapter.adapt(record)
      → Layer1CanonicalPipeline.process_adapted_record()
      → SCAFADMultilayerPipeline.process_record()   # L2–L6
    → CanonicalRuntimeResult.to_dict()
  → HTTP response assembly
```

This path enforces Invariant I-1: no alternative entry-point may bypass the canonical runtime. The implementation is distributed across `layer0_*.py`, `layers/layer1/`, `layers/layer2/`, and equivalent directories for L3–L6. The configuration factory at `layers/config.py` governs environment-specific parameters including GDPR, CCPA, and HIPAA compliance postures.

The configuration factory follows a strict cold-start initialisation protocol: on each Lambda container launch, it reads the `SCAFAD_ENV`, `GDPR_MODE`, and `HIPAA_MODE` environment variables and instantiates the corresponding `Layer0Config` variant (`development`, `staging`, `production`, or `gdpr_strict`). This single-source-of-truth design ensures that PII field lists, redaction budgets, and hashing algorithm choices are propagated consistently to all seven layers without duplication or drift between function instances. A companion error boundary in `lambda_handler.py` catches any unhandled exception before it surfaces to the AWS runtime: it serialises a `HANDLER_ERROR` sentinel into a partial `TelemetryRecord` and emits it through the standard L0 telemetry path, preserving audit-trail continuity for invocations that fail before `SCAFADCanonicalRuntime` fully initialises.

## 7.2 Basic Functionality Walkthrough

### L0 — Adaptive Telemetry Controller

Layer 0 is the observation layer. On each Lambda invocation, `Layer0_AdaptiveTelemetryController` coordinates five concurrent sub-engines: the `AnomalyDetectionEngine` running twenty-six detection algorithms; the `AdversarialAnomalyEngine` applying label-flip poisoning, GAN-evasion, and denial-of-wallet simulations; the `EconomicAbuseDetector` scoring billing-manipulation patterns; the `ProvenanceChainTracker` building a directed causal graph; and the `FormalVerificationEngine` computing a completeness score. Outputs are assembled into a twenty-eight-field `TelemetryRecord` (schema v4.2) with HMAC-SHA256 signature (Invariant I-6) and completeness score constrained to [0.0, 1.0] (Invariant I-7).

Layer 0 also hosts the `SignalNegotiator` (channel QoS scoring via exponential moving averages), the `RedundancyManager` (Active-Active and Active-Standby emission with a 300-second deduplication window), and the `Sampler` (adjusting per-invocation verbosity, with cold starts always at full verbosity).

### L1 — Behavioural Intake Zone

The `RCoreToLayer1Adapter` resolves five named breakpoints: schema version rewrite (v4.2→v2.1), field renames (`event_id`→`record_id`, `function_id`→`function_name`), numeric repacking, and a twenty-one-to-four-value anomaly taxonomy collapse. This adapter is the sole authorised translation boundary (Invariant I-2).

The `Layer1CanonicalPipeline` applies five conditioning phases: structural validation, sanitisation, GDPR/CCPA-aware PII redaction, SHA-256 field hashing with anomaly-critical field preservation, and a preservation scoring pass. Output is a `Layer1ProcessedRecord` with a `quality_report` and an `audit_record` logging every redaction. Invariant I-5 guarantees that no field is silently dropped.

The adapter's twenty-one-to-four anomaly taxonomy collapse presents a key integration trade-off: the reduction is required for L3 fusion compactness, but it is irreversible. To satisfy both constraints, the original twenty-one-value classification is preserved verbatim as the `raw_anomaly_types` audit field in `Layer1ProcessedRecord`, allowing L2–L3 to operate on the compact four-value taxonomy whilst audit queries retain full forensic resolution.

### L2 — Multi-Vector Detection Matrix

The `SCAFADMultilayerPipeline` routes the conditioned record to Layer 2, where twenty-six algorithms execute in parallel across four detector families, each emitting a `DetectionSignal` with a confidence score in [0, 1].

**Statistical detection (eight algorithms)** covers Z-Score, Isolation Forest, CUSUM, Pearson correlation-break, seasonal deviation, temporal burst, frequency-anomaly, and IQR methods targeting distributional anomalies across individual and combined features.

**Resource-based detection (six algorithms)** monitors memory spikes and leaks, CPU burst patterns, network I/O exfiltration signatures, packet-ratio anomalies, and ephemeral `/tmp` storage divergence.

**Execution-pattern detection (six algorithms)** tracks control-flow deviations, cold-start ratios, timeout-approach patterns, temporally clustered error bursts, Mann-Whitney latency regressions, and concurrency spikes.

**Advanced and economic detection (six algorithms)** applies Kullback-Leibler drift measurement, provenance-graph fan-out analysis, HMAC-and-schema composite scoring, and billing-manipulation detection for denial-of-wallet attacks.

The aggregate `Layer2DetectionResult` exposes `detector_consensus`, `anomaly_indicated`, and `aggregate_score` in [0, 1]. Multi-family diversity is deliberate: no single family dominates under every attack pattern, and the trust-weighted fusion at Layer 3 exploits this heterogeneity to achieve superior discrimination to any single-algorithm approach.

### L3 — Trust-Weighted Fusion

Layer 3 applies `trust_context` from the L1 quality report to the L2 consensus signals. The `Layer3FusionResult` carries `fused_score`, `volatility`, and `trust_adjusted_score`. Trust weighting corrects for source reliability, down-weighting channels with degraded QoS scores before fusion.

## 7.3 Advanced Functionality

### L4 — Explainability and Decision Trace

Layer 4 converts the fused score into a tiered decision (`benign`, `review`, or `alert`). The `Layer4DecisionTrace` carries a `tier`, a `redaction_budget`, and plain-language `rationale_blocks`; alert-tier events receive fully populated traces, benign records a single-line summary (Martin, 2008, p. 287).

### L5 — Threat Alignment and L6 — Feedback

Layer 5 maps each decision to MITRE ATT&CK tactics and techniques applicable to serverless environments, optionally assigning a `campaign_id` for multi-step attack chains. Layer 6 closes the feedback loop: analyst labels (`correct`, `false_positive`, `false_negative`) update trust-modulation coefficients, and a contrastive replay buffer stores hard mispredictions for periodic L2 recalibration (Beck, 2003, p. 41).

### Adversarial Robustness

The `AdversarialAnomalyEngine` applies eight attack simulations — including label-flip poisoning, query-free GAN evasion, and crypto-mining pattern injection — to assess L2 evasion resistance. The `ParallelTelemetryValidator` implements Merkle-tree batch integrity verification and per-record HMAC-SHA256 signing, detecting Byzantine faults and physically impossible metric combinations.

## 7.4 Version Control Evidence

The implementation is managed under Git with a conventional-commit discipline (`feat`, `fix`, `test`, `refactor`, `docs`, `chore` prefixes). Each commit references a Decision Log entry and records which tests were green at the time of merge. Representative recent commits include:

```
61d3411  refactor(core): retire legacy 3-line shims (WP-3.9, DL-012)
f952733  DL-051: Fix test_008c and test_l2_l6 evidence_items assertions
593d036  DL-050: Ignore generated evaluation results
d310de3  DL-049: Re-apply core/ shim rewrites and legacy test import fixes
782ba07  fix(layer1): remove spurious async from optimize_preservation_strategy
```

All feature development occurs on named branches (prefixed `feat/`, `fix/`, or `refactor/`) and merges to `main` via squash-merge after an architect review recorded in the Decision Log (Fowler, 2018, p. 63). Each merge requires the full unit and contract test suite to pass and work-package acceptance criteria to be checked off.

## 7.5 Test-Driven Development Evidence

All new or changed behaviour in SCAFAD is preceded by a failing test — the TDD discipline codified as a non-negotiable constraint in Appendix A of the Master Blueprint. Development follows a four-tier test pyramid.

**Tier 1 — Unit tests** cover individual classes in isolation with pytest fixtures, completing the full suite in under two seconds and providing sub-second feedback on every edit.

**Tier 2 — Contract tests** verify inter-layer invariants (Invariants I-3, I-4, I-7), confirming that `RCoreToLayer1Adapter` resolves all five named breakpoints and that `completeness_score` remains in [0.0, 1.0]; a failure indicates a broken inter-layer agreement rather than a logic error.

**Tier 3 — Integration tests** in `tests/integration/` exercise the `RCoreToLayer1Adapter → Layer1CanonicalPipeline → SCAFADMultilayerPipeline` end-to-end path, asserting on `Layer4DecisionTrace` output format and `audit_record` completeness.

**Tier 4 — Evaluation tests** reproduce headline metrics (F1 = 1.0000, ROC-AUC = 1.0000) from the cached 6,300-record test set via `make evaluate`, providing an end-to-end regression guard (Pressman and Maxim, 2015, p. 529). Permanent tests are never skipped or deleted, forming an immutable regression baseline.

## 7.6 Evaluation Artefacts and Reproducibility Evidence

End-to-end pipeline behaviour is evidenced through two complementary sources. The primary reproducibility evidence comprises the evaluation artefacts in `evaluation/results/`: the 6,300-record `scafad_results.json` score cache, the `headline_metrics.json` summary (F1 = 1.0000, ROC-AUC = 1.0000), and the `baselines_summary.md` comparison table showing SCAFAD above all fourteen baselines. These files collectively document pipeline execution from Lambda invocation through L6 analyst feedback and are fully reproducible via `make evaluate`. Supplementary demonstration evidence is provided by `presentation/showreel.md`, which contains a narrated script, an ordered shot list, and a capture checklist for a screen-recording walkthrough of the pipeline.

## 7.7 Conclusion

SCAFAD's implementation realises the seven-layer serverless architecture specified in Chapter 6. The canonical data path is enforced by ten architectural invariants; the twenty-six L2 detectors span four complementary algorithm families fused by dynamic tr