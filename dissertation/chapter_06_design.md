# Chapter 6: Design

## 6.1 Overview

This chapter delivers the three design artefacts established in Chapter 5: class diagrams for three layer groups (Figures 6.2–6.4), an end-to-end pipeline diagram (Figure 6.1), and a security analyst persona (§6.5). Each artefact is traceable to a corresponding implementation component in Chapter 7.

## 6.2 System Architecture

SCAFAD is structured as a seven-layer serverless pipeline deployed on AWS Lambda (Jonas et al., 2019). Each layer has a single charter and communicates with adjacent layers exclusively through typed Python dataclass contracts, enforcing the invariant that no component may bypass a layer. Table 6.1 identifies each layer by name.

**Table 6.1 — SCAFAD Seven-Layer Pipeline**

| Layer | Name | Charter |
|-------|------|---------|
| L0 | Adaptive Telemetry Controller | Capture and HMAC-sign a schema-v4.2 `TelemetryRecord` per invocation |
| L1 | Behavioural Intake Zone | Validate, sanitise, apply privacy conditioning, and hash anomaly-critical fields |
| L2 | Multi-Vector Detection Matrix | Execute 26 parallel anomaly detectors; aggregate per-detector signals |
| L3 | Trust-Weighted Fusion | Fuse signals with source trust context; emit a composite anomaly score |
| L4 | Explainability & Decision Trace | Produce a tiered, human-readable rationale for each detection decision |
| L5 | Threat Alignment | Map decisions to MITRE ATT&CK tactics; cluster invocations into campaigns |
| L6 | Feedback & Learning | Ingest analyst labels; update trust modulation and contrastive replay buffer |

`RCoreToLayer1Adapter` is the sole schema-translation boundary between L0 and L1, confining schema-evolution logic to a single testable component. Figure 6.1 presents the Level Diagram of the seven-layer pipeline, with each tier depicted in execution order.

**Figure 6.1 — End-to-End Pipeline Diagram**

```
AWS Lambda invocation
        │
        ▼
lambda_handler.lambda_handler(event, context)
        │
        ▼
SCAFADCanonicalRuntime.process_event(event)
        │
        ├─► L0: TelemetryRecord v4.2 · HMAC signing · invocation graph build
        │         │
        │         └─► RCoreToLayer1Adapter  (schema v4.2 → v2.1)
        │
        ├─► L1: validation · sanitisation · privacy conditioning · deferred hashing
        │
        ├─► L2: 26-algorithm detection matrix → per-detector signals
        │
        ├─► L3: trust-weighted fusion → composite anomaly score
        │
        ├─► L4: explainability trace → tiered decision rationale
        │
        ├─► L5: MITRE ATT&CK alignment → campaign cluster
        │
        └─► L6: analyst label ingest → trust modulation update
                │
                ▼
        CanonicalRuntimeResult → HTTP JSON response
```

**Figure 6.2 — Class Diagram: Layers 0–2**

```
┌────────────────────────────────────────────────────┐
│      Layer0_AdaptiveTelemetryController            │
│  + capture_telemetry() : TelemetryRecord           │
│  + sign(record)        : TelemetryRecord           │
│  + run_detection()     : DetectionResult           │
└───────────────────┬────────────────────────────────┘
                    │ owns
                    ▼
┌────────────────────────────────────────────────────┐
│           AnomalyDetectionEngine                   │
│  + run_detectors()  : DetectorSignals              │
│  + fuse_signals()   : AggregateScore               │
└───────────────────┬────────────────────────────────┘
                    │ outputs to
                    ▼
┌────────────────────────────────────────────────────┐
│           RCoreToLayer1Adapter                     │
│  + adapt(record) : Layer1ProcessedRecord           │
└───────────────────┬────────────────────────────────┘
                    │ feeds
                    ▼
┌────────────────────────────────────────────────────┐
│          Layer1CanonicalPipeline                   │
│  + process_adapted_record() : Layer1ProcessedRecord│
└───────────────────┬────────────────────────────────┘
                    │ feeds
                    ▼
┌────────────────────────────────────────────────────┐
│         MultiVectorDetectionMatrix                 │
│  + run_all() : DetectorSignalCollection            │
└────────────────────────────────────────────────────┘
```

**Figure 6.3 — Class Diagram: Layer 3**

```
┌──────────────────────────────────────────────────────────┐
│              TrustWeightedFusionEngine                   │
│  + fuse(signals, trust_ctx) : FusedScore                 │
│  + compute_volatility()     : VolatilityPosture          │
│  + emit_decision()          : TrustFusionResult          │
└──────────────────────────────────────────────────────────┘
```

**Figure 6.4 — Class Diagram: Layers 4–6**

```
┌──────────────────────────────────────────────────────────┐
│         ExplainabilityDecisionEngine            (L4)    │
│  + generate_trace()    : DecisionTrace                   │
└──────────────────────────┬───────────────────────────────┘
                           │ feeds
                           ▼
┌──────────────────────────────────────────────────────────┐
│            ThreatAlignmentEngine                (L5)    │
│  + align_to_attack()   : AlignmentResult                 │
│  + cluster_campaigns() : CampaignSet                     │
└──────────────────────────┬───────────────────────────────┘
                           │ feeds
                           ▼
┌──────────────────────────────────────────────────────────┐
│           FeedbackLearningEngine                (L6)    │
│  + ingest_label()      : LabelResult                     │
│  + update_trust()      : TrustModulationState            │
└──────────────────────────────────────────────────────────┘
```

## 6.3 Component Design Decisions

Three foundational decisions shaped the component architecture. First, all Lambda invocations route through `SCAFADCanonicalRuntime`, a single orchestration surface that eliminates behavioural divergence and makes compliance logging impossible to bypass. Second, Layer 0 detection is stateless at the invocation level, ensuring horizontal scalability; cross-invocation state is externalised to L6's trust modulation tables. Third, `AdvancedInvocationGraphBuilder` constructs a directed invocation graph per execution context, enabling GNN-based detection of lateral-movement campaigns in which individual invocations appear benign in isolation.

## 6.4 Data Flow

Each Lambda invocation traverses L0 (capture and signing), the `RCoreToLayer1Adapter` (schema translation), L1 (conditioning), L2–L3 (detection and fusion), and L4–L6 (explainability, threat alignment, and feedback). Chapter 7 documents each node in detail.

## 6.5 Security Analyst Persona

The primary consumer of SCAFAD's operational output is the Security Analyst — the individual who receives L4 decision traces and L5 MITRE ATT&CK alignments through the monitoring interface and whose label submissions drive L6 trust modulation. Table 6.2 captures the persona characteristics that shaped Layer 4 explainability constraints and the Layer 3 zero-false-positive operating target.

**Table 6.2 — Security Analyst Persona**

| Attribute | Detail |
|-----------|--------|
| Role | Security operations analyst, cloud infrastructure team |
| Technical level | Competent with AWS CloudWatch; limited familiarity with ML internals |
| Primary goal | Triage actionable alerts; dismiss benign events rapidly |
| Pain point | Alert fatigue from high false-positive rates in classical detectors |
| SCAFAD interaction | Reads L4 rationale; submits accept/reject labels through the L6 feedback interface |
| Success criterion | Zero false positives; explainable alerts within 500 ms of invocation completion |

---

## References

Bass, L., Clements, P. and Kazman, R. (2012) *Software Architecture in Practice*. 3rd edn. Upper Saddle River: Addison-Wesley.

Booch, G., Rumbaugh, J. and Jacobson, I. (2005) *The Unified Modeling Language User Guide*. 2nd edn. Reading: Addison-Wesley.

Jonas, E., Schleier-Smith, J., Sreekanti, V., Tsai, C.C., Yadwadkar, N.J., Gonzalez, J.E. and Stoica, I. (2019) *Cloud Programming Simplified: A Berkeley View on Serverless Computing*. Technical Report UCB/EECS-2019-3. Berkeley: University of California.

Scarselli, F., Gori, M., Tsoi, A.C., Hagenbuchner, M. and Monfardini, G. (2009) 'The Graph Neural Network Model', *IEEE Transactions on Neural Networks*, 20(1), pp. 61–80.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          