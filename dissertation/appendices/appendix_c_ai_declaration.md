# Appendix C — AI Usage Declaration

## C.1 Overview

This appendix constitutes the AI usage declaration required by Birmingham Newman University for CMU601 Computing Project submissions. It is produced in accordance with the University's academic integrity policy and the principle of transparent disclosure of AI-assisted work.

SCAFAD was developed with significant AI assistance across multiple phases. This declaration documents every category of AI use, specifying the scope, the human-review process applied, and the locus of residual responsibility. Under-declaration would constitute an academic integrity violation; this appendix errs on the side of full transparency.

## C.2 AI Usage Summary Table

The following table categorises all AI involvement in this project. The AI assistant used was Claude (Anthropic), accessed via an agentic development environment.

| Category | Scope of AI involvement | Human review process | Residual responsibility |
|----------|------------------------|---------------------|------------------------|
| Planning agents | Structured work-package planning for the entire project: task decomposition, sequencing, dependency mapping, sprint planning, and remediation road-mapping. An AI planning agent produced detailed task specifications (work packages WP-4.x through WP-5.x and WP-GA-x), architecture documents, and handoff notes throughout the project lifecycle. | All planning outputs were reviewed by the student before implementation. Supervisor reviewed chapter structure and research design at supervision meetings (see Appendix D). | The student author retains full responsibility for all research decisions, scope choices, and the intellectual framing of the dissertation. Planning agents advised; the student decided. |
| Code generation | Per-layer detector implementations, the trust-weighted fusion engine, the preservation-aware conditioning pipeline, the evaluation harness, and the threshold calibration script were generated with AI code-generation assistance. Test files (unit tests, contract tests, integration tests) were also generated with AI assistance. | All generated code was reviewed by the student, executed, and verified against acceptance criteria. Tests must pass before code is accepted. The student inspected all generated implementations for correctness and modified them where necessary. | The student author is responsible for all code submitted. AI-generated code that was accepted unchanged was accepted because the student verified its correctness. |
| Dissertation prose | Chapter drafts were produced with AI assistance. The AI planning agent provided structured outlines; the AI assistant generated prose from those outlines. Chapters 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, and 11 were all drafted with AI involvement. | The student reviewed all dissertation prose for accuracy, coherence, and alignment with the research findings. The supervisor reviewed chapter structures and provided feedback at supervision meetings. All numeric claims were verified against the evaluation results. | The student author takes full responsibility for the dissertation text. Arguments, interpretations, and conclusions are the student's own; the AI was a drafting assistant, not the intellectual originator. |
| Test generation | The automated test suites — unit tests for each layer, contract validation tests, compliance tests, and the appendices compliance test suite in this submission — were generated with AI assistance as part of the test-driven development process. | All tests were executed and reviewed by the student. Tests that passed trivially without testing genuine behaviour were rejected and regenerated. The student verified that each test asserts the correct property. | The student author is responsible for the quality and correctness of the test suite. The 515+ passing tests reflect the student's acceptance of the test quality, not merely the AI's generation of them. |
| Evaluation runs | Pipeline evaluation runs (processing the 6,300-record corpus, computing F1/precision/recall/ROC-AUC, running the grid-search threshold optimisation) were orchestrated by AI-generated scripts and in some cases initiated by the AI assistant during development sessions. | The student inspected all evaluation outputs, verified numeric results against expected ranges, and cross-checked the headline metrics (F1 = 1.000, ROC-AUC = 1.000, threshold = 0.09) against the confusion matrix and score distribution. Results are deterministic and reproducible. | The student author is responsible for the evaluation methodology and the reported results. The AI orchestrated the computation; the student validated the outcomes and reported them honestly. |

## C.3 Tools and Models Used

| Tool / Platform | Purpose |
|----------------|---------|
| Claude (Anthropic, claude-3.5-sonnet and claude-opus-4) | Primary AI assistant: planning, code generation, prose drafting, test generation |
| Claude Code (agentic environment) | Multi-step task orchestration, file editing, test execution |
| Python 3.11 standard library + pytest | Test execution (no AI involvement in the test runtime itself) |

## C.4 Newman University Policy Alignment

This declaration is submitted in accordance with Newman University's AI usage policy for 2025–2026, which requires students to:

1. Declare all AI tool usage in an appendix. ✓ (this appendix)
2. Identify which parts of the work involved AI assistance. ✓ (§C.2 table above)
3. Confirm that the student takes responsibility for all submitted content. ✓ (responsibility column in §C.2)
4. Confirm that AI-assisted work was reviewed and cannot be submitted as the student's unassisted original thought. ✓ (human review column in §C.2)

The student acknowledges that submitting AI-generated content without appropriate declaration and review would constitute academic misconduct under the University's regulations.

## C.5 Declaration Statement

I, Johann Gutu, declare that:

- The AI usage described in this appendix is complete and accurate to the best of my knowledge.
- I have reviewed all AI-generated content (code, prose, and tests) before submitting it as part of this dissertation.
- I accept full responsibility for the intellectual content, research decisions, and conclusions presented in this dissertation.
- Where AI assistance has been used, it has been used as a tool to support my own thinking and development, not to replace it.
