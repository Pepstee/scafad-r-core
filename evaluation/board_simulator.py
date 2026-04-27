#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median

from evidence_bundle import (
    BoardSimulationResult,
    EvidenceBundle,
    ExaminerAssessment,
    ModerationResult,
    ScenarioPrediction,
)
from grader import ScafadGrader, classify


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class ExaminerBoardSimulator:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.project_root = self.repo_root / "project" / "scafad-r-core"
        self.dissertation_dir = self.project_root / "dissertation"
        self.presentation_dir = self.project_root / "presentation"
        self.grader = ScafadGrader(self.repo_root)

    def build_evidence_bundle(self) -> EvidenceBundle:
        grader_result = self.grader.grade()
        chapter_stats = grader_result.dissertation_stats["chapters"]
        chapter_texts = {
            number: self.grader._read_text(self.dissertation_dir / f"chapter_{number:02d}_{self._chapter_suffix(number)}.md")
            for number in range(1, 12)
        }
        chapter_word_counts = {
            int(number): data["word_count"]
            for number, data in chapter_stats.items()
        }
        headline_metrics = self.grader._load_json(self.project_root / "evaluation" / "results" / "headline_metrics.json") or {}
        dimension_scores = {
            item.name: round((item.score / item.weight) * 100.0, 2)
            for item in grader_result.dimension_scores
            if item.weight
        }
        findings_by_severity: dict[str, int] = {}
        findings_by_category: dict[str, int] = {}
        for finding in grader_result.findings:
            findings_by_severity[finding.severity] = findings_by_severity.get(finding.severity, 0) + 1
            findings_by_category[finding.category] = findings_by_category.get(finding.category, 0) + 1

        showreel_status = self.presentation_dir / "SHOWREEL_STATUS.md"
        showreel_ready = False
        if showreel_status.exists():
            status_text = showreel_status.read_text(encoding="utf-8", errors="ignore").lower()
            showreel_ready = "complete" in status_text or "recorded" in status_text

        presentation_files = sum(
            1
            for path in self.presentation_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".ppt", ".pptx", ".md"}
        )

        return EvidenceBundle(
            repo_root=str(self.repo_root),
            dissertation_stats=grader_result.dissertation_stats,
            dimension_scores=dimension_scores,
            findings_by_severity=findings_by_severity,
            findings_by_category=findings_by_category,
            finding_titles=[item.title for item in grader_result.findings],
            headline_metrics=headline_metrics,
            chapter_texts=chapter_texts,
            chapter_word_counts=chapter_word_counts,
            bibliography_entries=grader_result.dissertation_stats["bibliography_entries"],
            declared_word_count=grader_result.dissertation_stats["declared_word_count"],
            body_word_count=grader_result.dissertation_stats["body_word_count"],
            showreel_ready=showreel_ready,
            presentation_files=presentation_files,
            grader_overall_score=grader_result.overall_score,
            grader_classification=grader_result.estimated_classification,
            grader_verdict=grader_result.board_verdict,
        )

    def simulate(self) -> BoardSimulationResult:
        evidence = self.build_evidence_bundle()
        assessments = [
            self._primary_marker(evidence),
            self._second_marker(evidence),
            self._methods_marker(evidence),
            self._technical_marker(evidence),
            self._integrity_marker(evidence),
        ]
        moderation = self._moderate(assessments, evidence)
        presentation_mark = self._estimate_presentation_mark(evidence)
        module_mark = round((moderation.moderated_report_mark * 0.8) + (presentation_mark * 0.2), 1)
        scenarios = self._build_scenarios(moderation.moderated_report_mark, presentation_mark, evidence)
        predicted_range = {
            "low": min(item.module_mark for item in scenarios),
            "expected": module_mark,
            "high": max(item.module_mark for item in scenarios),
        }
        confidence_band = self._confidence_band(evidence, moderation)
        board_summary = self._board_summary(evidence, moderation, presentation_mark, module_mark)
        return BoardSimulationResult(
            examiner_assessments=assessments,
            moderation=moderation,
            presentation_mark_estimate=presentation_mark,
            weighted_module_mark=module_mark,
            classification=classify(module_mark),
            confidence_band=confidence_band,
            predicted_range=predicted_range,
            stress_scenarios=scenarios,
            board_summary=board_summary,
        )

    def _chapter_suffix(self, number: int) -> str:
        return {
            1: "introduction",
            2: "literature_review",
            3: "research_methodologies",
            4: "sdlc",
            5: "design_methodology",
            6: "design",
            7: "implementation",
            8: "testing_methodology",
            9: "results_evaluation",
            10: "discussion",
            11: "conclusion",
        }[number]

    def _score_profile(
        self,
        *,
        role: str,
        emphasis: str,
        evidence: EvidenceBundle,
        structure_weight: float,
        research_weight: float,
        software_weight: float,
        integrity_weight: float,
        discussion_weight: float,
        implementation_weight: float,
        baseline: float,
        penalties: dict[str, float],
        likely_questions: list[str],
    ) -> ExaminerAssessment:
        structure = evidence.dimension_scores.get("Structure and template compliance", 0.0) / 100.0
        research = evidence.dimension_scores.get("Research rigour and academic method", 0.0) / 100.0
        software = evidence.dimension_scores.get("Software evidence and reproducibility", 0.0) / 100.0
        integrity = evidence.dimension_scores.get("Ethics, integrity, and declarations", 0.0) / 100.0
        discussion = min(evidence.chapter_word_counts.get(10, 0) / 2200.0, 1.0)
        implementation = min(evidence.chapter_word_counts.get(7, 0) / 1300.0, 1.0)

        score = baseline
        score += structure * structure_weight
        score += research * research_weight
        score += software * software_weight
        score += integrity * integrity_weight
        score += discussion * discussion_weight
        score += implementation * implementation_weight

        high_count = evidence.findings_by_severity.get("high", 0)
        medium_count = evidence.findings_by_severity.get("medium", 0)
        low_count = evidence.findings_by_severity.get("low", 0)
        score -= high_count * penalties.get("high", 0.0)
        score -= medium_count * penalties.get("medium", 0.0)
        score -= low_count * penalties.get("low", 0.0)

        if "Official module sources conflict on the allowed report word count" in evidence.finding_titles:
            score -= penalties.get("word_count_conflict", 0.0)
        if "Literature-review methodology may be under-signalled" in evidence.finding_titles:
            score -= penalties.get("slr_gap", 0.0)
        if not evidence.showreel_ready:
            score -= penalties.get("showreel_pending", 0.0)

        score = round(_clamp(score, 58.0, 88.0), 1)
        confidence = round(_clamp(0.78 - (high_count * 0.07) - (medium_count * 0.03), 0.45, 0.9), 2)

        positives = [
            "The written package is complete and aligns to the 11-chapter CMU601 structure.",
            "Technical evidence is unusually strong for an undergraduate project, with reproducibility and benchmark artefacts present.",
            f"The bibliography is substantial at approximately {evidence.bibliography_entries} entries.",
        ]
        reservations = []
        if "Official module sources conflict on the allowed report word count" in evidence.finding_titles:
            reservations.append("There is still a live word-count ambiguity between official sources, which a cautious marker would notice.")
        if "Literature-review methodology may be under-signalled" in evidence.finding_titles:
            reservations.append("The literature review appears academically credible but still under-signals parts of the SLR method.")
        if evidence.chapter_word_counts.get(10, 0) < 2200:
            reservations.append("The Discussion chapter looks shorter than ideal for a high-First dissertation and may not exploit the full evaluation evidence.")
        if not evidence.showreel_ready:
            reservations.append("Presentation readiness is not yet fully evidenced by a completed recording artefact.")

        return ExaminerAssessment(
            role=role,
            emphasis=emphasis,
            report_mark=score,
            classification=classify(score),
            confidence=confidence,
            positives=positives,
            reservations=reservations,
            likely_questions=likely_questions,
        )

    def _primary_marker(self, evidence: EvidenceBundle) -> ExaminerAssessment:
        return self._score_profile(
            role="Primary Marker",
            emphasis="Argument quality, structure, criticality, and whether the writing earns a First on its own terms.",
            evidence=evidence,
            structure_weight=8.0,
            research_weight=7.0,
            software_weight=4.0,
            integrity_weight=2.0,
            discussion_weight=5.0,
            implementation_weight=3.0,
            baseline=58.0,
            penalties={"high": 1.2, "medium": 0.8, "low": 0.2, "word_count_conflict": 2.0, "slr_gap": 2.0},
            likely_questions=[
                "Why does the discussion chapter not spend longer on limitations and alternative explanations?",
                "How exactly was the literature search conducted and filtered?",
            ],
        )

    def _second_marker(self, evidence: EvidenceBundle) -> ExaminerAssessment:
        return self._score_profile(
            role="Second Marker",
            emphasis="Independent cross-check of whether the strength survives a fresh reading without access to your intent.",
            evidence=evidence,
            structure_weight=7.0,
            research_weight=6.0,
            software_weight=5.0,
            integrity_weight=2.0,
            discussion_weight=4.0,
            implementation_weight=3.0,
            baseline=58.0,
            penalties={"high": 1.0, "medium": 0.7, "low": 0.2, "word_count_conflict": 1.8, "slr_gap": 1.5},
            likely_questions=[
                "What is the single clearest piece of evidence that SCAFAD outperforms baselines?",
                "Which chapter best demonstrates undergraduate critical evaluation rather than description?",
            ],
        )

    def _methods_marker(self, evidence: EvidenceBundle) -> ExaminerAssessment:
        return self._score_profile(
            role="Methods-Focused Examiner",
            emphasis="Methodological justification, literature interrogation, evaluation design, and academic caution.",
            evidence=evidence,
            structure_weight=6.0,
            research_weight=10.0,
            software_weight=2.0,
            integrity_weight=3.0,
            discussion_weight=6.0,
            implementation_weight=1.0,
            baseline=56.0,
            penalties={"high": 1.2, "medium": 1.0, "low": 0.3, "word_count_conflict": 2.0, "slr_gap": 3.0},
            likely_questions=[
                "Why should a marker trust the SLR process rather than the resulting bibliography alone?",
                "How do you justify the methodological stance in Chapter 3 beyond template compliance?",
            ],
        )

    def _technical_marker(self, evidence: EvidenceBundle) -> ExaminerAssessment:
        return self._score_profile(
            role="Technical Examiner",
            emphasis="Novelty and execution of the software artefact, reproducibility, and engineering ambition.",
            evidence=evidence,
            structure_weight=5.0,
            research_weight=4.0,
            software_weight=9.0,
            integrity_weight=2.0,
            discussion_weight=3.0,
            implementation_weight=5.0,
            baseline=60.0,
            penalties={"high": 0.8, "medium": 0.5, "low": 0.1, "word_count_conflict": 1.2, "slr_gap": 1.0},
            likely_questions=[
                "Which part of the system is genuinely novel rather than well-assembled?",
                "Why is the current evaluation sufficient to support the headline performance claims?",
            ],
        )

    def _integrity_marker(self, evidence: EvidenceBundle) -> ExaminerAssessment:
        return self._score_profile(
            role="Integrity and Compliance Reviewer",
            emphasis="Assessment packaging, declarations, ethics route, and whether procedural issues might suppress an otherwise strong mark.",
            evidence=evidence,
            structure_weight=8.0,
            research_weight=4.0,
            software_weight=2.0,
            integrity_weight=7.0,
            discussion_weight=2.0,
            implementation_weight=2.0,
            baseline=59.0,
            penalties={"high": 1.5, "medium": 0.7, "low": 0.2, "word_count_conflict": 2.5, "showreel_pending": 0.6},
            likely_questions=[
                "How was AI used and where is that use transparently declared?",
                "What is the defensible position on the conflicting official word-count guidance?",
            ],
        )

    def _moderate(
        self,
        assessments: list[ExaminerAssessment],
        evidence: EvidenceBundle,
    ) -> ModerationResult:
        raw_marks = [item.report_mark for item in assessments]
        spread = round(max(raw_marks) - min(raw_marks), 1)
        moderated = median(raw_marks)
        if spread >= 8:
            moderated -= 0.8
            agreement = "moderate disagreement"
            note = "Marks diverged enough that a board would likely moderate toward the central position rather than endorse the most optimistic view."
        elif spread >= 5:
            moderated -= 0.4
            agreement = "mild disagreement"
            note = "There is visible but ordinary examiner variation, so the moderated mark is slightly more conservative than the raw median."
        else:
            agreement = "strong agreement"
            note = "Examiner marks cluster tightly, so only light moderation is needed."

        if "Official module sources conflict on the allowed report word count" in evidence.finding_titles:
            moderated -= 0.6
            note += " A cautious further reduction was applied because official word-count guidance is still in conflict."
        if "Literature-review methodology may be under-signalled" in evidence.finding_titles:
            moderated -= 0.4
            note += " The under-signalled literature-review method also suppresses top-band confidence."

        moderated = round(_clamp(moderated, 58.0, 86.0), 1)
        return ModerationResult(
            pre_moderation_spread=spread,
            moderated_report_mark=moderated,
            moderation_note=note,
            agreement_level=agreement,
        )

    def _estimate_presentation_mark(self, evidence: EvidenceBundle) -> float:
        mark = 70.0
        if evidence.presentation_files >= 1:
            mark += 4.0
        if evidence.showreel_ready:
            mark += 6.0
        else:
            mark -= 3.0
        if evidence.dimension_scores.get("Software evidence and reproducibility", 0.0) >= 95:
            mark += 3.0
        if evidence.chapter_word_counts.get(7, 0) < 1300:
            mark -= 1.0
        return round(_clamp(mark, 60.0, 82.0), 1)

    def _build_scenarios(
        self,
        moderated_report_mark: float,
        presentation_mark: float,
        evidence: EvidenceBundle,
    ) -> list[ScenarioPrediction]:
        worst_report = round(_clamp(moderated_report_mark - 4.0, 55.0, 86.0), 1)
        best_report = round(_clamp(moderated_report_mark + 2.5, 55.0, 88.0), 1)
        worst_presentation = round(_clamp(presentation_mark - 6.0, 55.0, 85.0), 1)
        best_presentation = round(_clamp(presentation_mark + (6.0 if evidence.showreel_ready else 3.0), 55.0, 85.0), 1)
        expected_module = round((moderated_report_mark * 0.8) + (presentation_mark * 0.2), 1)
        return [
            ScenarioPrediction(
                name="Harsh External",
                report_mark=worst_report,
                presentation_mark=worst_presentation,
                module_mark=round((worst_report * 0.8) + (worst_presentation * 0.2), 1),
                classification=classify(round((worst_report * 0.8) + (worst_presentation * 0.2), 1)),
                rationale="A sceptical board gives extra weight to methodological signalling gaps and unresolved procedural ambiguity.",
            ),
            ScenarioPrediction(
                name="Expected Board",
                report_mark=moderated_report_mark,
                presentation_mark=presentation_mark,
                module_mark=expected_module,
                classification=classify(expected_module),
                rationale="Most likely outcome once typical marker disagreement is moderated and the current evidence profile is taken at face value.",
            ),
            ScenarioPrediction(
                name="Best Case",
                report_mark=best_report,
                presentation_mark=best_presentation,
                module_mark=round((best_report * 0.8) + (best_presentation * 0.2), 1),
                classification=classify(round((best_report * 0.8) + (best_presentation * 0.2), 1)),
                rationale="A sympathetic board credits the technical quality strongly and the presentation lands convincingly.",
            ),
        ]

    def _confidence_band(self, evidence: EvidenceBundle, moderation: ModerationResult) -> str:
        medium = evidence.findings_by_severity.get("medium", 0)
        high = evidence.findings_by_severity.get("high", 0)
        if high == 0 and medium <= 1 and moderation.pre_moderation_spread <= 4:
            return "high"
        if high <= 1 and moderation.pre_moderation_spread <= 8:
            return "medium"
        return "low"

    def _board_summary(
        self,
        evidence: EvidenceBundle,
        moderation: ModerationResult,
        presentation_mark: float,
        module_mark: float,
    ) -> list[str]:
        return [
            f"The current report reads like a strong First-class undergraduate dissertation, but not a safely predictable 90%+ script.",
            f"The moderated report prediction is {moderation.moderated_report_mark:.1f}, with a weighted module prediction of {module_mark:.1f} once the 80/20 report-presentation split is applied.",
            f"The largest reasons the model resists the current GRADER score of {evidence.grader_overall_score:.1f} are the literature-method signalling gap, shorter-than-ideal Discussion depth, and unresolved word-count ambiguity.",
            f"The estimated presentation mark is {presentation_mark:.1f}; this remains capped until a completed showreel or equivalent live-delivery evidence exists.",
        ]


def render_markdown(result: BoardSimulationResult) -> str:
    lines = [
        "# Examiner Board Simulation",
        "",
        "## Predicted Outcome",
        f"- Weighted module mark: **{result.weighted_module_mark:.1f}**",
        f"- Classification: **{result.classification}**",
        f"- Confidence band: **{result.confidence_band}**",
        f"- Predicted range: **{result.predicted_range['low']:.1f} - {result.predicted_range['high']:.1f}**",
        f"- Moderated report mark: **{result.moderation.moderated_report_mark:.1f}**",
        f"- Presentation mark estimate: **{result.presentation_mark_estimate:.1f}**",
        "",
        "## Board Summary",
    ]
    for item in result.board_summary:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Moderation",
            f"- Pre-moderation spread: **{result.moderation.pre_moderation_spread:.1f}**",
            f"- Agreement level: **{result.moderation.agreement_level}**",
            f"- Note: {result.moderation.moderation_note}",
            "",
            "## Examiner Roles",
        ]
    )
    for assessment in result.examiner_assessments:
        lines.append(f"### {assessment.role}")
        lines.append(f"- Report mark: **{assessment.report_mark:.1f}** ({assessment.classification})")
        lines.append(f"- Confidence: **{assessment.confidence:.2f}**")
        lines.append(f"- Emphasis: {assessment.emphasis}")
        lines.append("- Positives:")
        for item in assessment.positives:
            lines.append(f"  - {item}")
        lines.append("- Reservations:")
        for item in assessment.reservations:
            lines.append(f"  - {item}")
        lines.append("- Likely questions:")
        for item in assessment.likely_questions:
            lines.append(f"  - {item}")

    lines.extend(["", "## Stress Scenarios"])
    for scenario in result.stress_scenarios:
        lines.append(f"- {scenario.name}: **{scenario.module_mark:.1f}** ({scenario.classification})")
        lines.append(f"  Report={scenario.report_mark:.1f}, Presentation={scenario.presentation_mark:.1f}. {scenario.rationale}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the examiner-board simulation for the current SCAFAD package.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Workspace root containing project/scafad-r-core",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "examiner_board_report.json",
        help="Path to write the machine-readable board output.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "examiner_board_report.md",
        help="Path to write the board-style markdown report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    simulator = ExaminerBoardSimulator(args.repo_root)
    result = simulator.simulate()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
