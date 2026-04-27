from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceBundle:
    repo_root: str
    dissertation_stats: dict[str, Any]
    dimension_scores: dict[str, float]
    findings_by_severity: dict[str, int]
    findings_by_category: dict[str, int]
    finding_titles: list[str]
    headline_metrics: dict[str, Any]
    chapter_texts: dict[int, str]
    chapter_word_counts: dict[int, int]
    bibliography_entries: int
    declared_word_count: int | None
    body_word_count: int
    showreel_ready: bool
    presentation_files: int
    grader_overall_score: float
    grader_classification: str
    grader_verdict: str


@dataclass
class ExaminerAssessment:
    role: str
    emphasis: str
    report_mark: float
    classification: str
    confidence: float
    positives: list[str] = field(default_factory=list)
    reservations: list[str] = field(default_factory=list)
    likely_questions: list[str] = field(default_factory=list)


@dataclass
class ModerationResult:
    pre_moderation_spread: float
    moderated_report_mark: float
    moderation_note: str
    agreement_level: str


@dataclass
class ScenarioPrediction:
    name: str
    report_mark: float
    presentation_mark: float
    module_mark: float
    classification: str
    rationale: str


@dataclass
class BoardSimulationResult:
    examiner_assessments: list[ExaminerAssessment]
    moderation: ModerationResult
    presentation_mark_estimate: float
    weighted_module_mark: float
    classification: str
    confidence_band: str
    predicted_range: dict[str, float]
    stress_scenarios: list[ScenarioPrediction]
    board_summary: list[str]
