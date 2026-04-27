#!/usr/bin/env python3
"""
GRADER: board-style dissertation and compliance stress test for SCAFAD.

This module simulates the behaviour of a cautious UK university examiner or
exam board. It inspects the current dissertation package, software evidence,
and governance artefacts, then produces:

- a machine-readable JSON summary for automation
- a board-style markdown report for human review

The grading model is intentionally conservative. It does not "pass" work
simply because the software is strong; it also checks template compliance,
ethics/integrity evidence, scope calibration, and submission readiness.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


SEVERITY_PENALTY = {
    "critical": 1.0,
    "high": 0.6,
    "medium": 0.3,
    "low": 0.1,
}

CHAPTER_FILES = {
    1: "chapter_01_introduction.md",
    2: "chapter_02_literature_review.md",
    3: "chapter_03_research_methodologies.md",
    4: "chapter_04_sdlc.md",
    5: "chapter_05_design_methodology.md",
    6: "chapter_06_design.md",
    7: "chapter_07_implementation.md",
    8: "chapter_08_testing_methodology.md",
    9: "chapter_09_results_evaluation.md",
    10: "chapter_10_discussion.md",
    11: "chapter_11_conclusion.md",
}

CHAPTER_WORD_TARGETS = {
    1: (450, 900),
    2: (2200, 3400),
    3: (450, 900),
    4: (250, 700),
    5: (250, 700),
    6: (350, 900),
    7: (1300, 2300),
    8: (350, 900),
    9: (1300, 2300),
    10: (2200, 3400),
    11: (450, 900),
}

OFFICIAL_WORD_COUNT_BANDS = [
    {
        "source": "Module Information.docx",
        "min": 10000,
        "max": 12000,
        "weight": "high",
    },
    {
        "source": "CMU601 Moodle assignment brief snapshot (user-provided)",
        "min": 8000,
        "max": 10000,
        "weight": "high",
    },
]

REQUIRED_APPENDIX_PATTERNS = {
    "ethics_form": ["*ethics*form*"],
    "ethics_approval": ["*ethics*approval*", "*approval*certificate*"],
    "record_of_meetings": ["*record*meeting*", "*meeting*log*", "*supervisor*meeting*", "*supervision*log*"],
    "ai_declaration": ["*ai*declaration*", "*generative*ai*", "*ai*usage*"],
    "showreel": ["*.mp4", "*.mov", "*.mkv", "*.webm"],
    "presentation": ["*.pptx", "*.ppt", "*slides*.md", "*slide*deck*.md", "*presentation*.md"],
    "surveys_and_questionnaires": ["*survey*", "*questionnaire*", "*participant*information*", "*consent*form*", "*instrument*"],
    "detailed_design_docs": ["*design*document*", "*class*diagram*", "*level*diagram*", "*persona*", "*appendix*f*design*", "*detailed*design*"],
    "requirement_specification": ["*requirement*spec*", "*appendix*e*requirement*", "*requirements*"],
    "literature_review_data": ["*prisma*", "*literature*review*", "*search*string*", "*data*extract*"],
    "qualitative_results": ["*qualitative*result*", "*interview*", "*theme*"],
}

APPENDIX_CONTENT_MARKERS = {
    "record_of_meetings": ["record of meetings", "supervision log", "meeting log"],
    "surveys_and_questionnaires": ["questionnaire", "participant information", "consent form", "research instruments"],
    "detailed_design_docs": ["detailed design", "class diagram", "level diagram", "persona"],
    "requirement_specification": ["requirements specification", "functional requirements", "non-functional requirements"],
}

OFFICIAL_REFERENCES = [
    {
        "name": "Birmingham Newman University generative AI guidelines",
        "url": "https://newman.ac.uk/student-life-and-support/ai-guidelines/",
    },
    {
        "name": "Birmingham Newman University research policies and strategy",
        "url": "https://newman.ac.uk/about-us/governance-and-management/policies-and-procedures/research-policies-and-strategy/",
    },
]

EXAMINER_BEHAVIOUR_SOURCES = [
    r"C:\Users\Gutua\Downloads\examiner_article.pdf",
    r"C:\Users\Gutua\Downloads\ExaminerreportsfeedbakorassessmentJEAP.pdf",
    r"C:\Users\Gutua\Downloads\What examiners do what thesis students should know.pdf",
    r"C:\Users\Gutua\Downloads\examguid.pdf",
]

CHAPTER_EXPECTATIONS = {
    1: ["research question", "aim", "objective"],
    2: ["gap", "review", "literature"],
    3: ["quantitative", "qualitative", "methodology"],
    4: ["agile", "waterfall", "iterative"],
    5: ["uml", "ssadm", "eer"],
    7: ["version control", "showreel", "tdd"],
    8: ["functional", "structural", "error", "contract"],
    9: ["baseline", "roc-auc", "f1", "confusion matrix"],
    10: ["rq1", "rq2", "rq3", "limitation"],
    11: ["future work", "recommendation", "contribution"],
}

TEMPLATE_SUBSECTION_EXPECTATIONS = {
    6: ["class diagram", "level diagram", "persona"],
    7: ["basic functionality", "advanced", "version control", "tdd", "showreel"],
    8: ["functional testing", "structural testing", "error-oriented", "playtesting", "alpha", "beta"],
    9: ["procedure", "materials", "results"],
    10: ["comparison with other studies", "interesting points"],
    11: ["main findings", "recommendation", "future research"],
}


@dataclass
class Finding:
    severity: str
    category: str
    title: str
    detail: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class DimensionScore:
    name: str
    weight: int
    score: float
    rationale: str


@dataclass
class LensResult:
    name: str
    score: float
    classification: str
    verdict: str
    emphasis: str
    key_findings: list[Finding] = field(default_factory=list)


@dataclass
class TaskPriority:
    title: str
    priority_score: int
    priority_bucket: str
    execution_type: str
    suggested_task_type: str
    rationale: str
    depends_on: list[str] = field(default_factory=list)
    supporting_findings: list[str] = field(default_factory=list)


@dataclass
class GraderResult:
    overall_score: float
    estimated_classification: str
    board_verdict: str
    submission_readiness: str
    academic_quality_score: float
    compliance_score: float
    board_lenses: list[LensResult]
    findings: list[Finding]
    dimension_scores: list[DimensionScore]
    task_priorities: list[TaskPriority]
    dissertation_stats: dict[str, Any]
    evidence_sources: dict[str, Any]
    run_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


def markdown_to_text(content: str) -> str:
    """Strip common markdown syntax for rough word-counting."""
    content = re.sub(r"```.*?```", " ", content, flags=re.DOTALL)
    content = re.sub(r"`[^`]+`", " ", content)
    content = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", content)
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
    content = re.sub(r"^#+\s*", " ", content, flags=re.MULTILINE)
    content = re.sub(r"^\|.*\|$", " ", content, flags=re.MULTILINE)
    content = re.sub(r"[*_>#-]", " ", content)
    return re.sub(r"\s+", " ", content).strip()


def count_words(content: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", markdown_to_text(content)))


def extract_abstract(front_matter: str) -> str:
    match = re.search(r"## Abstract\s+(.*?)(?:\n---|\n## )", front_matter, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_declared_word_count(front_matter: str) -> int | None:
    match = re.search(r"Word Count:\s*.*?(\d[\d,]*)", front_matter, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def find_files(
    search_roots: list[Path],
    patterns: list[str],
    content_markers: list[str] | None = None,
) -> list[str]:
    matches: list[str] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file() and not any(
                    blocked in path.parts
                    for blocked in {".aws-sam", "__pycache__", ".git", ".pytest_cache", "archive"}
                ):
                    matches.append(str(path))
        if matches or not content_markers:
            continue
        # Fallback to content-aware appendix discovery so stable appendix names
        # such as appendix_e_requirements.md are recognised even when they do
        # not include the exact filename phrase the template uses.
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".rst"}:
                continue
            if any(
                blocked in path.parts
                for blocked in {".aws-sam", "__pycache__", ".git", ".pytest_cache", "archive"}
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if any(marker in text for marker in content_markers):
                matches.append(str(path))
    return sorted(set(matches))


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = [path for path in directory.glob(pattern) if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def classify(score: float) -> str:
    if score >= 90:
        return "Outstanding / likely 90%+"
    if score >= 80:
        return "High First / likely 80-89%"
    if score >= 70:
        return "First / likely 70-79%"
    if score >= 60:
        return "Upper Second / likely 60-69%"
    if score >= 50:
        return "Lower Second / likely 50-59%"
    if score >= 40:
        return "Third / borderline pass"
    return "Fail risk / below pass threshold"


class ScafadGrader:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.project_root = self.repo_root / "project" / "scafad-r-core"
        self.dissertation_dir = self.project_root / "dissertation"
        self.docs_dir = self.project_root / "docs"
        self.evaluation_dir = self.project_root / "evaluation"
        self.results_dir = self.evaluation_dir / "results"
        self.context_reports_dir = self.repo_root / "context" / "reports"
        self.front_matter_path = self.dissertation_dir / "front_matter.md"
        self.bibliography_path = self.dissertation_dir / "bibliography.md"
        self.search_roots = [
            self.dissertation_dir,
            self.docs_dir,
            self.project_root / "appendices",
        ]
        self.presentation_dir = self.project_root / "presentation"
        self.appendix_search_roots = [
            self.project_root / "appendices",
            self.dissertation_dir / "appendices",
            self.docs_dir / "appendices",
            self.presentation_dir,  # presentation package (slides.md, speaker_notes.md, etc.)
        ]

    def grade(self) -> GraderResult:
        findings: list[Finding] = []

        front_matter = self._read_text(self.front_matter_path)
        bibliography = self._read_text(self.bibliography_path)
        chapter_stats = self._chapter_stats()
        body_word_count = sum(item["word_count"] for item in chapter_stats.values())
        abstract = extract_abstract(front_matter)
        abstract_words = count_words(abstract)
        declared_word_count = parse_declared_word_count(front_matter)
        bibliography_count = self._count_references(bibliography)
        appendix_evidence = {
            key: find_files(
                self.appendix_search_roots,
                patterns,
                APPENDIX_CONTENT_MARKERS.get(key),
            )
            for key, patterns in REQUIRED_APPENDIX_PATTERNS.items()
        }
        headline_metrics = self._load_json(self.results_dir / "headline_metrics.json")
        chapter_audit = self._read_latest_chapter_audit()

        findings.extend(self._structure_findings(front_matter, abstract_words, body_word_count, chapter_stats))
        findings.extend(self._examiner_practice_findings(front_matter, abstract_words, bibliography_count))
        findings.extend(self._chapter_substance_findings())
        findings.extend(self._research_findings(bibliography_count, chapter_stats))
        findings.extend(self._appendix_findings(appendix_evidence))
        findings.extend(self._software_findings(headline_metrics, chapter_audit))
        findings.extend(self._ethics_findings(appendix_evidence, front_matter))
        findings.extend(self._scope_findings(front_matter, declared_word_count, body_word_count))
        findings.extend(self._presentation_findings(appendix_evidence))
        findings.extend(self._board_readiness_findings(chapter_audit))

        dimensions = [
            self._score_structure(findings, body_word_count, bibliography_count),
            self._score_chapter_quality(findings),
            self._score_appendices(findings),
            self._score_research(findings, bibliography_count),
            self._score_software(findings, headline_metrics, chapter_audit),
            self._score_ethics(findings, appendix_evidence),
            self._score_scope(findings, declared_word_count, body_word_count),
            self._score_presentation(findings, appendix_evidence),
        ]

        academic_dimensions = [
            item for item in dimensions
            if item.name
            not in {
                "Ethics, integrity, and declarations",
                "Presentation and submission readiness",
            }
        ]
        compliance_dimensions = [
            item for item in dimensions
            if item.name
            in {
                "Structure and template compliance",
                "Ethics, integrity, and declarations",
                "Undergraduate scope calibration",
                "Presentation and submission readiness",
            }
        ]
        academic_quality_score = round(sum(item.score for item in academic_dimensions), 1)
        compliance_score = round(sum(item.score for item in compliance_dimensions), 1)
        overall_score = round(sum(item.score for item in dimensions), 1)
        critical_count = sum(1 for item in findings if item.severity == "critical")
        high_count = sum(1 for item in findings if item.severity == "high")
        if critical_count >= 2:
            verdict = "Board hold: major compliance blockers remain."
            readiness = "Not submission-ready"
        elif critical_count == 1 or high_count >= 4:
            verdict = "Conditional board caution: substantial repair required."
            readiness = "Conditionally submission-ready at best"
        elif high_count >= 1:
            verdict = "Provisionally viable, but risk remains."
            readiness = "Submission possible with targeted cleanup"
        else:
            verdict = "Credible submission package."
            readiness = "Submission-ready"
        board_lenses = self._build_board_lenses(findings, dimensions, academic_quality_score, compliance_score)

        task_priorities = self._build_task_priorities(findings)

        return GraderResult(
            overall_score=overall_score,
            estimated_classification=classify(overall_score),
            board_verdict=verdict,
            submission_readiness=readiness,
            academic_quality_score=academic_quality_score,
            compliance_score=compliance_score,
            board_lenses=board_lenses,
            findings=findings,
            dimension_scores=dimensions,
            task_priorities=task_priorities,
            dissertation_stats={
                "body_word_count": body_word_count,
                "abstract_word_count": abstract_words,
                "declared_word_count": declared_word_count,
                "bibliography_entries": bibliography_count,
                "chapters": chapter_stats,
            },
            evidence_sources={
                "local": [
                    str(self.front_matter_path),
                    str(self.bibliography_path),
                    str(self.results_dir / "headline_metrics.json"),
                    str(self.context_reports_dir / "chapter_audit.md"),
                    str(self.repo_root / "context" / "CMU601_90_PERCENT_PLAN.md"),
                    str(self.repo_root / "SCAFAD_DISSERTATION_BIBLE.md"),
                ],
                "official": OFFICIAL_REFERENCES,
                "examiner_behaviour": EXAMINER_BEHAVIOUR_SOURCES,
            },
        )

    def _chapter_stats(self) -> dict[int, dict[str, Any]]:
        stats: dict[int, dict[str, Any]] = {}
        for number, filename in CHAPTER_FILES.items():
            path = self.dissertation_dir / filename
            content = self._read_text(path)
            stats[number] = {
                "path": str(path),
                "exists": path.exists(),
                "word_count": count_words(content) if content else 0,
            }
        return stats

    def _count_references(self, bibliography: str) -> int:
        return sum(
            1
            for line in bibliography.splitlines()
            if re.match(r"^[A-Z][A-Za-z'`.-]+, .*?\(\d{4}\)", line.strip())
        )

    def _load_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_latest_chapter_audit(self) -> str:
        report = self.context_reports_dir / "chapter_audit.md"
        if report.exists():
            return report.read_text(encoding="utf-8", errors="ignore")
        latest_audit = latest_file(self.context_reports_dir, "audit_report_*.md")
        if latest_audit is None:
            return ""
        return latest_audit.read_text(encoding="utf-8", errors="ignore")

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")

    def _structure_findings(
        self,
        front_matter: str,
        abstract_words: int,
        body_word_count: int,
        chapter_stats: dict[int, dict[str, Any]],
    ) -> list[Finding]:
        findings: list[Finding] = []
        if not self.front_matter_path.exists():
            findings.append(Finding("critical", "structure", "Front matter missing", "The dissertation package is missing front_matter.md.", [str(self.front_matter_path)]))
        if not self.bibliography_path.exists():
            findings.append(Finding("critical", "structure", "Bibliography missing", "The dissertation package is missing bibliography.md.", [str(self.bibliography_path)]))
        missing = [str(number) for number, stat in chapter_stats.items() if not stat["exists"]]
        if missing:
            findings.append(Finding("critical", "structure", "Missing required chapters", f"CMU601 expects 11 numbered chapters; missing chapters: {', '.join(missing)}.", [self.dissertation_dir.as_posix()]))
        if "CMU601" not in front_matter:
            findings.append(Finding("high", "structure", "Module code not stated clearly", "Front matter should identify the submission as CMU601.", [str(self.front_matter_path)]))
        if "MSc" in front_matter:
            findings.append(Finding("critical", "structure", "Front matter still labels the project as MSc", "The current governing material describes a CMU601 undergraduate-style project structure; an MSc label is a major compliance mismatch.", [str(self.front_matter_path)]))
        if abstract_words and not 200 <= abstract_words <= 300:
            findings.append(Finding("medium", "structure", "Abstract appears outside CMU601 target range", f"Abstract is approximately {abstract_words} words; current planning documents expect 200-300 words.", [str(self.front_matter_path)]))
        band_pairs = {(item["min"], item["max"]) for item in OFFICIAL_WORD_COUNT_BANDS}
        exceeds_all = all(not band["min"] <= body_word_count <= band["max"] for band in OFFICIAL_WORD_COUNT_BANDS)
        if len(band_pairs) > 1:
            # When the body is within at least one official band the submission IS calibrated to a
            # valid target; the conflict is administrative/institutional.  Use MEDIUM severity so
            # it surfaces as a risk-to-monitor rather than a board-blocking HIGH finding.
            # When the body sits outside ALL known bands the conflict is compounded by
            # non-compliance with every official source — that remains HIGH or CRITICAL (handled
            # by the exceeds_all branch below).
            conflict_severity = "high" if exceeds_all else "medium"
            findings.append(Finding(conflict_severity, "structure", "Official module sources conflict on the allowed report word count", "The supplied module materials do not agree fully: one source states 10,000-12,000 words, while the live Moodle brief states 8,000-10,000 words. The submission should be calibrated conservatively and the discrepancy clarified with the supervisor.", [str(self.repo_root / "context" / "CMU601_90_PERCENT_PLAN.md"), str(self.front_matter_path)]))
        if exceeds_all:
            strict_upper = min(item["max"] for item in OFFICIAL_WORD_COUNT_BANDS)
            lenient_upper = max(item["max"] for item in OFFICIAL_WORD_COUNT_BANDS)
            severity = "critical" if body_word_count > lenient_upper or body_word_count < min(item["min"] for item in OFFICIAL_WORD_COUNT_BANDS) else "high"
            findings.append(Finding(severity, "structure", "Body word count is outside all known official target bands", f"Chapters 1-11 total approximately {body_word_count} words; this exceeds both the stricter 8,000-10,000 band and the more lenient 10,000-12,000 band.", [self.dissertation_dir.as_posix()]))
        for number, stat in chapter_stats.items():
            low, high = CHAPTER_WORD_TARGETS[number]
            if stat["exists"] and stat["word_count"] > 0 and not low <= stat["word_count"] <= high:
                findings.append(Finding("low", "structure", f"Chapter {number} is outside its suggested size band", f"Chapter {number} is approximately {stat['word_count']} words; suggested range is {low}-{high}.", [stat["path"]]))
        return findings

    def _chapter_substance_findings(self) -> list[Finding]:
        findings: list[Finding] = []
        for number, expectations in CHAPTER_EXPECTATIONS.items():
            content = self._read_text(self.dissertation_dir / CHAPTER_FILES[number]).lower()
            if not content:
                continue
            missing = [term for term in expectations if term not in content]
            if len(missing) >= 2:
                findings.append(
                    Finding(
                        "medium",
                        "chapter_quality",
                        f"Chapter {number} may be missing expected examiner-facing signals",
                        f"Could not strongly evidence several expected markers for Chapter {number}: {', '.join(missing[:4])}.",
                        [str(self.dissertation_dir / CHAPTER_FILES[number])],
                    )
                )
        for number, expectations in TEMPLATE_SUBSECTION_EXPECTATIONS.items():
            content = self._read_text(self.dissertation_dir / CHAPTER_FILES[number]).lower()
            if not content:
                continue
            missing = [term for term in expectations if term not in content]
            if missing:
                findings.append(
                    Finding(
                        "medium" if len(missing) >= 2 else "low",
                        "chapter_quality",
                        f"Chapter {number} is missing one or more template-specific subsections",
                        f"Template-oriented markers not found for Chapter {number}: {', '.join(missing[:6])}.",
                        [str(self.dissertation_dir / CHAPTER_FILES[number])],
                    )
                )
        return findings

    def _examiner_practice_findings(
        self,
        front_matter: str,
        abstract_words: int,
        bibliography_count: int,
    ) -> list[Finding]:
        findings: list[Finding] = []
        front_lower = front_matter.lower()
        first_impression_gaps: list[str] = []

        if abstract_words and not 200 <= abstract_words <= 300:
            first_impression_gaps.append(f"abstract length ({abstract_words} words)")
        if "list of appendices" not in front_lower and "appendix" not in front_lower:
            first_impression_gaps.append("appendices are not clearly signposted in the front matter")
        if bibliography_count < 30:
            first_impression_gaps.append(f"reference depth looks light ({bibliography_count} entries)")
        if "word count:" in front_lower and any(token in front_lower for token in ["14089", "14866", "22000"]):
            first_impression_gaps.append("front-matter word-count cues immediately suggest over-length")

        if len(first_impression_gaps) >= 2:
            findings.append(
                Finding(
                    "high",
                    "board_readiness",
                    "Examiner first-impression package is likely to create avoidable doubt",
                    "Examiner-behaviour literature suggests that abstract, contents/front matter, references, and conclusion shape the tone of the whole read. Current early-reading signals raise concern around "
                    + ", ".join(first_impression_gaps[:4])
                    + ".",
                    [str(self.front_matter_path), str(self.bibliography_path)],
                )
            )

        if "contribution" not in front_lower and "principal contributions" not in front_lower:
            findings.append(
                Finding(
                    "medium",
                    "board_readiness",
                    "Front matter does not foreground contribution clearly enough for a skimming examiner",
                    "Examiner guidance suggests readers often skim the abstract and early pages to locate the thesis warrant and contribution quickly.",
                    [str(self.front_matter_path)],
                )
            )
        return findings

    def _research_findings(self, bibliography_count: int, chapter_stats: dict[int, dict[str, Any]]) -> list[Finding]:
        findings: list[Finding] = []
        if bibliography_count < 20:
            findings.append(Finding("high", "research", "Bibliography depth is below the stated minimum", f"Detected approximately {bibliography_count} Harvard-style references; the current project plan expects at least 20.", [str(self.bibliography_path)]))
        lit_review = self._read_text(self.dissertation_dir / CHAPTER_FILES[2]).lower()
        if lit_review:
            missing_signals = [
                term for term in ["inclusion", "exclusion", "snowball", "keyword", "quality"] if term not in lit_review
            ]
            if len(missing_signals) >= 3:
                findings.append(Finding("medium", "research", "Literature-review methodology may be under-signalled", f"Chapter 2 does not strongly evidence several SLR markers: {', '.join(missing_signals)}.", [str(self.dissertation_dir / CHAPTER_FILES[2])]))
        for chapter_number in (3, 4, 5, 8):
            chapter = self._read_text(self.dissertation_dir / CHAPTER_FILES[chapter_number]).lower()
            if not chapter:
                continue
            if chapter_number == 3 and not any(term in chapter for term in ["epistemology", "ontology", "quantitative", "qualitative"]):
                findings.append(Finding("medium", "research", "Research methodology chapter lacks expected methodological vocabulary", "Chapter 3 should make the research stance explicit.", [str(self.dissertation_dir / CHAPTER_FILES[3])]))
            if chapter_number == 4 and not any(term in chapter for term in ["agile", "waterfall", "iterative"]):
                findings.append(Finding("medium", "research", "SDLC methodology chapter lacks clear comparison language", "Chapter 4 should justify the chosen development methodology against alternatives.", [str(self.dissertation_dir / CHAPTER_FILES[4])]))
            if chapter_number == 5 and not any(term in chapter for term in ["uml", "ssadm", "eer"]):
                findings.append(Finding("medium", "research", "Design methodology chapter lacks method-comparison evidence", "Chapter 5 should show why the selected design method was chosen.", [str(self.dissertation_dir / CHAPTER_FILES[5])]))
            if chapter_number == 8 and not any(term in chapter for term in ["functional", "structural", "error", "contract"]):
                findings.append(Finding("medium", "research", "Testing methodology chapter lacks breadth markers", "Chapter 8 should cover the main testing categories and justify the chosen mix.", [str(self.dissertation_dir / CHAPTER_FILES[8])]))
        return findings

    def _appendix_findings(self, appendix_evidence: dict[str, list[str]]) -> list[Finding]:
        findings: list[Finding] = []
        appendix_requirements = {
            "ethics_form": ("high", "No ethics form appendix located", "The report template expects an ethics form in the appendices."),
            "ethics_approval": ("high", "No ethics approval appendix located", "The report template expects ethics approval documentation or an equivalent approved non-human-participant route."),
            "surveys_and_questionnaires": ("medium", "No survey/questionnaire or participant-information appendix located", "The template expects research instruments or related participant materials where relevant."),
            "detailed_design_docs": ("medium", "No detailed design appendix artefacts located", "The template expects supporting design documentation in the appendices."),
            "requirement_specification": ("medium", "No requirement-specification appendix located", "Requirement specifications are named explicitly in the template appendices."),
            "literature_review_data": ("medium", "No literature-review appendix data located", "The template expects supporting literature-review material such as PRISMA, search strings, or extraction sheets."),
            "qualitative_results": ("low", "No qualitative-results appendix located", "If qualitative or mixed-method evidence is claimed, raw or summarised qualitative appendices should exist."),
            "ai_declaration": ("high", "No AI-usage appendix located", "The supplied report template and recent university AI guidance expect a declaration of generative-AI use."),
        }
        for key, (severity, title, detail) in appendix_requirements.items():
            if not appendix_evidence.get(key):
                findings.append(Finding(severity, "appendices", title, detail, [self.project_root.as_posix()]))
        return findings

    def _software_findings(self, headline_metrics: dict[str, Any] | None, chapter_audit: str) -> list[Finding]:
        findings: list[Finding] = []
        if headline_metrics is None:
            findings.append(Finding("high", "software", "No headline evaluation metrics found", "The board cannot verify the reported software performance without headline_metrics.json.", [str(self.results_dir / "headline_metrics.json")]))
            return findings
        claims = headline_metrics.get("dissertation_claims", {})
        if not claims.get("all_pass"):
            findings.append(Finding("high", "software", "Dissertation claims do not currently reproduce cleanly", "The evaluation harness reports at least one dissertation claim outside tolerance.", [str(self.results_dir / "headline_metrics.json")]))
        scafad = headline_metrics.get("scafad", {})
        if scafad.get("f1", 0) < 0.8:
            findings.append(Finding("medium", "software", "Headline F1 remains weak", f"Current F1 is {scafad.get('f1')}; this may attract examiner scrutiny unless framed carefully.", [str(self.results_dir / "headline_metrics.json")]))
        if not (self.evaluation_dir / "run_all.py").exists():
            findings.append(Finding("high", "software", "Reproducibility harness script missing", "A board-style audit expects a reproducible evaluation entry point.", [str(self.evaluation_dir / "run_all.py")]))
        if not (self.project_root / "Makefile").exists():
            findings.append(Finding("low", "software", "Makefile missing", "A single-command reproduction path strengthens the software evidence.", [str(self.project_root / "Makefile")]))
        if chapter_audit:
            regression_match = re.search(r"(\d+)\s+passed,\s+(\d+)\s+failed", chapter_audit)
            if regression_match:
                passed = int(regression_match.group(1))
                failed = int(regression_match.group(2))
                if failed > 0:
                    findings.append(Finding("high", "software", "Regression baseline is not currently clean", f"The latest chapter audit records {passed} passed and {failed} failed tests, so software/dissertation support evidence is not yet fully green.", [str(self.context_reports_dir / "chapter_audit.md")]))
        return findings

    def _ethics_findings(self, appendix_evidence: dict[str, list[str]], front_matter: str) -> list[Finding]:
        findings: list[Finding] = []
        ethics_chapters = (
            self._read_text(self.dissertation_dir / CHAPTER_FILES[3]).lower()
            + " "
            + self._read_text(self.dissertation_dir / CHAPTER_FILES[11]).lower()
        )
        ethics_rationale_present = any(
            marker in ethics_chapters
            for marker in ["ethics exemption", "no human participant", "synthetic dataset", "synthetic corpus"]
        )
        if not appendix_evidence["ethics_approval"]:
            severity = "high" if ethics_rationale_present else "critical"
            findings.append(Finding(severity, "ethics", "No ethics approval evidence located", "The current dissertation planning documents expect ethics evidence or a clearly documented exemption position.", [self.dissertation_dir.as_posix(), (self.repo_root / "context" / "CMU601_90_PERCENT_PLAN.md").as_posix()]))
        if not appendix_evidence["record_of_meetings"]:
            findings.append(Finding("high", "ethics", "No record-of-meetings evidence located", "A signed supervisor meeting trail is expected in the current submission plan.", [self.repo_root.as_posix()]))
        if not appendix_evidence["ai_declaration"]:
            findings.append(Finding("high", "ethics", "No AI usage declaration located", "Current Newman guidance expects declared and bounded generative-AI use where applicable.", [self.repo_root.as_posix()]))
        if "synthetic" not in front_matter.lower():
            findings.append(Finding("low", "ethics", "Front matter does not foreground synthetic-data provenance", "Where ethics sensitivity exists, the dissertation should be explicit about synthetic or non-human-subject data provenance.", [str(self.front_matter_path)]))
        return findings

    def _scope_findings(
        self,
        front_matter: str,
        declared_word_count: int | None,
        body_word_count: int,
    ) -> list[Finding]:
        findings: list[Finding] = []
        if declared_word_count is not None and declared_word_count > 14000:
            findings.append(Finding("critical", "scope", "Declared word count looks oversized for the current module", f"Front matter declares approximately {declared_word_count} words, which is materially above the current CMU601 target band.", [str(self.front_matter_path)]))
        if body_word_count > 12000:
            findings.append(Finding("high", "scope", "Body length suggests over-scoping", "The dissertation currently reads more like an oversized project than a tightly controlled undergraduate submission package.", [self.dissertation_dir.as_posix()]))
        if "MSc" in front_matter or "postgraduate" in front_matter.lower():
            findings.append(Finding("high", "scope", "Submission framing is misaligned with undergraduate expectations", "Markers may treat postgraduate framing as evidence that the submission has not been calibrated to the actual module.", [str(self.front_matter_path)]))
        return findings

    def _presentation_findings(self, appendix_evidence: dict[str, list[str]]) -> list[Finding]:
        findings: list[Finding] = []
        chapter_7 = self._read_text(self.dissertation_dir / CHAPTER_FILES[7]).lower()
        showreel_status_path = self.presentation_dir / "SHOWREEL_STATUS.md"
        if not appendix_evidence["showreel"]:
            # Check whether a showreel preparation kit exists even if the recording
            # has not yet been committed.  SHOWREEL_STATUS.md with status PENDING means
            # the repo-editable preparation is done and the human recording step is next.
            if showreel_status_path.exists():
                status_text = self._read_text(showreel_status_path).lower()
                if "pending" in status_text:
                    findings.append(Finding(
                        "medium",
                        "presentation",
                        "Showreel preparation kit present; recording still pending human action",
                        "The showreel script, shot list, and status tracker (SHOWREEL_STATUS.md) are committed. "
                        "The screen-recording itself is a human-dependent step that has not yet been completed.",
                        [showreel_status_path.as_posix()],
                    ))
                else:
                    # Status file exists but indicates a non-PENDING state — treat as partial evidence
                    findings.append(Finding(
                        "medium",
                        "presentation",
                        "Showreel status file present but no video artefact committed",
                        "SHOWREEL_STATUS.md exists but no video file was found; verify the recording "
                        "and commit it (or set an external URL) per the instructions in SHOWREEL_STATUS.md.",
                        [showreel_status_path.as_posix()],
                    ))
            else:
                findings.append(Finding(
                    "high",
                    "presentation",
                    "No showreel evidence located",
                    "The current project plan treats the demonstration video as a required "
                    "presentation-support artefact.",
                    [self.repo_root.as_posix()],
                ))
        elif "showreel" not in chapter_7 and "video" not in chapter_7:
            findings.append(Finding("medium", "presentation", "Showreel artefact exists but is not clearly referenced in Chapter 7", "The implementation chapter should point examiners to the recorded demonstration.", [str(self.dissertation_dir / CHAPTER_FILES[7])]))
        if not appendix_evidence["presentation"]:
            findings.append(Finding("medium", "presentation", "No presentation artefact located", "A 20-minute presentation component exists; the repo currently shows no slide deck or presentation package.", [self.repo_root.as_posix()]))
        return findings

    def _board_readiness_findings(self, chapter_audit: str) -> list[Finding]:
        findings: list[Finding] = []
        discussion = self._read_text(self.dissertation_dir / CHAPTER_FILES[10]).lower()
        conclusion = self._read_text(self.dissertation_dir / CHAPTER_FILES[11]).lower()
        if not all(term in discussion for term in ["rq1", "rq2", "rq3"]):
            findings.append(Finding("high", "board_readiness", "Discussion chapter does not explicitly answer all research questions", "A board-style reading expects direct answers to RQ1-RQ3 in the Discussion chapter.", [str(self.dissertation_dir / CHAPTER_FILES[10])]))
        if "future work" not in conclusion and "future research" not in conclusion:
            findings.append(Finding("medium", "board_readiness", "Conclusion lacks explicit future-work framing", "The conclusion should propose further work and risk-aware next steps.", [str(self.dissertation_dir / CHAPTER_FILES[11])]))
        if "recommendation" not in conclusion and "practitioner" not in conclusion:
            findings.append(Finding("medium", "board_readiness", "Conclusion lacks practitioner-facing recommendations", "The current CMU601 guidance expects recommendations and next action, not only summary.", [str(self.dissertation_dir / CHAPTER_FILES[11])]))
        if "reflect" not in conclusion and "learning" not in conclusion and "personal development" not in conclusion:
            findings.append(Finding("medium", "board_readiness", "Conclusion lacks explicit reflection on the learning process", "The module learning outcomes include reflection and further personal development; the conclusion should evidence that directly.", [str(self.dissertation_dir / CHAPTER_FILES[11])]))
        if not chapter_audit:
            findings.append(Finding("low", "board_readiness", "No recent chapter-level audit summary located", "A cross-chapter audit strengthens confidence in consistency and readiness.", [str(self.context_reports_dir / "chapter_audit.md")]))
        return findings

    def _score_dimension(self, name: str, weight: int, relevant_findings: list[Finding], base_rationale: str) -> DimensionScore:
        penalty_points = sum(SEVERITY_PENALTY[item.severity] for item in relevant_findings)
        score = max(0.0, round(weight - penalty_points * (weight / 6.0), 1))
        return DimensionScore(name=name, weight=weight, score=score, rationale=base_rationale)

    def _score_structure(self, findings: list[Finding], body_word_count: int, bibliography_count: int) -> DimensionScore:
        relevant = [item for item in findings if item.category == "structure"]
        rationale = f"Structure score reflects chapter completeness, front-matter alignment, abstract sizing, and total body words ({body_word_count}); bibliography currently counts approximately {bibliography_count} entries."
        return self._score_dimension("Structure and template compliance", 20, relevant, rationale)

    def _score_chapter_quality(self, findings: list[Finding]) -> DimensionScore:
        relevant = [item for item in findings if item.category in {"chapter_quality", "board_readiness"}]
        rationale = "Chapter-quality score reflects whether examiner-facing content is explicit: RQ answers, limitations, recommendations, and chapter-specific required signals."
        return self._score_dimension("Chapter substance and examiner-readiness", 15, relevant, rationale)

    def _score_appendices(self, findings: list[Finding]) -> DimensionScore:
        relevant = [item for item in findings if item.category == "appendices"]
        rationale = "Appendix score reflects the presence of the supporting evidence named in the supplied report structure and ethics materials."
        return self._score_dimension("Appendices and supporting evidence", 10, relevant, rationale)

    def _score_research(self, findings: list[Finding], bibliography_count: int) -> DimensionScore:
        relevant = [item for item in findings if item.category == "research"]
        rationale = f"Research score reflects literature depth, methodology signalling, and critical chapter coverage; bibliography count is approximately {bibliography_count}."
        return self._score_dimension("Research rigour and academic method", 10, relevant, rationale)

    def _score_software(self, findings: list[Finding], headline_metrics: dict[str, Any] | None, chapter_audit: str) -> DimensionScore:
        relevant = [item for item in findings if item.category == "software"]
        if headline_metrics:
            scafad = headline_metrics.get("scafad", {})
            metrics = f"headline metrics show F1={scafad.get('f1')} and ROC-AUC={scafad.get('roc_auc')}"
        else:
            metrics = "headline evaluation evidence is missing"
        audit_note = " chapter audit reviewed." if chapter_audit else " no chapter audit located."
        rationale = f"Software score reflects reproducibility and quantitative evidence; {metrics};{audit_note}"
        return self._score_dimension("Software evidence and reproducibility", 15, relevant, rationale)

    def _score_ethics(self, findings: list[Finding], appendix_evidence: dict[str, list[str]]) -> DimensionScore:
        relevant = [item for item in findings if item.category == "ethics"]
        located = [name for name, files in appendix_evidence.items() if files]
        rationale = f"Ethics/integrity score reflects evidence of approval, supervision, disclosure, and provenance; located artefact groups: {', '.join(located) or 'none'}."
        return self._score_dimension("Ethics, integrity, and declarations", 15, relevant, rationale)

    def _score_scope(self, findings: list[Finding], declared_word_count: int | None, body_word_count: int) -> DimensionScore:
        relevant = [item for item in findings if item.category == "scope"]
        rationale = f"Scope score reflects whether the work is calibrated to undergraduate project expectations rather than postgraduate framing; declared words={declared_word_count}, body words={body_word_count}."
        return self._score_dimension("Undergraduate scope calibration", 10, relevant, rationale)

    def _score_presentation(self, findings: list[Finding], appendix_evidence: dict[str, list[str]]) -> DimensionScore:
        relevant = [item for item in findings if item.category == "presentation"]
        rationale = f"Presentation score reflects showreel and viva/presentation readiness; showreel files={len(appendix_evidence['showreel'])}, presentation files={len(appendix_evidence['presentation'])}."
        return self._score_dimension("Presentation and submission readiness", 5, relevant, rationale)

    def _build_board_lenses(
        self,
        findings: list[Finding],
        dimensions: list[DimensionScore],
        academic_quality_score: float,
        compliance_score: float,
    ) -> list[LensResult]:
        dimension_map = {item.name: item for item in dimensions}

        def normalized(dimension_name: str) -> float:
            dimension = dimension_map[dimension_name]
            if dimension.weight <= 0:
                return 0.0
            return (dimension.score / dimension.weight) * 100.0

        def weighted_score(parts: list[tuple[str, float]]) -> float:
            total_weight = sum(weight for _, weight in parts)
            if total_weight <= 0:
                return 0.0
            return round(sum(normalized(name) * weight for name, weight in parts) / total_weight, 1)

        def select_findings(categories: set[str], limit: int = 5) -> list[Finding]:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            chosen = [item for item in findings if item.category in categories]
            chosen.sort(key=lambda item: (severity_order[item.severity], item.category, item.title))
            return chosen[:limit]

        def lens_verdict(score: float, critical_count: int, high_count: int, *, strict: bool = False) -> str:
            if critical_count >= 2:
                return "Serious reservations"
            if strict and critical_count >= 1:
                return "Escalate for formal concern"
            if critical_count >= 1 or high_count >= 3:
                return "Conditional concern"
            if score >= 80:
                return "Strongly positive"
            if score >= 70:
                return "Positive with reservations"
            return "Material weaknesses remain"

        marker_categories = {"structure", "chapter_quality", "research", "scope"}
        second_marker_categories = {"chapter_quality", "research", "software", "board_readiness"}
        chair_categories = {"structure", "appendices", "scope", "presentation"}
        integrity_categories = {"ethics", "appendices", "structure", "scope"}
        viva_categories = {"software", "board_readiness", "chapter_quality", "presentation"}

        marker_findings = select_findings(marker_categories)
        second_marker_findings = select_findings(second_marker_categories)
        chair_findings = select_findings(chair_categories)
        integrity_findings = select_findings(integrity_categories)
        viva_findings = select_findings(viva_categories)

        marker_score = weighted_score([
            ("Chapter substance and examiner-readiness", 0.35),
            ("Research rigour and academic method", 0.25),
            ("Structure and template compliance", 0.20),
            ("Undergraduate scope calibration", 0.20),
        ])
        second_marker_score = weighted_score([
            ("Research rigour and academic method", 0.30),
            ("Chapter substance and examiner-readiness", 0.30),
            ("Software evidence and reproducibility", 0.20),
            ("Structure and template compliance", 0.20),
        ])
        chair_score = weighted_score([
            ("Structure and template compliance", 0.30),
            ("Appendices and supporting evidence", 0.20),
            ("Undergraduate scope calibration", 0.25),
            ("Presentation and submission readiness", 0.10),
            ("Ethics, integrity, and declarations", 0.15),
        ])
        integrity_score = weighted_score([
            ("Ethics, integrity, and declarations", 0.50),
            ("Appendices and supporting evidence", 0.20),
            ("Structure and template compliance", 0.15),
            ("Undergraduate scope calibration", 0.15),
        ])
        viva_score = weighted_score([
            ("Software evidence and reproducibility", 0.35),
            ("Chapter substance and examiner-readiness", 0.25),
            ("Presentation and submission readiness", 0.20),
            ("Research rigour and academic method", 0.20),
        ])

        return [
            LensResult(
                name="Marker",
                score=marker_score,
                classification=classify(marker_score),
                verdict=lens_verdict(
                    marker_score,
                    sum(1 for item in marker_findings if item.severity == "critical"),
                    sum(1 for item in marker_findings if item.severity == "high"),
                ),
                emphasis="Primary academic marker: argument quality, structure, scope calibration, and chapter-level completeness.",
                key_findings=marker_findings,
            ),
            LensResult(
                name="Second Marker",
                score=second_marker_score,
                classification=classify(second_marker_score),
                verdict=lens_verdict(
                    second_marker_score,
                    sum(1 for item in second_marker_findings if item.severity == "critical"),
                    sum(1 for item in second_marker_findings if item.severity == "high"),
                ),
                emphasis="Independent academic cross-check: research depth, technical contribution, and whether strengths survive re-reading.",
                key_findings=second_marker_findings,
            ),
            LensResult(
                name="Chair",
                score=chair_score,
                classification=classify(chair_score),
                verdict=lens_verdict(
                    chair_score,
                    sum(1 for item in chair_findings if item.severity == "critical"),
                    sum(1 for item in chair_findings if item.severity == "high"),
                ),
                emphasis="Board chair lens: submission compliance, packaging, and whether the work is safe to advance without procedural challenge.",
                key_findings=chair_findings,
            ),
            LensResult(
                name="Integrity Reviewer",
                score=integrity_score,
                classification=classify(integrity_score),
                verdict=lens_verdict(
                    integrity_score,
                    sum(1 for item in integrity_findings if item.severity == "critical"),
                    sum(1 for item in integrity_findings if item.severity == "high"),
                    strict=True,
                ),
                emphasis="Integrity/risk lens: ethics route, AI declaration, authorship transparency, and evidence sufficiency.",
                key_findings=integrity_findings,
            ),
            LensResult(
                name="Viva Examiner",
                score=viva_score,
                classification=classify(viva_score),
                verdict=lens_verdict(
                    viva_score,
                    sum(1 for item in viva_findings if item.severity == "critical"),
                    sum(1 for item in viva_findings if item.severity == "high"),
                ),
                emphasis="Oral defence lens: software mastery, evaluator confidence, demo readiness, and whether the candidate can explain the work under pressure.",
                key_findings=viva_findings,
            ),
        ]

    def _build_task_priorities(self, findings: list[Finding]) -> list[TaskPriority]:
        priorities: list[TaskPriority] = []

        def matching_findings(*needles: str) -> list[Finding]:
            lowered_needles = [needle.lower() for needle in needles]
            matches: list[Finding] = []
            for item in findings:
                haystacks = [item.title.lower(), item.detail.lower(), item.category.lower()]
                if any(needle in haystack for needle in lowered_needles for haystack in haystacks):
                    matches.append(item)
            return matches

        def bucket(score: int) -> str:
            if score >= 90:
                return "critical"
            if score >= 75:
                return "high"
            if score >= 60:
                return "medium"
            return "low"

        def add_task(
            title: str,
            score: int,
            execution_type: str,
            suggested_task_type: str,
            rationale: str,
            depends_on: list[str],
            supporting: list[Finding],
        ) -> None:
            adjusted = score
            if execution_type == "human_dependent":
                adjusted = min(adjusted, 45)
            else:
                adjusted = max(adjusted, 50)
            priorities.append(
                TaskPriority(
                    title=title,
                    priority_score=adjusted,
                    priority_bucket=bucket(adjusted),
                    execution_type=execution_type,
                    suggested_task_type=suggested_task_type,
                    rationale=rationale,
                    depends_on=depends_on,
                    supporting_findings=[item.title for item in supporting],
                )
            )

        word_count_findings = matching_findings("word count", "over-scoping", "scope calibration")
        conflict_findings = matching_findings("official module sources conflict", "clarified with the supervisor")
        presentation_findings = matching_findings("presentation artefact")
        showreel_findings = matching_findings("showreel")
        chapter_signposting_findings = matching_findings("chapter 7", "template-specific subsections")

        if word_count_findings:
            add_task(
                title="Trim dissertation body to a submission-safe word-count band and update the declared count honestly",
                score=97,
                execution_type="engineering",
                suggested_task_type="implement",
                rationale="Scope calibration is the largest board blocker and can be reduced directly in-repo before any external clarification arrives.",
                depends_on=[],
                supporting=word_count_findings,
            )

        if presentation_findings:
            add_task(
                title="Create a submission-ready presentation deck and in-repo presentation package",
                score=88,
                execution_type="engineering",
                suggested_task_type="implement",
                rationale="A presentation package reduces Chair and viva risk while remaining entirely repo-editable.",
                depends_on=[],
                supporting=presentation_findings,
            )

        if showreel_findings or chapter_signposting_findings:
            add_task(
                title="Create a showreel script, shot list, capture checklist, and Chapter 7 signposting updates",
                score=84,
                execution_type="engineering",
                suggested_task_type="implement",
                rationale="Prepare every repo-editable part of the showreel workflow before asking for the human recording step.",
                depends_on=[],
                supporting=showreel_findings + chapter_signposting_findings,
            )

        if word_count_findings or presentation_findings or showreel_findings:
            add_task(
                title="Re-run GRADER and refresh board-lens analysis after engineering remediation lands",
                score=72,
                execution_type="engineering",
                suggested_task_type="implement",
                rationale="A final engineering validation pass should confirm whether board-facing risks actually moved.",
                depends_on=[
                    "Trim dissertation body to a submission-safe word-count band and update the declared count honestly",
                    "Create a submission-ready presentation deck and in-repo presentation package",
                    "Create a showreel script, shot list, capture checklist, and Chapter 7 signposting updates",
                ],
                supporting=word_count_findings + presentation_findings + showreel_findings,
            )

        if showreel_findings:
            add_task(
                title="Record and export the final showreel/demo artefact",
                score=65,
                execution_type="human_dependent",
                suggested_task_type="plan",
                rationale="The recording itself needs a human operator, so it follows the engineering preparation rather than displacing it.",
                depends_on=[
                    "Create a showreel script, shot list, capture checklist, and Chapter 7 signposting updates",
                ],
                supporting=showreel_findings,
            )

        if conflict_findings:
            add_task(
                title="Clarify the official word-count ruling with the supervisor or module lead and record the decision in-repo",
                score=60,
                execution_type="human_dependent",
                suggested_task_type="plan",
                rationale="The clarification matters, but it stays below engineering work that can already reduce risk immediately.",
                depends_on=[
                    "Trim dissertation body to a submission-safe word-count band and update the declared count honestly",
                ],
                supporting=conflict_findings,
            )

        priorities.sort(
            key=lambda item: (
                0 if item.execution_type == "engineering" else 1,
                -item.priority_score,
                item.title,
            )
        )
        return priorities


def render_markdown(result: GraderResult) -> str:
    critical = [item for item in result.findings if item.severity == "critical"]
    high = [item for item in result.findings if item.severity == "high"]
    lines = [
        "# GRADER Report",
        "",
        f"_Generated: {result.run_at}_",
        "",
        "## Board Summary",
        f"- Overall score: **{result.overall_score:.1f}/100**",
        f"- Academic quality score: **{result.academic_quality_score:.1f}**",
        f"- Compliance score: **{result.compliance_score:.1f}**",
        f"- Estimated classification: **{result.estimated_classification}**",
        f"- Board verdict: **{result.board_verdict}**",
        f"- Submission readiness: **{result.submission_readiness}**",
        "",
        "## Board Lenses",
    ]
    for lens in result.board_lenses:
        lines.append(f"- {lens.name}: **{lens.score:.1f}/100** ({lens.classification})")
        lines.append(f"  {lens.emphasis}")
        lines.append(f"  Verdict: {lens.verdict}")

    lines.extend([
        "",
        "## Key Risks",
    ])
    if critical or high:
        for item in critical + high[:6]:
            lines.append(f"- [{item.severity.upper()}] {item.title}: {item.detail}")
    else:
        lines.append("- No critical or high-severity issues were detected.")

    lines.extend(["", "## Dimension Scores"])
    for dimension in result.dimension_scores:
        lines.append(f"- {dimension.name}: **{dimension.score:.1f}/{dimension.weight}**")
        lines.append(f"  {dimension.rationale}")

    lines.extend(["", "## Task Priority Queue"])
    if result.task_priorities:
        lines.append("- Engineering tasks are always ranked ahead of human-dependent tasks.")
        for item in result.task_priorities:
            lines.append(
                f"- [{item.priority_score:>3}] {item.execution_type} / {item.suggested_task_type}: {item.title}"
            )
            lines.append(f"  Rationale: {item.rationale}")
            if item.depends_on:
                lines.append(f"  Depends on: {', '.join(item.depends_on)}")
            if item.supporting_findings:
                lines.append(f"  Supporting findings: {', '.join(item.supporting_findings)}")
    else:
        lines.append("- No remediation task priorities were generated.")

    lines.extend(["", "## Lens Findings"])
    for lens in result.board_lenses:
        lines.append(f"### {lens.name}")
        lines.append(f"- Score: **{lens.score:.1f}/100**")
        lines.append(f"- Verdict: **{lens.verdict}**")
        if lens.key_findings:
            for item in lens.key_findings:
                lines.append(f"- [{item.severity.upper()}] {item.title}: {item.detail}")
        else:
            lines.append("- No major concerns surfaced for this lens.")

    stats = result.dissertation_stats
    lines.extend(
        [
            "",
            "## Dissertation Stats",
            f"- Body word count (Ch 1-11): {stats['body_word_count']}",
            f"- Abstract word count: {stats['abstract_word_count']}",
            f"- Declared word count: {stats['declared_word_count']}",
            f"- Bibliography entries detected: {stats['bibliography_entries']}",
            f"- Mean chapter word count: {mean(item['word_count'] for item in stats['chapters'].values()):.1f}",
            "",
            "## All Findings",
        ]
    )
    for item in result.findings:
        lines.append(f"- [{item.severity.upper()}] {item.category}: {item.title} - {item.detail}")
        if item.evidence:
            lines.append(f"  Evidence: {', '.join(item.evidence)}")

    lines.extend(["", "## Sources", "### Local"])
    for item in result.evidence_sources["local"]:
        lines.append(f"- {item}")
    lines.append("### Official")
    for item in result.evidence_sources["official"]:
        lines.append(f"- {item['name']}: {item['url']}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade the current SCAFAD submission package.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Workspace root containing project/scafad-r-core",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "grader_report.json",
        help="Path to write the machine-readable grading output.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "grader_report.md",
        help="Path to write the examiner-style markdown report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grader = ScafadGrader(args.repo_root)
    result = grader.grade()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
