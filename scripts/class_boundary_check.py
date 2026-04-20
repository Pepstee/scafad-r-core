#\!/usr/bin/env python3
"""
class_boundary_check.py -- SCAFAD class-boundary invariant checker (WP-1.7, I-10)

Verifies that every (file, class, method) tuple in INVENTORY is present in the
codebase at the specified location.  Exits 0 on success, 1 on any violation.

Usage:
    python3 scripts/class_boundary_check.py [--root <path>]

Add new entries to INVENTORY as the project grows (see WP-1.7, WP-2.7).
Never remove entries -- add a REMOVED comment instead.
"""
from __future__ import annotations

import ast
import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# INVENTORY -- (file_path, class_name, method_name)
# Extend at each phase.  Never remove entries.
# ---------------------------------------------------------------------------
INVENTORY: list[tuple[str, str, str]] = [
    # ------------------------------------------------------------------
    # Phase 1 / WP-1.3 / I-10: telemetry emission class boundaries
    # ------------------------------------------------------------------
    # _emit_with_queue must live inside MultiChannelTelemetry, not
    # FallbackTelemetryProcessor.  Enforces the class-boundary fix of WP-1.3.
    ("app_telemetry.py", "MultiChannelTelemetry", "_emit_with_queue"),
    # _emit_immediate is also a MultiChannelTelemetry method (routing target).
    ("app_telemetry.py", "MultiChannelTelemetry", "_emit_immediate"),
    # emit_telemetry is the public dispatch entry point on MultiChannelTelemetry.
    ("app_telemetry.py", "MultiChannelTelemetry", "emit_telemetry"),
    # ------------------------------------------------------------------
    # Phase 1 / WP-1.1 / I-1: adversarial engine interface
    # ------------------------------------------------------------------
    ("app_adversarial.py", "AdversarialAnomalyEngine", "get_attack_effectiveness_report"),
    # ------------------------------------------------------------------
    # Layer 0->1 adapter (I-2): adapt() must stay in RCoreToLayer1Adapter
    # ------------------------------------------------------------------
    ("layers/layer1/adapter.py", "RCoreToLayer1Adapter", "adapt"),
    # ------------------------------------------------------------------
    # Layer 1 canonical pipeline
    # ------------------------------------------------------------------
    ("layers/layer1/pipeline.py", "Layer1CanonicalPipeline", "process_l0_record"),
    ("layers/layer1/pipeline.py", "Layer1CanonicalPipeline", "process_adapted_record"),
    ("layers/layer1/pipeline.py", "Layer1ProcessedRecord", "to_dict"),
    # ------------------------------------------------------------------
    # Canonical runtime (I-1): process_event is the single Lambda entry
    # ------------------------------------------------------------------
    ("layers/runtime.py", "SCAFADCanonicalRuntime", "process_event"),
    ("layers/runtime.py", "CanonicalRuntimeResult", "to_dict"),
    # ------------------------------------------------------------------
    # L0 TelemetryRecord contract (I-15, WP-2.9)
    # ------------------------------------------------------------------
    ("app_telemetry.py", "TelemetryRecord", "to_dict"),
    # ------------------------------------------------------------------
    # Phase 2 / WP-2.7: permanent test class boundaries (T-010, T-011)
    # Ensures test classes are not accidentally renamed or split.
    # ------------------------------------------------------------------
    (
        "tests/test_lambda_handler_routes_canonical.py",
        "TestLambdaHandlerRoutesCanonical",
        "test_runtime_constructed_and_process_event_called",
    ),
    (
        "tests/test_app_telemetry_queue_emission.py",
        "TestEmitWithQueueAST",
        "test_emit_with_queue_is_method_of_multichanneltelemetry",
    ),
]


def _methods_in_class(tree: ast.Module, class_name: str) -> set[str]:
    """Return the set of method names defined directly inside *class_name*."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in ast.walk(node)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def check(root: Path) -> list[str]:
    """Return a list of violation messages (empty = clean)."""
    violations: list[str] = []
    cache: dict[str, ast.Module] = {}

    for rel_path, class_name, method_name in INVENTORY:
        full_path = root / rel_path
        if not full_path.exists():
            violations.append(
                f"MISSING FILE  {rel_path}  "
                f"(expected class {class_name}, method {method_name})"
            )
            continue

        if rel_path not in cache:
            try:
                source = full_path.read_bytes().rstrip(b"\x00").decode(
                    "utf-8", errors="replace"
                )
                cache[rel_path] = ast.parse(source)
            except SyntaxError as exc:
                violations.append(f"SYNTAX ERROR  {rel_path}: {exc}")
                continue

        methods = _methods_in_class(cache[rel_path], class_name)
        if not methods:
            violations.append(
                f"MISSING CLASS  {rel_path}::{class_name}  "
                f"(method {method_name} not checked)"
            )
        elif method_name not in methods:
            violations.append(
                f"BOUNDARY VIOLATION  {rel_path}::{class_name}.{method_name}  "
                f"not found in class"
            )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).parent.parent),
        help="Root of scafad-r-core (default: parent of scripts/)",
    )
    args = parser.parse_args()
    root = Path(args.root)

    violations = check(root)
    if violations:
        print("class_boundary_check FAILED:")
        for v in violations:
            print(f"  x  {v}")
        return 1

    print(
        f"class_boundary_check OK -- {len(INVENTORY)} invariant(s) verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
