#\!/usr/bin/env python3
"""
class_boundary_check.py — SCAFAD class-boundary invariant checker (WP-1.7, I-10)

Verifies that every (file, class, method) tuple in INVENTORY is present in the
codebase at the specified location.  Exits 0 on success, 1 on any violation.

Usage:
    python3 scripts/class_boundary_check.py [--root <path>]

Add new entries to INVENTORY as the project grows (see WP-1.7, WP-2.7).
"""
from __future__ import annotations

import ast
import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# INVENTORY — (file_path, class_name, method_name)
# Extend at each phase.  Never remove entries; add a REMOVED comment instead.
# ---------------------------------------------------------------------------
INVENTORY: list[tuple[str, str, str]] = [
    # I-10 / WP-1.3: _emit_with_queue must be inside MultiChannelTelemetry
    ("app_telemetry.py", "MultiChannelTelemetry", "_emit_with_queue"),
    # I-10: _emit_immediate must be inside MultiChannelTelemetry
    ("app_telemetry.py", "MultiChannelTelemetry", "_emit_immediate"),
    # Adversarial engine interface
    ("app_adversarial.py", "AdversarialAnomalyEngine", "get_attack_effectiveness_report"),
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
                f"MISSING FILE  {rel_path}  (expected class {class_name}, method {method_name})"
            )
            continue

        if rel_path not in cache:
            try:
                cache[rel_path] = ast.parse(full_path.read_text(encoding="utf-8"))
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
            print(f"  ✗  {v}")
        return 1

    print(f"class_boundary_check OK — {len(INVENTORY)} invariant(s) verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
