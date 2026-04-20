#\!/usr/bin/env python3
"""
audit_imports.py -- SCAFAD Import Auditor
==========================================

WP-2.2 / Blueprint Appendix A.4

Walks every .py file in scafad-r-core (excluding legacy/, __pycache__/,
pytest-cache directories, and the scripts/ dir itself), parses the AST,
extracts all top-level third-party import names, and checks each against:

  1. The Python standard library.
  2. The declared packages in requirements.txt (with install-name to import-name
     mapping to handle common mismatches such as PyYAML to yaml).
  3. Packages that are guarded by try/except ImportError, or inside
     "if TYPE_CHECKING:" / "if <AVAILABILITY_FLAG>:" blocks.
  4. Names that resolve to local modules or directories in the repo root.

Exits 0 if no undeclared hard imports are found.
Exits 1 and prints a report otherwise.

Usage
-----
    python3 scripts/audit_imports.py [--root <path>]
"""

from __future__ import annotations

import ast
import argparse
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, NamedTuple, Set, Tuple


# ---------------------------------------------------------------------------
# Mapping: requirement / pip install name -> top-level importable name(s)
# ---------------------------------------------------------------------------
_INSTALL_TO_IMPORT: Dict[str, List[str]] = {
    "scikit-learn":       ["sklearn"],
    "pyyaml":             ["yaml"],
    "PyYAML":             ["yaml"],
    "boto3":              ["boto3"],
    "botocore":           ["botocore"],
    "aiofiles":           ["aiofiles"],
    "typing-extensions":  ["typing_extensions"],
    "email-validator":    ["email_validator"],
    "pip-tools":          [],
    "pip_tools":          [],
    "pytest-asyncio":     ["pytest_asyncio"],
    "torch":              ["torch"],
    "torch-geometric":    ["torch_geometric"],
    "pyg":                ["torch_geometric"],
    "packaging":          ["packaging"],
    "networkx":           ["networkx"],
    "numpy":              ["numpy", "np"],
    "pandas":             ["pandas", "pd"],
    "scipy":              ["scipy"],
    "matplotlib":         ["matplotlib"],
    "seaborn":            ["seaborn"],
    "psutil":             ["psutil"],
    "cerberus":           ["cerberus"],
    "validators":         ["validators"],
    "semver":             ["semver"],
    "jsonpatch":          ["jsonpatch"],
    "jsonpointer":        ["jsonpointer"],
    "brotli":             ["brotli"],
    "lz4":                ["lz4"],
    "python-snappy":      ["snappy"],
    "snappy":             ["snappy"],
    "polyglot":           ["polyglot"],
    "numba":              ["numba"],
    "cryptography":       ["cryptography"],
    "blake3":             ["blake3"],
    "jsonschema":         ["jsonschema"],
    "aiohttp":            ["aiohttp"],
    "aws-xray-sdk":       ["aws_xray_sdk"],
    "aws_xray_sdk":       ["aws_xray_sdk"],
    "awslambdaric":       ["awslambdaric"],
    "statsmodels":        ["statsmodels"],
    "ruff":               [],
    "black":              [],
    "isort":              [],
    "pytest":             ["pytest"],
}

# ---------------------------------------------------------------------------
# Directories to skip.
# ---------------------------------------------------------------------------
_SKIP_DIRS: FrozenSet[str] = frozenset({
    "legacy",
    "__pycache__",
    ".git",
    ".github",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
})
_SKIP_PREFIXES = ("pytest-cache-files-",)

# ---------------------------------------------------------------------------
# Conditional guard variable names that make an import "soft" at runtime.
# Imports inside "if <NAME>:" are treated as guarded.
# ---------------------------------------------------------------------------
_CONDITIONAL_GUARD_NAMES: FrozenSet[str] = frozenset({
    "TYPE_CHECKING",       # typing.TYPE_CHECKING -- never true at runtime
    "_DELTA_AVAILABLE",    # scafad-delta availability flag (test_006/007)
    "_TORCH_AVAILABLE",
    "TORCH_AVAILABLE",
    "HAS_TORCH",
    "HAS_SCIPY",
    "HAS_NUMBA",
})


def _stdlib_names() -> FrozenSet[str]:
    """Return the set of top-level standard-library module names."""
    try:
        stdlib: Set[str] = set(sys.stdlib_module_names)  # Python 3.10+
    except AttributeError:
        stdlib = set()
    stdlib.update({
        "abc", "ast", "asyncio", "base64", "binascii", "builtins", "cmath",
        "collections", "concurrent", "contextlib", "copy", "csv", "dataclasses",
        "datetime", "decimal", "email", "enum", "functools", "gc", "glob",
        "gzip", "hashlib", "heapq", "hmac", "html", "http", "importlib",
        "inspect", "io", "itertools", "json", "keyword", "linecache", "locale",
        "logging", "math", "mimetypes", "multiprocessing", "numbers", "operator",
        "os", "pathlib", "pickle", "platform", "pprint", "queue", "random",
        "re", "select", "shutil", "signal", "socket", "sqlite3", "ssl",
        "stat", "statistics", "string", "struct", "subprocess", "sys",
        "tempfile", "textwrap", "threading", "time", "timeit", "traceback",
        "types", "typing", "unicodedata", "unittest", "urllib", "uuid",
        "warnings", "weakref", "xml", "zipfile", "zipimport", "zlib",
        "_thread", "__future__", "argparse", "configparser", "ctypes",
        "curses", "dbm", "difflib", "dis", "doctest", "fileinput",
        "fractions", "ftplib", "grp", "imaplib", "ipaddress", "lzma",
        "mailbox", "mmap", "netrc", "pty", "pwd", "readline",
        "rlcompleter", "sched", "secrets", "shelve", "smtplib",
        "spwd", "sunau", "symtable", "syslog", "tarfile", "telnetlib",
        "termios", "token", "tokenize", "tty", "turtle",
        "uu", "venv", "wave", "xdrlib", "xmlrpc",
    })
    return frozenset(stdlib)


def _declared_import_names(requirements_path: Path) -> FrozenSet[str]:
    """Parse requirements.txt and return the set of importable names."""
    names: Set[str] = set()
    if not requirements_path.exists():
        return frozenset()
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pkg = line.split("#")[0].strip()
        for ch in (">=", "<=", "==", "\!=", "~=", ">", "<", "["):
            pkg = pkg.split(ch)[0].strip()
        pkg_lower = pkg.lower()
        matched = False
        for key, import_names in _INSTALL_TO_IMPORT.items():
            if key.lower() == pkg_lower:
                names.update(import_names)
                matched = True
                break
        if not matched:
            names.add(pkg_lower.replace("-", "_"))
            names.add(pkg_lower)
    return frozenset(names)


class ImportFinding(NamedTuple):
    file: Path
    lineno: int
    name: str
    is_guarded: bool


def _build_parent_map(tree: ast.Module) -> Dict[int, ast.AST]:
    parent: Dict[int, ast.AST] = {}
    for n in ast.walk(tree):
        for child in ast.iter_child_nodes(n):
            parent[id(child)] = n
    return parent


def _is_guarded(node: ast.AST, parent: Dict[int, ast.AST]) -> bool:
    """Return True if *node* is in a try/except ImportError or conditional guard.

    Guards recognised:
      - ast.Try whose handlers catch ImportError / ModuleNotFoundError / Exception
      - ast.If whose test is ast.Name with id in _CONDITIONAL_GUARD_NAMES
    """
    current = parent.get(id(node))
    while current is not None:
        # --- try/except guard ---
        if isinstance(current, ast.Try):
            for handler in current.handlers:
                exc = handler.type
                if exc is None:
                    return True  # bare except
                exc_names: List[str] = []
                if isinstance(exc, ast.Name):
                    exc_names.append(exc.id)
                elif isinstance(exc, ast.Tuple):
                    for elt in exc.elts:
                        if isinstance(elt, ast.Name):
                            exc_names.append(elt.id)
                if any(
                    n in ("ImportError", "ModuleNotFoundError", "Exception")
                    for n in exc_names
                ):
                    return True
        # --- conditional guard (TYPE_CHECKING, availability flags) ---
        if isinstance(current, ast.If):
            test = current.test
            guard_name: str = ""
            if isinstance(test, ast.Name):
                guard_name = test.id
            elif isinstance(test, ast.Attribute):
                guard_name = test.attr
            if guard_name in _CONDITIONAL_GUARD_NAMES:
                return True
        current = parent.get(id(current))
    return False


def _collect_imports(py_file: Path) -> List[ImportFinding]:
    try:
        source = py_file.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        return []

    parent_map = _build_parent_map(tree)
    findings: List[ImportFinding] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                guarded = _is_guarded(node, parent_map)
                findings.append(ImportFinding(py_file, node.lineno, top, guarded))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module is None:
                continue
            top = node.module.split(".")[0]
            guarded = _is_guarded(node, parent_map)
            findings.append(ImportFinding(py_file, node.lineno, top, guarded))

    return findings


def _local_module_names(root: Path) -> FrozenSet[str]:
    """Return top-level .py stems and directory names (namespace packages included)."""
    names: Set[str] = set()
    _SKIP = {".git", ".github", "__pycache__", "legacy", "node_modules"}
    for item in root.iterdir():
        if item.is_dir() and item.name not in _SKIP and not item.name.startswith("."):
            names.add(item.name)
        elif item.is_file() and item.suffix == ".py":
            names.add(item.stem)
    return frozenset(names)


def audit(root: Path) -> Tuple[List[ImportFinding], List[ImportFinding]]:
    """Return (undeclared_hard, undeclared_guarded)."""
    stdlib = _stdlib_names()
    declared = _declared_import_names(root / "requirements.txt")
    local = _local_module_names(root)

    known = stdlib | declared | local | {
        # Delta modules loaded dynamically in test_006 / test_007
        "layer1_core", "layer1_config",
        # Enterprise test runner imports (filename as module)
        "test_mitre_attack_coverage", "test_operational_readiness",
        "test_performance_benchmarks", "test_security_penetration",
        "test_soc_compliance", "test_adversarial",
        # Dev / build tools
        "_pytest", "pytest_asyncio", "pip",
    }

    undeclared_hard: List[ImportFinding] = []
    undeclared_guarded: List[ImportFinding] = []

    for py_file in sorted(root.rglob("*.py")):
        rel_parts = py_file.relative_to(root).parts
        skip = False
        for part in rel_parts:
            if part in _SKIP_DIRS:
                skip = True
                break
            if any(part.startswith(pfx) for pfx in _SKIP_PREFIXES):
                skip = True
                break
        if skip:
            continue

        for finding in _collect_imports(py_file):
            if finding.name in known:
                continue
            if finding.is_guarded:
                undeclared_guarded.append(finding)
            else:
                undeclared_hard.append(finding)

    return undeclared_hard, undeclared_guarded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit third-party imports vs requirements.txt"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/)",
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()

    print("Auditing imports in:", root)
    hard, guarded = audit(root)

    if guarded:
        print(
            f"\n[INFO] {len(guarded)} guarded import(s) not in requirements.txt "
            "(try/except-protected or conditional -- informational only):"
        )
        seen: Set[Tuple[str, str]] = set()
        for g in sorted(guarded, key=lambda x: (x.name, str(x.file))):
            key = (g.name, str(g.file.relative_to(root)))
            if key not in seen:
                seen.add(key)
                print(f"  {g.name:30s}  {g.file.relative_to(root)}")

    if hard:
        print(
            f"\n[FAIL] {len(hard)} undeclared hard import(s) found "
            "(add to requirements.txt or wrap in try/except):"
        )
        seen2: Set[Tuple[str, str]] = set()
        for h in sorted(hard, key=lambda x: (x.name, str(x.file))):
            key2 = (h.name, str(h.file.relative_to(root)))
            if key2 not in seen2:
                seen2.add(key2)
                print(f"  {h.name:30s}  {h.file.relative_to(root)}")
        return 1

    print("\n[PASS] All hard imports are declared in requirements.txt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
