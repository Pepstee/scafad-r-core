"""
Root conftest — extend sys.path so scafad/runtime/runtime.py intra-package
imports (e.g. ``from layer0.telemetry import ...``) resolve when tests are
collected from the repo-root tests/ directory (DL-039).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SCAFAD_PKG = _REPO_ROOT / "scafad"

for _p in (_REPO_ROOT, _SCAFAD_PKG):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
