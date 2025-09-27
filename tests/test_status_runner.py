"""Pytest wrapper around the quick status runner script."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .run_quick_test_check import main as run_quick_status  # noqa: E402


def test_quick_status_runner_passes(capsys) -> None:
    """Ensure the quick status runner reports success."""
    success = run_quick_status()
    captured = capsys.readouterr()
    assert success, (
        "Quick status runner reported failure.\n"
        f"stdout:\n{captured.out}\n"
        f"stderr:\n{captured.err}"
    )
