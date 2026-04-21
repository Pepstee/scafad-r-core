"""
Root conftest — path bootstrap + namespace-alias hook to prevent dual module
identity (DL-040).

Background
----------
``pytest.ini`` adds both the repo root *and* ``scafad/`` to ``sys.path``
(``pythonpath = . scafad``).  That means Python can reach the same source file
via two distinct module keys, e.g.::

    layer0.app_telemetry          (when scafad/ is the root)
    scafad.layer0.app_telemetry   (when repo root is the root)

When both paths are exercised in a single pytest session the runtime's
``isinstance(record, TelemetryRecord)`` check fails because the two keys map to
*different* class objects — the classic "same file, two identities" pitfall.

Fix
---
``_ScafadNamespaceAlias`` is a ``sys.meta_path`` finder/loader that intercepts
any import of the form ``scafad.layerX[.submod]`` and ensures it resolves to
the *bare* module object already in ``sys.modules`` under ``layerX[.submod]``.
The bare namespace is authoritative (it is what the runtime and all layer tests
use internally); the ``scafad.*`` namespace becomes a transparent alias
(DL-040).
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent
_SCAFAD_PKG = _REPO_ROOT / "scafad"

for _p in (_REPO_ROOT, _SCAFAD_PKG):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


# ---------------------------------------------------------------------------
# Namespace alias hook
# ---------------------------------------------------------------------------

class _AliasLoader(importlib.abc.Loader):
    """Trivial loader that returns an already-initialised module unchanged."""

    def __init__(self, module: object) -> None:
        self._module = module

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> object:  # noqa: ARG002
        return self._module

    def exec_module(self, module: object) -> None:  # noqa: ARG002
        # The module is the bare module — already fully initialised.
        pass


class _ScafadNamespaceAlias(importlib.abc.MetaPathFinder):
    """Redirect ``scafad.layerX[.submod]`` imports to their bare equivalents.

    This ensures that ``from scafad.layer0 import TelemetryRecord`` and
    ``from layer0.app_telemetry import TelemetryRecord`` resolve to the same
    class object, fixing the dual-identity ``isinstance`` failure (DL-040).
    """

    _PREFIX = "scafad."
    _LAYERS = frozenset(
        ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "runtime"]
    )

    def find_spec(
        self,
        fullname: str,
        path: object,  # noqa: ARG002
        target: object = None,  # noqa: ARG002
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(self._PREFIX):
            return None

        rest = fullname[len(self._PREFIX):]   # e.g. "layer0" or "layer0.app_telemetry"
        top_layer = rest.split(".")[0]
        if top_layer not in self._LAYERS:
            return None

        # Already aliased — nothing to do; let sys.modules cache serve it.
        if fullname in sys.modules:
            return None

        # Ensure the bare module is loaded.
        if rest not in sys.modules:
            try:
                importlib.import_module(rest)
            except ImportError:
                return None

        bare_mod = sys.modules.get(rest)
        if bare_mod is None:
            return None

        # Register the alias so subsequent lookups bypass this hook entirely.
        sys.modules[fullname] = bare_mod

        spec = importlib.machinery.ModuleSpec(fullname, _AliasLoader(bare_mod))
        # Preserve subpackage search path so dotted sub-imports also work.
        spec.submodule_search_locations = getattr(bare_mod, "__path__", None)
        return spec


# Insert once, before the standard finders.
if not any(isinstance(f, _ScafadNamespaceAlias) for f in sys.meta_path):
    sys.meta_path.insert(0, _ScafadNamespaceAlias())
