from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Execute asyncio-marked coroutine tests without pytest-asyncio.

    Some of the historical tests rely on ``@pytest.mark.asyncio`` while others
    simply define ``async def`` bodies.  Pytest normally delegates to
    ``pytest-asyncio`` in these scenarios, but the plugin is not available in
    our execution environment.  This hook reproduces the minimal behaviour we
    need by spawning a temporary event loop and executing the coroutine in a
    synchronous context.
    """

    test_func = pyfuncitem.obj
    marker = pyfuncitem.get_closest_marker("asyncio")

    if not inspect.iscoroutinefunction(test_func):
        # Respect the historical marker behaviour so that non-async tests keep
        # using the default pytest execution path.
        if marker is None:
            return None
        return None

    loop = asyncio.new_event_loop()
    try:
        previous_loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous_loop = None

    try:
        asyncio.set_event_loop(loop)

        signature = inspect.signature(test_func)
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        call_kwargs = {}
        for name, value in pyfuncitem.funcargs.items():
            if name in signature.parameters or accepts_var_kwargs:
                call_kwargs[name] = value

        loop.run_until_complete(test_func(**call_kwargs))
    finally:
        asyncio.set_event_loop(previous_loop)
        loop.close()

    return True
