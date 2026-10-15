# coding: utf-8
"""Progress for a long run: a host that asked for it hears about each episode or browser step, nobody else does.

The tool series and the browser decision model call ``report``; it is a no-op unless an entry point installed a
reporter with ``reporting``. ``s1a-mcp`` installs one when the request carries a ``progressToken``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

Reporter = Callable[[float, float | None, str], Awaitable[None]]

log = logging.getLogger(__name__)
_REPORTER: ContextVar[Reporter | None] = ContextVar("s1a_progress_reporter", default=None)


@contextmanager
def reporting(reporter: Reporter) -> Iterator[None]:
    """Route ``report`` calls made in this context (and the tasks it starts) to ``reporter``."""
    token = _REPORTER.set(reporter)
    try:
        yield
    finally:
        _REPORTER.reset(token)


async def report(done: float, total: float | None, message: str) -> None:
    """One progress point: ``done`` of ``total`` (None when unknown) with a short message such as ``episode 3/20``.

    A reporter that fails is logged and dropped for this point; progress never ends a run."""
    reporter = _REPORTER.get()
    if reporter is None:
        return
    try:
        await reporter(done, total, message)
    except Exception as exc:  # a closed transport or a host that went away must not fail the run itself
        log.warning("progress report %s failed: %s", message, exc)
