# coding: utf-8
"""The injection guard as a BrowseSafe++ detector: ``predict(request) -> browsesafe_pp.Prediction``.

BrowseSafe++ ships an SDK (``browsesafe-pp``, from GitHub, not on PyPI) whose ``evaluate`` and ``predict`` commands
call a detector once per page. This module is that detector for ``injection_guard`` (#119):

- the page (``request.content``) is clipped to the runtime's ``MAX_CHARS`` (6,000 characters), as the guard clips a
  tool result, and sent as ``{"tool": ..., "text": ...}``: ``browser_snapshot`` for the accessibility view,
  ``fetch_webpage`` for html and markdown;
- act band (p >= 0.7) is ``injected=True``, allow band (p <= 0.3) is ``injected=False``, and the uncertain band is
  ``injected=None``, which the SDK counts as an abstention. The runtime passes an uncertain page whole, so treating
  it as "not injected" is the other defensible mapping; we abstain so the SDK's report shows the band separately;
- a failed decision raises, so the SDK counts it as an error. ``S1A_BSPP_ON_FAILURE=closed`` instead answers
  ``injected=True``, as the runtime guard quarantines on failure. The raised message names the failure's category
  only, never its text: an endpoint's error body can quote the page back.

The SDK is never imported by core code or core CI; it is imported only when ``predict`` builds a prediction::

    uv run --with "browsesafe-pp[eval] @ git+https://github.com/perplexityai/browsesafe-plus-plus.git@<sha>" \\
        browsesafe evaluate --predict s1a.browsesafe_pp:predict --view markdown --output results/s1a-markdown

``S1A_BSPP_MODEL`` picks the decision backend (default ``jev``; any ``s1a run injection_guard --model`` name).
"""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from collections.abc import Callable
from typing import Any

from openjiuwen.core.common.exception.errors import BaseError

from s1a import rails
from s1a.agents.injection_guard import MAX_CHARS, SPEC
from s1a.decision_models import DecisionModel, build_model
from s1a.spec import Json

VIEW_TOOL = {"html": "fetch_webpage", "markdown": "fetch_webpage", "accessibility": "browser_snapshot"}
BAND_INJECTED: dict[str, bool | None] = {"act": True, "allow": False, "uncertain": None}
ON_FAILURE = ("error", "closed")


class DecisionFailed(RuntimeError):
    """A decision that failed; the SDK records the page as an error. Carries the failure category, not its text."""


def state_of(request: Any) -> Json:
    """The guard's state for one SDK request: the view's tool name and the page cut to ``MAX_CHARS``."""
    view = getattr(request, "view", None)
    if view not in VIEW_TOOL:
        raise ValueError(f"unknown BrowseSafe++ view {view!r}; one of {sorted(VIEW_TOOL)}")
    content = getattr(request, "content", None)
    if not isinstance(content, str):
        raise TypeError(f"request.content is {type(content).__name__}, not str")
    return {"tool": VIEW_TOOL[view], "text": content[:MAX_CHARS]}


def injected_of(band: str) -> bool | None:
    """The SDK's ``injected`` for a band: act True, allow False, uncertain None (abstain)."""
    if band not in BAND_INJECTED:
        raise ValueError(f"unknown band {band!r}")
    return BAND_INJECTED[band]


def sdk_prediction(injected: bool | None) -> Any:
    import browsesafe_pp as bs  # optional: only where the SDK is installed

    return bs.Prediction(injected=injected)


class Detector:
    """One decision backend on its own event loop, called synchronously once per page (the SDK's detector shape)."""

    def __init__(
        self,
        decision_model: DecisionModel,
        *,
        on_failure: str = "error",
        prediction: Callable[[bool | None], Any] | None = None,
    ) -> None:
        if on_failure not in ON_FAILURE:
            raise ValueError(f"on_failure must be one of {ON_FAILURE}, not {on_failure!r}")
        self.decision_model = decision_model
        self.on_failure = on_failure
        self._prediction = prediction  # None: the SDK's Prediction, imported on first use
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()  # one loop, one call at a time, if the caller uses threads
        self._warm = False
        self.calls = 0
        self.bands: dict[str, int] = {"allow": 0, "uncertain": 0, "act": 0, "failure": 0}

    def decide(self, request: Any) -> bool | None:
        """``injected`` for one page; raises ``DecisionFailed`` on a failed decision unless failing closed."""
        state = state_of(request)
        with self._lock:
            self.calls += 1
            try:
                if not self._warm:
                    self._loop.run_until_complete(self.decision_model.warm())
                    self._warm = True
                verdict = self._loop.run_until_complete(rails.ask(SPEC, state, self.decision_model))
            except BaseError as exc:
                self.bands["failure"] += 1
                if self.on_failure == "closed":
                    return True
                raise DecisionFailed(rails.failure_category(exc)) from None
            self.bands[verdict.band] += 1
            return injected_of(verdict.band)

    def __call__(self, request: Any) -> Any:
        injected = self.decide(request)
        return (self._prediction or sdk_prediction)(injected)

    def close(self) -> None:
        with self._lock:
            if self._loop.is_closed():
                return
            try:
                self._loop.run_until_complete(self.decision_model.close())
            finally:
                self._loop.close()


_detector: Detector | None = None
_detector_lock = threading.Lock()


def detector() -> Detector:
    """The process's detector, built on first use from ``S1A_BSPP_MODEL`` and ``S1A_BSPP_ON_FAILURE``."""
    global _detector
    with _detector_lock:
        if _detector is None:
            model = os.getenv("S1A_BSPP_MODEL", "jev")
            if model not in rails.RAIL_MODEL_NAMES:
                raise ValueError(f"S1A_BSPP_MODEL={model!r}; one of {rails.RAIL_MODEL_NAMES}")
            _detector = Detector(build_model(model), on_failure=os.getenv("S1A_BSPP_ON_FAILURE", "error"))
            atexit.register(_detector.close)
        return _detector


def predict(request: Any) -> Any:
    """The SDK entry point: ``browsesafe evaluate --predict s1a.browsesafe_pp:predict``."""
    return detector()(request)
