# coding: utf-8
"""One native window as an ``Env``: offer only actions grounded in its current snapshot."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image as PILImage

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.types import Box, BoxQuestion, Image, NoulQuestion, Observation, Question
from s1a.desktop.driver import Capture, Driver, DriverError, Element, Snapshot, Window

DONE = "done"
ABSTAIN = "abstain"
RESERVED = {
    ABSTAIN: "No offered action moves the task forward; stop without acting.",
}
_CLICKABLE_ROLES = ("button", "checkbox", "radiobutton", "menubutton", "link", "popupbutton", "disclosuretriangle")
_EDITABLE_ROLES = ("textfield", "textarea", "textview", "searchfield", "combobox", "editabletext")
_MENU_ROLES = frozenset({"menu", "menubar", "menubaritem", "menuitem"})

# Uncalibrated defaults for --box-target gating (#98); tune on your own captures (see scripts/measure_box_targets.py).
DEFAULT_BOX_PRESENCE_THRESHOLD = 0.5
DEFAULT_BOX_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class BoxHit:
    """One box target that passed presence and confidence gates for the current capture."""

    key: str
    description: str
    box: Box
    presence: float
    capture_id: str

    @property
    def centre(self) -> tuple[float, float]:
        return self.box.centre

    def record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "box": {"top": self.box.top, "bottom": self.box.bottom, "left": self.box.left, "right": self.box.right},
            "presence": self.presence,
            "box_confidence": self.box.confidence,
            "capture_id": self.capture_id,
            "centre": list(self.centre),
        }


def clickable(element: Element) -> bool:
    role = element.role.casefold().removeprefix("ax")
    if not element.label.strip() or role in _MENU_ROLES:
        return False
    return (
        element.enabled
        and element.token is not None
        and (
            role in _CLICKABLE_ROLES or any("press" in a.casefold() or "click" in a.casefold() for a in element.actions)
        )
    )


def editable(element: Element) -> bool:
    return (
        element.enabled and element.token is not None and element.role.casefold().removeprefix("ax") in _EDITABLE_ROLES
    )


def observable(element: Element) -> bool:
    """Keep the named controls and visible values; omit macOS menu trees from the decision state."""
    role = element.role.casefold().removeprefix("ax")
    return role not in _MENU_ROLES and bool(element.label.strip() or element.value.strip())


def visual_progress(capture: Capture) -> dict[str, Any]:
    """Hash decoded pixels so capture IDs and PNG metadata cannot count as progress."""
    with PILImage.open(io.BytesIO(capture.image.data)) as picture:
        rgba = picture.convert("RGBA")
        return {"width": rgba.width, "height": rgba.height, "sha256": hashlib.sha256(rgba.tobytes()).hexdigest()}


class WindowEnv:
    """Bind to one app window, offer grounded actions, then re-read it after each action.

    ``goal`` rides in every observation for the model. ``done_when`` reads a snapshot and says whether the task is finished; it is also the score. Without ``execute``
    the first chosen action is recorded as ``planned`` and the episode ends, so a dry run shows one decision and acts on
    nothing. ``clear_labels`` names a button pressed on ``reset`` when the window has one (a calculator's All Clear).
    """

    def __init__(
        self,
        driver: Driver,
        *,
        app_name: str,
        goal: str,
        done_when: Callable[[Snapshot], bool],
        execute: bool,
        clear_labels: tuple[str, ...],
        window_title: str = "",
        text: str = "",
        text_target: str = "",
        text_mode: str = "insert",
        pixel_targets: dict[str, tuple[float, float]] | None = None,
        box_targets: dict[str, str] | None = None,
        presence_threshold: float = DEFAULT_BOX_PRESENCE_THRESHOLD,
        box_confidence_threshold: float = DEFAULT_BOX_CONFIDENCE_THRESHOLD,
        screenshot: bool = False,
    ) -> None:
        self._driver = driver
        self._app_name = app_name
        self._window_title = window_title
        self._goal = goal
        self._done_when = done_when
        self._execute = execute
        self._clear_labels = clear_labels
        self._text = text
        self._text_target = text_target
        if text_mode not in {"insert", "replace"}:
            raise ValueError("text_mode must be insert or replace")
        self._text_mode = text_mode
        self._verified_value: str | None = None
        self._text_field: tuple[int, int, str, str] | None = None
        self._pixel_targets = dict(pixel_targets or {})
        self._box_targets = dict(box_targets or {})
        self._presence_threshold = presence_threshold
        self._box_confidence_threshold = box_confidence_threshold
        self._box_hits: dict[str, BoxHit] = {}
        self._screenshot = screenshot or bool(self._pixel_targets) or bool(self._box_targets)
        self._window: Window | None = None
        self._snapshot: Snapshot | None = None
        self._visual: dict[str, Any] | None = None
        self._keys: dict[str, Element] = {}
        self._text_keys: dict[str, Element] = {}
        self._presses: list[str] = []
        self._typed = False
        self._completion_stale = False
        self._completion_action: str | None = None
        self._planned: dict[str, Any] | None = None
        self._ended: str | None = None

    async def reset(self) -> None:
        self._window = (
            await self._driver.find_window(self._app_name, self._window_title)
            if self._window_title
            else await self._driver.find_window(self._app_name)
        )
        self._verified_value = None
        self._text_field = None
        self._completion_stale = False
        self._completion_action = None
        self._presses, self._planned, self._ended, self._typed = [], None, None, False
        await self._refresh()
        clear = next((e for e in self._keys.values() if e.label in self._clear_labels), None)
        if clear is not None and self._execute:
            await self._driver.click(self._window, self._token(clear))
            await self._refresh()
        if self._done_when(self._require_snapshot()):
            raise RuntimeError(
                "the window already shows --expect before any action; pass --clear <label> or reset the app"
            )

    async def refresh(self) -> None:
        """Re-read the window without clicking; a done or dry-run episode stays ended and candidates get new tokens."""
        if self._window is None:
            raise RuntimeError("the window was never observed: call reset first")
        await self._refresh()

    async def _refresh(self) -> None:
        assert self._window is not None
        self._snapshot = (
            await self._driver.window_state(self._window, screenshot=True)
            if self._screenshot
            else await self._driver.window_state(self._window)
        )
        if self._screenshot and self._snapshot.capture is None:
            raise DriverError("visual observations require a valid screenshot capture")
        self._visual = None
        self._box_hits = {}  # a new capture invalidates prior box centres
        if self._screenshot:
            capture = self._snapshot.capture
            assert capture is not None
            self._visual = visual_progress(capture)  # once per refreshed snapshot, not per observation read
        self._update_candidates()

    def _update_candidates(self) -> None:
        self._keys = {}
        self._text_keys = {}
        locator_counts = Counter(self._locator(e) for e in self._require_snapshot().elements)
        for element in self._require_snapshot().elements:
            if clickable(element):
                key = f"click:{element.label or element.role}"
                self._keys[key if key not in self._keys else f"{key}#{element.index}"] = element
            if not editable(element):
                continue
            locator = self._locator(element)
            if locator == self._text_field:
                self._typed = locator_counts[locator] == 1 and element.value == self._verified_value
            if (
                self._text
                and not self._typed
                and locator_counts[locator] == 1
                and (not self._text_target or self._text_target in (element.label, element.identifier))
                and (self._text_field is None or locator == self._text_field)
            ):
                name = self._text_target or element.identifier or element.label or element.role
                key = f"type:{name}"
                self._text_keys[key if key not in self._text_keys else f"{key}#{element.index}"] = element

    async def observe(self) -> dict[str, Any]:
        snapshot = self._require_snapshot()
        visible = [e for e in snapshot.elements if observable(e)]
        values = [e.value for e in visible if e.value and not clickable(e)]
        elements = [
            {
                "role": e.role,
                "label": e.label,
                "value": e.value,
                **({"identifier": e.identifier} if e.identifier else {}),
            }
            for e in visible
        ]
        # progress is the window itself, not the click count: a click that changes nothing must look stuck
        progress: dict[str, Any] = {"title": snapshot.window.title, "elements": elements, "values": values}
        if self._visual is not None:
            progress["image"] = self._visual
        state: dict[str, Any] = {
            "goal": self._goal,
            "app": self._app_name,
            "title": snapshot.window.title,
            "elements": elements,
            "presses": list(self._presses),
            "text_pending": bool(self._text and not self._typed),
            "progress": progress,
        }
        if self._planned is not None:
            state["planned"] = self._planned
        return state

    async def candidates(self) -> dict[str, str]:
        if self.done:
            return {}
        offered = {key: f'{e.role} "{e.label}"' + (f" = {e.value}" if e.value else "") for key, e in self._keys.items()}
        for key, element in self._text_keys.items():
            offered[key] = f'Type the task text into {element.role} "{key.removeprefix("type:")}"'
        offered.update(
            {
                f"pixel:{key}": f"Click {key}, at {x:.0%} across and {y:.0%} down the attached screenshot"
                for key, (x, y) in self._pixel_targets.items()
            }
        )
        for key, hit in self._box_hits.items():
            x, y = hit.centre
            offered[f"box:{key}"] = (
                f'Click {key} ("{hit.description}"), at box centre {x:.0%} across and {y:.0%} down the attached screenshot'
            )
        return {**offered, **RESERVED}

    async def images(self) -> tuple[Image, ...]:
        capture = self._require_snapshot().capture
        return (capture.image,) if self._screenshot and capture is not None else ()

    async def step(self, key: str) -> None:
        if key == ABSTAIN:
            self._ended = key
            return
        window = self._require_snapshot().window
        element = self._keys.get(key) or self._text_keys.get(key)
        if element is None and key not in await self.candidates():
            raise KeyError(key)
        if not self._execute:
            self._planned = {"key": key}
            if element is not None:
                self._planned.update(role=element.role, label=element.label, token=element.token)
            elif key.startswith("box:") and key.removeprefix("box:") in self._box_hits:
                self._planned["box"] = self._box_hits[key.removeprefix("box:")].record()
            self._ended = "planned"
            return
        completion_before_action = self._done_when(self._require_snapshot())
        if key in self._keys:
            assert element is not None
            await self._driver.click(window, self._token(element))
            self._presses.append(element.label)
        elif key in self._text_keys:
            assert element is not None
            text = self._text
            locator = self._locator(element)
            operation = self._driver.set_value if self._text_mode == "replace" else self._driver.type_text
            result = await operation(window, self._token(element), text)
            await self._refresh()
            current = [e for e in self._require_snapshot().elements if self._locator(e) == locator]
            if result.get("effect") != "confirmed":
                raise DriverError(f"text verification failed: driver effect {result.get('effect', 'missing')}")
            value = current[0].value if len(current) == 1 else None
            verified = value == text
            if value is not None and self._text_mode == "insert" and element.value:
                verified |= value.count(text) > element.value.count(text)
                # Replacing a selected occurrence with identical text leaves the value unchanged.
                # The driver must still confirm the action, and a fresh snapshot must contain it.
                verified |= value == element.value and text in value
            if not verified:
                raise DriverError("text verification failed: current field does not contain the requested content")
            self._verified_value = current[0].value
            self._text_field = locator
            self._typed = True
            # A result already visible before input must be produced again afterward.
            self._completion_stale = completion_before_action
            self._presses.append(key)
            self._update_candidates()
            return
        elif key.startswith("pixel:"):
            x, y = self._pixel_targets[key.removeprefix("pixel:")]
            capture = self._require_snapshot().capture
            assert capture is not None
            await self._driver.click_at(window, capture, x * capture.width, y * capture.height)
            self._presses.append(key)
        else:
            hit = self._box_hits[key.removeprefix("box:")]
            x, y = hit.centre
            capture = self._require_snapshot().capture
            assert capture is not None
            if capture.capture_id != hit.capture_id:
                raise DriverError(f"stale box target {key!r}: capture changed since locate")
            # same window binding / bounds check as pixel targets (click_at rejects out-of-bounds)
            await self._driver.click_at(window, capture, x * capture.width, y * capture.height)
            self._presses.append(key)
        await self._refresh()
        if not self._done_when(self._require_snapshot()):
            self._completion_stale = False
            self._completion_action = None
        elif not completion_before_action or key == self._completion_action:
            # A new result, or repeating its producing action (e.g. Save), refreshes the evidence.
            self._completion_stale = False
            self._completion_action = key

    @property
    def done(self) -> bool:
        return self._ended is not None or self.score == 1.0

    @property
    def score(self) -> float:
        if (self._text and not self._typed) or self._completion_stale:
            return 0.0
        return 1.0 if self._snapshot is not None and self._done_when(self._snapshot) else 0.0

    @property
    def box_targets(self) -> dict[str, str]:
        return dict(self._box_targets)

    def box_hit(self, key: str) -> BoxHit | None:
        """The resolved hit for ``box:KEY`` or ``KEY``, if still offered for the current capture."""
        name = key.removeprefix("box:")
        return self._box_hits.get(name)

    async def locate_box_targets(self, model: DecisionModel) -> dict[str, Any] | None:
        """One request: per configured target, a presence ``noul`` and a ``box``; offer those that pass thresholds.

        Targets with presence below ``presence_threshold``, box confidence below ``box_confidence_threshold``, or a
        centre outside the capture (fractions not in ``[0, 1)``) are not offered. Returns a locate record for the
        step log, or ``None`` when no box targets are configured.
        """
        self._box_hits = {}
        if not self._box_targets:
            return None
        capture = self._require_snapshot().capture
        if capture is None:
            raise DriverError("box targets require a valid screenshot capture")
        questions: dict[str, Question] = {}
        for key, description in self._box_targets.items():
            questions[f"present:{key}"] = NoulQuestion(f"Is {description} visible in `screenshot`?")
            questions[f"box:{key}"] = BoxQuestion(f"{description} in `screenshot`")
        decision = await model.decide_many(Observation(await self.observe(), await self.images()), questions)
        offered: dict[str, BoxHit] = {}
        rejected: list[dict[str, Any]] = []
        for key, description in self._box_targets.items():
            presence = decision.noul(f"present:{key}").p
            box = decision.box(f"box:{key}")
            cx, cy = box.centre
            reason = None
            if presence < self._presence_threshold:
                reason = "low_presence"
            elif box.confidence < self._box_confidence_threshold:
                reason = "low_box_confidence"
            elif not (0 <= cx < 1 and 0 <= cy < 1):
                reason = "centre_outside_capture"
            if reason is not None:
                rejected.append(
                    {
                        "key": key,
                        "reason": reason,
                        "presence": presence,
                        "box_confidence": box.confidence,
                        "centre": [cx, cy],
                    }
                )
                continue
            offered[key] = BoxHit(
                key=key, description=description, box=box, presence=presence, capture_id=capture.capture_id
            )
        self._box_hits = offered
        return {
            "offered": {key: hit.record() for key, hit in offered.items()},
            "rejected": rejected,
            "presence_threshold": self._presence_threshold,
            "box_confidence_threshold": self._box_confidence_threshold,
            "model": decision.model,
            "ms": decision.latency_ms,
            "input_tokens": decision.usage.input_tokens,
            "output_tokens": decision.usage.output_tokens,
            "usage_known": decision.usage.known,
            **decision.provenance,
        }

    @staticmethod
    def _token(element: Element) -> str:
        assert element.token is not None, "only elements with a token are offered"  # clickable() guarantees it
        return element.token

    def _require_snapshot(self) -> Snapshot:
        if self._snapshot is None:
            raise RuntimeError("the window was never observed: call reset first")
        return self._snapshot

    def _locator(self, element: Element) -> tuple[int, int, str, str]:
        window = self._require_snapshot().window
        return (window.pid, window.window_id, element.role, element.identifier or element.label)
