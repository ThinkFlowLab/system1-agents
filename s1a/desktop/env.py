# coding: utf-8
"""One native window as an ``Env``: its elements are the state, one click per clickable element is a candidate."""

from __future__ import annotations

from typing import Any, Callable

from s1a.desktop.driver import Driver, Element, Snapshot, Window

DONE = "done"
ABSTAIN = "abstain"
RESERVED = {
    DONE: "The window shows the task finished; stop here.",
    ABSTAIN: "No offered click moves the task forward; stop without acting.",
}
_CLICKABLE_ROLES = ("button", "checkbox", "radiobutton", "menubutton", "link", "popupbutton", "disclosuretriangle")


def clickable(element: Element) -> bool:
    role = element.role.casefold().removeprefix("ax")
    return element.token is not None and (
        role in _CLICKABLE_ROLES or any("press" in a.casefold() or "click" in a.casefold() for a in element.actions)
    )


class WindowEnv:
    """The mechanics: bind to the app's window, offer its clickable elements, click one, re-read the window.

    ``goal`` rides in every observation for the model. ``done_when`` reads a snapshot and says whether the task is finished; it is also the score. Without ``execute``
    the first chosen click is recorded as ``planned`` and the episode ends, so a dry run shows one decision and acts on
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
    ) -> None:
        self._driver = driver
        self._app_name = app_name
        self._goal = goal
        self._done_when = done_when
        self._execute = execute
        self._clear_labels = clear_labels
        self._window: Window | None = None
        self._snapshot: Snapshot | None = None
        self._keys: dict[str, Element] = {}
        self._presses: list[str] = []
        self._planned: dict[str, Any] | None = None
        self._ended: str | None = None

    async def reset(self) -> None:
        self._window = await self._driver.find_window(self._app_name)
        self._presses, self._planned, self._ended = [], None, None
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
        self._snapshot = await self._driver.window_state(self._window)
        self._keys = {}
        for element in self._snapshot.elements:
            if clickable(element):
                key = f"click:{element.label or element.role}"
                self._keys[key if key not in self._keys else f"{key}#{element.index}"] = element

    async def observe(self) -> dict[str, Any]:
        snapshot = self._require_snapshot()
        values = [e.value for e in snapshot.elements if e.value and not clickable(e)]
        elements = [{"role": e.role, "label": e.label, "value": e.value} for e in snapshot.elements]
        state: dict[str, Any] = {
            "goal": self._goal,
            "app": self._app_name,
            "title": snapshot.window.title,
            "elements": elements,
            "presses": list(self._presses),
            # progress is the window itself, not the click count: a click that changes nothing must look stuck
            "progress": {"title": snapshot.window.title, "elements": elements, "values": values},
        }
        if self._planned is not None:
            state["planned"] = self._planned
        return state

    async def candidates(self) -> dict[str, str]:
        if self.done:
            return {}
        offered = {key: f'{e.role} "{e.label}"' + (f" = {e.value}" if e.value else "") for key, e in self._keys.items()}
        return {**offered, **RESERVED}

    async def step(self, key: str) -> None:
        if key in RESERVED:
            self._ended = key
            return
        element = self._keys[key]
        window = self._require_snapshot().window
        if not self._execute:
            self._planned = {"key": key, "role": element.role, "label": element.label, "token": element.token}
            self._ended = "planned"
            return
        await self._driver.click(window, self._token(element))
        self._presses.append(element.label)
        await self._refresh()

    @property
    def done(self) -> bool:
        return self._ended is not None or (self._snapshot is not None and self._done_when(self._snapshot))

    @property
    def score(self) -> float:
        return 1.0 if self._snapshot is not None and self._done_when(self._snapshot) else 0.0

    @staticmethod
    def _token(element: Element) -> str:
        assert element.token is not None, "only elements with a token are offered"  # clickable() guarantees it
        return element.token

    def _require_snapshot(self) -> Snapshot:
        if self._snapshot is None:
            raise RuntimeError("the window was never observed: call reset first")
        return self._snapshot
