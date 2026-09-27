# coding: utf-8
"""One native window as an ``Env``: offer only actions grounded in its current snapshot."""

from __future__ import annotations

from typing import Any, Callable

from s1a.desktop.driver import Driver, DriverError, Element, Snapshot, Window

DONE = "done"
ABSTAIN = "abstain"
RESERVED = {
    ABSTAIN: "No offered action moves the task forward; stop without acting.",
}
_CLICKABLE_ROLES = ("button", "checkbox", "radiobutton", "menubutton", "link", "popupbutton", "disclosuretriangle")
_EDITABLE_ROLES = ("textfield", "textarea", "textview", "searchfield", "combobox", "editabletext")
_MENU_ROLES = frozenset({"menu", "menubar", "menubaritem", "menuitem"})


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
        self._window: Window | None = None
        self._snapshot: Snapshot | None = None
        self._keys: dict[str, Element] = {}
        self._text_keys: dict[str, Element] = {}
        self._presses: list[str] = []
        self._typed = False
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

    async def _refresh(self) -> None:
        assert self._window is not None
        self._snapshot = await self._driver.window_state(self._window)
        self._update_candidates()

    def _update_candidates(self) -> None:
        self._keys = {}
        self._text_keys = {}
        for element in self._require_snapshot().elements:
            if clickable(element):
                key = f"click:{element.label or element.role}"
                self._keys[key if key not in self._keys else f"{key}#{element.index}"] = element
            if not editable(element):
                continue
            locator = self._locator(element)
            if locator == self._text_field:
                self._typed = element.value == self._verified_value
            if (
                self._text
                and not self._typed
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
        state: dict[str, Any] = {
            "goal": self._goal,
            "app": self._app_name,
            "title": snapshot.window.title,
            "elements": [
                {
                    "role": e.role,
                    "label": e.label,
                    "value": e.value,
                    **({"identifier": e.identifier} if e.identifier else {}),
                }
                for e in visible
            ],
            "presses": list(self._presses),
            "text_pending": bool(self._text and not self._typed),
            "progress": {"values": values, "presses": len(self._presses)},
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
        return {**offered, **RESERVED}

    async def step(self, key: str) -> None:
        if key == ABSTAIN:
            self._ended = key
            return
        window = self._require_snapshot().window
        element = self._keys.get(key) or self._text_keys.get(key)
        if element is None and key not in await self.candidates():
            raise KeyError(key)
        if not self._execute:
            self._planned = {
                "key": key,
                "role": element.role if element else None,
                "label": element.label if element else None,
                "token": element.token if element else None,
            }
            self._ended = "planned"
            return
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
            if not verified:
                raise DriverError("text verification failed: current field does not contain the requested content")
            self._verified_value = current[0].value
            self._text_field = locator
            self._typed = True
            self._presses.append(key)
            self._update_candidates()
            return
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

    def _locator(self, element: Element) -> tuple[int, int, str, str]:
        window = self._require_snapshot().window
        return (window.pid, window.window_id, element.role, element.identifier or element.label)
