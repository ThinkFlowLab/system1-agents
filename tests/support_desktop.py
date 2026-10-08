# coding: utf-8
"""A ``Driver`` double over a toy calculator: digits, Multiply, Equals, All Clear, a display; tokens bound to a snapshot."""

from __future__ import annotations

from typing import Any

from s1a.desktop.driver import DriverError, Element, Snapshot, Window

BUTTONS = [str(d) for d in range(10)] + ["Multiply", "Equals", "All Clear"]


class FakeCalculator:
    def __init__(self, *, app_name: str = "Calculator", windows: int = 1) -> None:
        self.app_name = app_name
        self.windows = windows
        self.display = ""
        self.pending: int | None = None
        self.clicks: list[tuple[int, int, str]] = []
        self.snapshots = 0
        self.opened = 0
        self.launched_apps: list[str] = []

    async def launch_app(self, app_name: str, window_title: str = "") -> None:
        self.launched_apps.append(app_name)

    async def open(self) -> None:
        self.opened += 1

    async def close(self) -> None:
        self.opened -= 1

    async def find_window(self, app_name: str) -> Window:
        if app_name != self.app_name or self.windows != 1:
            raise DriverError(f"list_windows: {self.windows} on-screen window(s) of {app_name!r}")
        return Window(42, 7, app_name, "Calculator")

    async def window_state(self, window: Window) -> Snapshot:
        self.snapshots += 1
        elements = [Element(0, "AXStaticText", "", self.display, None, ())]
        for index, label in enumerate(BUTTONS, start=1):
            elements.append(Element(index, "AXButton", label, "", f"tok-{label}-{self.snapshots}", ("AXPress",)))
        elements.append(Element(99, "AXGroup", "keypad", "", "tok-keypad", ()))  # no press action: never a candidate
        return Snapshot(window, f"snap-{self.snapshots}", tuple(elements), {})

    async def click(self, window: Window, token: str) -> dict[str, Any]:
        self.clicks.append((window.pid, window.window_id, token))
        if not token.endswith(f"-{self.snapshots}"):
            raise DriverError(f"click: stale token {token!r}")
        label = token.split("-")[1]
        if label.isdigit():
            self.display += label
        elif label == "Multiply":
            self.pending, self.display = int(self.display or 0), ""
        elif label == "Equals":
            self.display, self.pending = str((self.pending or 0) * int(self.display or 0)), None
        elif label == "All Clear":
            self.display, self.pending = "", None
        return {"effect": "confirmed", "route": "accessibility"}


class FakeWindowsCalculator(FakeCalculator):
    """The same calculator with labels and roles observed in Windows UI Automation."""

    async def window_state(self, window: Window) -> Snapshot:
        snapshot = await super().window_state(window)
        names = {"1": "One", "2": "Two", "7": "Seven", "Multiply": "Multiply by", "All Clear": "Clear"}
        elements = tuple(
            Element(e.index, "Text", f"Display is {e.value or '0'}", "", e.token, ("invoke",))
            if e.index == 0
            else Element(
                e.index,
                "Button" if e.role == "AXButton" else "Group",
                names.get(e.label, e.label),
                e.value,
                e.token,
                ("invoke",) if e.role == "AXButton" else (),
            )
            for e in snapshot.elements
        )
        return Snapshot(window, snapshot.snapshot_id, elements, {})
