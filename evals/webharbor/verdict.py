# coding: utf-8
"""The site verifier's output, as ``tests/test.sh`` leaves it under ``/logs/verifier/``.

``reward.json`` holds ``reward`` (1/0) and check counts; ``verifier_verdict.json`` the verifier's own JSON with its
evidence. The reward is the official verdict. A missing or malformed reward is a verifier error, never a pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Verdict:
    passed: bool
    error: str | None = None  # set when the verifier produced no usable reward
    checks_total: int | None = None
    checks_passed: int | None = None
    failed_checks: tuple[str, ...] = field(default_factory=tuple)


def parse_verdict_json(raw: str) -> dict[str, Any]:
    """The verifier's JSON object even when it printed log lines before it, the way ``test.sh`` reads it."""
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    lines = raw.splitlines()
    for start in reversed([i for i, line in enumerate(lines) if line.strip() == "{"]):
        try:
            value = json.loads("\n".join(lines[start:]))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _failed(verdict: dict[str, Any]) -> tuple[str, ...]:
    names = []
    for item in verdict.get("evidence") or []:
        if isinstance(item, dict) and not item.get("pass"):
            names.append(str(item.get("name") or item.get("check") or "unnamed"))
        elif isinstance(item, str) and item.lstrip("[").startswith("FAIL"):
            names.append(item[:80])
    return tuple(names)


def read_verdict(verifier_dir: Path) -> Verdict:
    reward_path = verifier_dir / "reward.json"
    if not reward_path.is_file():
        return Verdict(False, error="verifier wrote no reward.json")
    try:
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return Verdict(False, error="reward.json is not JSON")
    if not isinstance(reward, dict) or reward.get("reward") not in (0, 1, True, False):
        return Verdict(False, error="reward.json has no 0/1 reward")
    verdict_path = verifier_dir / "verifier_verdict.json"
    details = parse_verdict_json(verdict_path.read_text(encoding="utf-8")) if verdict_path.is_file() else {}
    total, passed = reward.get("checks_total"), reward.get("checks_passed")
    return Verdict(
        passed=bool(reward["reward"]),
        checks_total=total if isinstance(total, int) else None,
        checks_passed=passed if isinstance(passed, int) else None,
        failed_checks=_failed(details),
    )
