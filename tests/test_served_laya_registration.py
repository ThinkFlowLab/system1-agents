# coding: utf-8
"""``laya-served`` is offered wherever ``laya`` is: every tuple, list or set of model names, every ``match`` over
them and every ``Literal`` that lists ``laya`` lists ``laya-served`` too. A place added later without it fails here."""

from __future__ import annotations

import ast
import typing
from pathlib import Path

from s1a import mcp_server
from s1a.cli import DECIDE_MODEL_NAMES
from s1a.decision_models import DECISION_MODEL_NAMES
from s1a.rails import RAIL_MODEL_NAMES
from s1a.tool.loop import MODEL_NAMES

SOURCE = Path(__file__).resolve().parents[1] / "s1a"


def constants(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def places_missing_served() -> list[str]:
    missing = []
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                values = {e.value for e in node.elts if isinstance(e, ast.Constant)}
            elif isinstance(node, ast.Match):
                values = set().union(*(constants(case.pattern) for case in node.cases))
            elif (
                isinstance(node, ast.Subscript)
                and getattr(node.value, "id", getattr(node.value, "attr", "")) == "Literal"
            ):
                values = constants(node.slice)
            else:
                continue
            if "laya" in values and "laya-served" not in values:
                missing.append(f"{path.relative_to(SOURCE.parent)}:{node.lineno}")
    return missing


def test_every_place_that_offers_laya_offers_laya_served() -> None:
    assert places_missing_served() == []


def test_the_named_lists() -> None:
    for names in (DECISION_MODEL_NAMES, DECIDE_MODEL_NAMES, RAIL_MODEL_NAMES, MODEL_NAMES):
        assert "laya-served" in names
    decide_model = typing.get_type_hints(
        mcp_server.decide.__wrapped__ if hasattr(mcp_server.decide, "__wrapped__") else mcp_server.decide
    )["model"]
    assert "laya-served" in typing.get_args(decide_model)


def test_the_check_would_catch_a_missing_place(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "s1a").mkdir()
    (tmp_path / "s1a" / "new_front.py").write_text('NAMES = ("jev", "laya")\n')
    monkeypatch.setattr(__name__ + ".SOURCE", tmp_path / "s1a")
    assert places_missing_served() == ["s1a/new_front.py:1"]
