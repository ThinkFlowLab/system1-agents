# coding: utf-8
"""The recorded served-Laya responses match the as-implemented spec, so the mock server in
``test_decision_models_served`` answers the way the real ones do."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = sorted((ROOT / "tests" / "data" / "served_laya").glob("*.json"))
SPEC = yaml.safe_load((ROOT / "docs" / "api" / "laya-systemone.current.openapi.yaml").read_text(encoding="utf-8"))
REGISTRY = Registry().with_resource("urn:spec", Resource.from_contents(SPEC, default_specification=DRAFT202012))


def errors(schema: str, instance: object) -> list[str]:
    validator = Draft202012Validator({"$ref": f"urn:spec#/components/schemas/{schema}"}, registry=REGISTRY)
    return [error.message for error in validator.iter_errors(instance)]


def test_there_are_fixtures_from_every_server() -> None:
    prefixes = {path.name.split(".")[0] for path in FIXTURES}
    assert prefixes == {"worker", "frontend", "frontend-down", "laya-serve"}


@pytest.mark.parametrize("path", FIXTURES, ids=[path.stem for path in FIXTURES])
def test_fixture_matches_the_spec(path: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    server, case = path.stem.split(".", 1)
    status, body = record["status"], record["body"]
    if status in (502, 504):  # the frontend's own errors are plain text
        assert record["content_type"] == "text/plain"
        return
    assert record["content_type"] == "application/json"
    if case == "health":
        if server == "laya-serve":  # answers before warmup, with the configured device only
            assert set(body) == {"status", "loaded", "device"}
        else:
            assert errors("Health", body) == []
        return
    schema = "DecisionResponse" if status == 200 else "Error"
    assert errors(schema, body) == []


@pytest.mark.parametrize(
    "path", [p for p in FIXTURES if "request" in json.loads(p.read_text(encoding="utf-8"))], ids=lambda p: p.stem
)
def test_sent_requests_match_the_spec_unless_meant_to_fail(path: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    found = errors("DecisionRequest", record["request"])
    if record["status"] in (400, 422):
        assert found, "a request the server rejected should not pass the schema"
    else:
        assert found == []
