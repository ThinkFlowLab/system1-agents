# coding: utf-8
"""The BrowseSafe++ detector adapter, offline: no SDK install, no network, a fake request and a scripted model."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a import browsesafe_pp as bspp
from s1a.agents.injection_guard import MAX_CHARS
from s1a.decision_models import ScriptedModel

ROOT = Path(__file__).resolve().parents[1]


def _request(content: str = "Recipe text.", view: str = "markdown") -> SimpleNamespace:
    return SimpleNamespace(content=content, view=view)


def _detector(ps: list[float] | None = None, *, fail: bool = False, on_failure: str = "error") -> bspp.Detector:
    error = build_error(StatusCode.MODEL_CALL_FAILED, error_msg="echo of the page: SECRET") if fail else None
    model = ScriptedModel(noul=list(ps or []), error=error)
    return bspp.Detector(model, on_failure=on_failure, prediction=lambda injected: {"injected": injected})


class TestState(TestCase):
    def test_views_map_to_the_guarded_tools(self) -> None:
        self.assertEqual(bspp.state_of(_request(view="accessibility"))["tool"], "browser_snapshot")
        self.assertEqual(bspp.state_of(_request(view="html"))["tool"], "fetch_webpage")
        self.assertEqual(bspp.state_of(_request(view="markdown"))["tool"], "fetch_webpage")

    def test_content_is_clipped_like_the_runtime(self) -> None:
        state = bspp.state_of(_request("x" * (MAX_CHARS + 500)))
        self.assertEqual(len(state["text"]), MAX_CHARS)
        self.assertEqual(bspp.state_of(_request("short"))["text"], "short")

    def test_bad_requests_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            bspp.state_of(_request(view="screenshot"))
        with self.assertRaises(TypeError):
            bspp.state_of(SimpleNamespace(content=None, view="html"))


class TestBands(TestCase):
    def test_band_mapping(self) -> None:
        self.assertIs(bspp.injected_of("act"), True)
        self.assertIs(bspp.injected_of("allow"), False)
        self.assertIsNone(bspp.injected_of("uncertain"))
        with self.assertRaises(ValueError):
            bspp.injected_of("failure")

    def test_probabilities_go_through_the_guard_thresholds(self) -> None:
        detector = _detector([0.9, 0.1, 0.5, 0.7, 0.3])
        got = [detector(_request())["injected"] for _ in range(5)]
        self.assertEqual(got, [True, False, None, True, False])
        self.assertEqual(detector.bands, {"allow": 2, "uncertain": 1, "act": 2, "failure": 0})
        detector.close()

    def test_the_model_sees_the_clipped_state(self) -> None:
        detector = _detector([0.1])
        detector(_request("y" * (MAX_CHARS * 2), view="accessibility"))
        observation, _ = detector.decision_model.calls[0]
        self.assertEqual(observation.state, {"tool": "browser_snapshot", "text": "y" * MAX_CHARS})
        detector.close()


class TestFailures(TestCase):
    def test_a_failed_decision_raises_without_its_message(self) -> None:
        detector = _detector(fail=True)
        with self.assertRaises(bspp.DecisionFailed) as caught:
            detector(_request())
        self.assertIn("MODEL_CALL_FAILED", str(caught.exception))
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertEqual(detector.bands["failure"], 1)
        detector.close()

    def test_fail_closed_answers_injected(self) -> None:
        detector = _detector(fail=True, on_failure="closed")
        self.assertEqual(detector(_request()), {"injected": True})
        detector.close()

    def test_unknown_failure_policy_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _detector(on_failure="open")

    def test_close_is_idempotent(self) -> None:
        detector = _detector([0.1])
        detector(_request())
        detector.close()
        detector.close()


class TestEntryPoint(TestCase):
    def test_predict_builds_one_detector_from_the_environment(self) -> None:
        model = ScriptedModel(noul=[0.9, 0.1])
        with (
            patch.object(bspp, "_detector", None),
            patch.object(bspp, "build_model", return_value=model) as build,
            patch.object(bspp, "sdk_prediction", side_effect=lambda injected: ("pred", injected)),
            patch.object(bspp.atexit, "register"),
            patch.dict("os.environ", {"S1A_BSPP_MODEL": "jev"}),
        ):
            first = bspp.predict(_request())
            second = bspp.predict(_request())
            build.assert_called_once_with("jev")
            self.assertEqual((first, second), (("pred", True), ("pred", False)))
            assert bspp._detector is not None
            bspp._detector.close()

    def test_unknown_model_name_is_refused(self) -> None:
        with patch.object(bspp, "_detector", None), patch.dict("os.environ", {"S1A_BSPP_MODEL": "gpt"}):
            with self.assertRaises(ValueError):
                bspp.detector()

    def test_core_code_never_imports_the_sdk_at_module_level(self) -> None:
        for path in (ROOT / "s1a").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:  # module level only; the adapter imports inside a function
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                self.assertFalse(any(n.split(".")[0] == "browsesafe_pp" for n in names), path)

    def test_the_sdk_is_not_a_core_dependency(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        deps = list(project.get("dependencies", []))
        deps += [d for extra in project.get("optional-dependencies", {}).values() for d in extra]
        self.assertFalse([d for d in deps if "browsesafe" in d.lower()])
