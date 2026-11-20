# coding: utf-8
"""Cross-question interference on decide_many (#127): isolated vs bundled vs reversed-bundle, plus a same-order control."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

from s1a import cli, probe_bundle
from s1a.decision_models import Choice, ChoiceQuestion, Noul, NoulQuestion, Reply, ScriptedModel
from s1a.decision_models.types import Observation, Question


def _cases(n: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "state": {"i": i},
            "questions": {
                "pick": {"type": "choice", "options": {"a": "keep this", "b": "the other one"}, "rules": "pick a"},
                "flag": {
                    "type": "noul",
                    "question": "Is this request urgent?",
                    "criteria": {"true": "urgent", "false": "not urgent"},
                },
            },
            "note": f"case {i}",
        }
        for i in range(n)
    ]


def _text(question: Question) -> str:
    if isinstance(question, ChoiceQuestion):
        return " ".join(str(value) for value in question.options.values())
    assert isinstance(question, NoulQuestion)
    return question.question


class _ContentModel(ScriptedModel):
    """Answers from the question itself; earlier heads in the same request are ignored."""

    def _answer(self, question: Question) -> dict[str, Any]:
        if isinstance(question, ChoiceQuestion):
            key = "a" if "a" in question.ids else question.ids[0]
            return {
                "choice": key,
                "probabilities": {i: 1.0 if i == key else 0.0 for i in question.ids},
                "confidence": 1.0,
            }
        return {"noul": 0.2}


class _InterferingModel(_ContentModel):
    """A later head flips when it can read earlier question text in the same request; alone it matches content."""

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        self.calls.append((observation, dict(questions)))
        prior: list[str] = []
        answers = {}
        for name, question in questions.items():
            if any(text.strip() for text in prior):
                answers[name] = self._flipped(question)
            else:
                answers[name] = self._answer(question)
            prior.append(_text(question))
        return Reply(answers=answers, latency_ms=self._latency_ms, usage=self._usage, model=self._model)

    def _flipped(self, question: Question) -> dict[str, Any]:
        if isinstance(question, ChoiceQuestion):
            key = "b" if "b" in question.ids else question.ids[-1]
            return {
                "choice": key,
                "probabilities": {i: 1.0 if i == key else 0.0 for i in question.ids},
                "confidence": 1.0,
            }
        return {"noul": 0.8}


class _AlternatingModel(_ContentModel):
    """Flips on every even file-order bundled ask: the same-order repeat disagrees, the first bundled ask does not."""

    def __init__(self) -> None:
        super().__init__()
        self._file_order_bundles = 0

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        self.calls.append((observation, dict(questions)))
        answers = {name: self._answer(question) for name, question in questions.items()}
        if list(questions) == ["pick", "flag"]:
            self._file_order_bundles += 1
            if self._file_order_bundles % 2 == 0:
                answers = {name: _InterferingModel()._flipped(question) for name, question in questions.items()}
        return Reply(answers=answers, latency_ms=self._latency_ms, usage=self._usage, model=self._model)


def _run(model: ScriptedModel, cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return asyncio.run(probe_bundle.run(cases or _cases(), model))


class TestTotalVariation(TestCase):
    def test_one_hot_move_is_one_and_noul_shift_is_the_probability_gap(self) -> None:
        left = probe_bundle.snapshot(Choice("a", {"a": 1.0, "b": 0.0}, 1.0))
        right = probe_bundle.snapshot(Choice("b", {"a": 0.0, "b": 1.0}, 1.0))
        self.assertEqual(probe_bundle.total_variation(left, right), 1.0)
        noul_low = probe_bundle.snapshot(Noul(0.2, 0.8))
        noul_high = probe_bundle.snapshot(Noul(0.8, 0.8))
        self.assertAlmostEqual(probe_bundle.total_variation(noul_low, noul_high), 0.6)


class TestReadCases(TestCase):
    def test_a_malformed_or_single_question_file_is_a_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                probe_bundle.read_cases(path)
            path.write_text(
                json.dumps({"state": {}, "questions": {"only": {"type": "choice", "options": {"a": "a"}}}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                probe_bundle.read_cases(path)


class TestRunBundle(TestCase):
    def test_an_interfering_backend_flips_only_the_later_head_and_only_when_bundled(self) -> None:
        model = _InterferingModel()
        summary = _run(model)
        pick, flag = summary["per_head"]["pick"], summary["per_head"]["flag"]
        self.assertEqual((pick["bundled_flips"], pick["bundled_flip_rate"]), (0, 0.0))
        self.assertEqual((flag["bundled_flips"], flag["bundled_flip_rate"]), (8, 1.0))
        self.assertEqual((pick["reversed_flips"], pick["reversed_flip_rate"]), (8, 1.0))
        self.assertEqual((flag["reversed_flips"], flag["reversed_flip_rate"]), (0, 0.0))
        self.assertEqual((pick["control_flips"], flag["control_flips"]), (0, 0))
        self.assertEqual(flag["mean_tv_bundled"], 0.6)
        self.assertEqual(pick["mean_tv_reversed"], 1.0)
        # isolated asks never see another question, so they stay on the content answer
        self.assertEqual({row["isolated"]["pick"]["key"] for row in summary["rows"]}, {"a"})
        self.assertEqual({row["isolated"]["flag"]["noul"] for row in summary["rows"]}, {0.2})
        self.assertEqual({row["bundled"]["answers"]["flag"]["noul"] for row in summary["rows"]}, {0.8})
        # file order is pick then flag; reversed is flag then pick
        self.assertEqual(summary["rows"][0]["bundled"]["heads"], ["pick", "flag"])
        self.assertEqual(summary["rows"][0]["reversed"]["heads"], ["flag", "pick"])
        # isolated calls are one question each; the first bundled call carries both, later first
        isolated_heads = [list(questions) for _obs, questions in model.calls if len(questions) == 1]
        self.assertEqual(isolated_heads[:2], [["pick"], ["flag"]])

    def test_a_content_only_backend_never_flips(self) -> None:
        summary = _run(_ContentModel())
        for name, stats in summary["per_head"].items():
            with self.subTest(head=name):
                self.assertEqual((stats["bundled_flips"], stats["reversed_flips"], stats["control_flips"]), (0, 0, 0))
                self.assertEqual(
                    (stats["mean_tv_bundled"], stats["mean_tv_reversed"], stats["mean_tv_control"]), (0.0, 0.0, 0.0)
                )

    def test_the_same_order_repeat_is_the_control_for_non_determinism(self) -> None:
        summary = _run(_AlternatingModel())
        # isolated + first bundled + reversed are followed by a control repeat; the even bundled-sized call flips
        self.assertEqual(summary["per_head"]["pick"]["control_flips"], 8)
        self.assertEqual(summary["per_head"]["flag"]["control_flip_rate"], 1.0)
        # the first bundled ask still matches isolated (both content-only on this model's odd calls)
        self.assertEqual(summary["per_head"]["pick"]["bundled_flips"], 0)
        self.assertEqual(summary["per_head"]["flag"]["bundled_flips"], 0)


def _main(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with patch.dict(os.environ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestBundleCli(TestCase):
    def _cases_file(self, tmp: str, cases: list[dict[str, Any]] | None = None) -> Path:
        path = Path(tmp) / "cases.jsonl"
        path.write_text("".join(json.dumps(case) + "\n" for case in (cases or _cases())), encoding="utf-8")
        return path

    def test_without_bundle_the_probe_output_and_exit_code_are_unchanged(self) -> None:
        from s1a.decision_models import JevModel
        from s1a.decision_models.fakes import ScriptedTransport

        single = [
            {
                "state": {"i": i},
                "options": {"a": "first", "b": "second"},
                "rules": "pick b",
                "accept": ["b"],
                "note": f"case {i}",
            }
            for i in range(8)
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cli, "build_model", lambda *_a, **_k: JevModel(ScriptedTransport(choose="b"))),
        ):
            path = Path(tmp) / "fit.jsonl"
            path.write_text("".join(json.dumps(case) + "\n" for case in single), encoding="utf-8")
            code, out, _err = _main(["probe", str(path)])
        self.assertEqual(code, 0)
        self.assertNotIn("bundle:", out)
        self.assertNotIn("flips", out)
        self.assertIn("verdict: fits", out)

    def test_bundle_prints_per_head_flip_rates_and_writes_both_answers_to_json(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cli, "build_model", lambda *_a, **_k: _InterferingModel()),
        ):
            out_path = Path(tmp) / "bundle.json"
            code, out, _err = _main(["probe", str(self._cases_file(tmp)), "--bundle", "--json", str(out_path)])
            saved = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)  # a measurement, not a fit verdict
        self.assertIn("bundled flips 0/8 (0%)", out)
        self.assertIn("bundled flips 8/8 (100%)", out)
        self.assertIn("reversed flips 8/8 (100%)", out)
        self.assertEqual((saved["model"], saved["bundle"], saved["cases"]), ("jev", True, 8))
        self.assertEqual(saved["per_head"]["flag"]["bundled_flip_rate"], 1.0)
        self.assertEqual(saved["rows"][0]["isolated"]["flag"]["noul"], 0.2)
        self.assertEqual(saved["rows"][0]["bundled"]["answers"]["flag"]["noul"], 0.8)

    def test_bundle_and_order_together_is_a_usage_error_before_any_model_loads(self) -> None:
        built: list[str] = []

        def build(name: str, **_kwargs: Any) -> ScriptedModel:
            built.append(name)
            return _ContentModel()

        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "build_model", build):
            code, out, err = _main(["probe", str(self._cases_file(tmp)), "--bundle", "--order", "reverse"])
        self.assertEqual((code, out, built), (2, "", []))
        self.assertIn("cannot be used together", err)

    def test_the_committed_interference_file_loads(self) -> None:
        path = Path("evals/probe/interference.jsonl")
        cases = probe_bundle.read_cases(path)
        self.assertGreaterEqual(len(cases), 8)
        kinds = [str(case.get("note") or "") for case in cases]
        self.assertTrue(any("negative control" in note for note in kinds))
        self.assertTrue(any("browser" in note.lower() for note in kinds))
        for case in cases:
            questions = probe_bundle.questions_of(case)
            self.assertGreaterEqual(len(questions), 2)
            types = {question.type for question in questions.values()}
            if "negative control" in str(case.get("note") or ""):
                self.assertEqual(types, {"choice", "noul"})
            if "browser" in str(case.get("note") or "").lower():
                self.assertIn("operation", questions)
