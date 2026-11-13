# coding: utf-8
"""The fit probe's option-order check (#90): flip rate under reversed or shuffled options, keys and texts unchanged."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

from s1a import cli, probe
from s1a.decision_models import JevModel, ScriptedTransport


def _cases(n: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "state": {"i": i},
            "options": {"a": "first text", "b": "second text", "c": "third text"},
            "rules": "pick b",
            "accept": ["b"],
            "note": f"case {i}",
        }
        for i in range(n)
    ]


class _AlternatingTransport(ScriptedTransport):
    """Answers ``a`` then ``b`` then ``a``...: a stand-in for a non-deterministic backend."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def _answer(self, question: Any) -> Any:
        self._calls += 1
        self._choose = "a" if self._calls % 2 else "b"
        return super()._answer(question)


def _run(transport: ScriptedTransport, order: str, cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return asyncio.run(probe.run_order(cases or _cases(), JevModel(transport), order, seed=3))


class TestParseOrder(TestCase):
    def test_reverse_and_shuffle_parse_and_anything_else_is_a_value_error(self) -> None:
        self.assertEqual(probe.parse_order("reverse"), ("reverse", 1))
        self.assertEqual(probe.parse_order("shuffle:4"), ("shuffle", 4))
        self.assertEqual(probe.parse_order("same:2"), ("same", 2))
        for bad in (
            "shuffle",
            "shuffle:0",
            "same",
            "same:0",
            f"same:{probe.MAX_SHUFFLES + 1}",
            "shuffle:x",
            f"shuffle:{probe.MAX_SHUFFLES + 1}",
            "sort",
            "",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.parse_order(bad)


class TestReorderings(TestCase):
    def test_keys_and_texts_are_identical_only_the_order_changes(self) -> None:
        options = {"x": "ex", "y": "why", "z": "zed", "w": "double u"}
        (reversed_,) = probe.reorderings(options, "reverse", 1, random.Random(0))
        self.assertEqual(list(reversed_), ["w", "z", "y", "x"])
        self.assertEqual(reversed_, options)  # dict equality ignores order: same keys, same texts
        shuffled = probe.reorderings(options, "shuffle", 10, random.Random(0))
        self.assertEqual(len(shuffled), 10)
        for copy in shuffled:
            self.assertEqual(copy, options)
            self.assertNotEqual(list(copy), list(options))

    def test_same_repeats_the_file_order(self) -> None:
        options = {"x": "ex", "y": "why", "z": "zed"}
        copies = probe.reorderings(options, "same", 3, random.Random(0))
        self.assertEqual([list(copy) for copy in copies], [["x", "y", "z"]] * 3)
        self.assertEqual(copies, [options] * 3)

    def test_a_one_option_case_has_no_other_order(self) -> None:
        self.assertEqual(probe.reorderings({"only": "t"}, "shuffle", 2, random.Random(0)), [{"only": "t"}] * 2)


class TestRunOrder(TestCase):
    def test_a_position_biased_backend_flips_every_case_under_reverse(self) -> None:
        transport = ScriptedTransport()  # always the first offered key
        summary = _run(transport, "reverse")
        self.assertEqual((summary["flips"], summary["cases"], summary["flip_rate"]), (8, 8, 1.0))
        self.assertEqual({row["choice"] for row in summary["rows"]}, {"a"})
        self.assertEqual({tuple(row["reordered_choices"]) for row in summary["rows"]}, {("c",)})
        self.assertEqual((summary["accuracy"], summary["reordered_accuracy"]), (0.0, 0.0))
        # the request bodies show both orders with byte-identical keys and texts
        first, second = (body["questions"]["pick"]["criteria"] for body in transport.bodies[:2])
        self.assertEqual((list(first), list(second)), (["a", "b", "c"], ["c", "b", "a"]))
        self.assertEqual(first, second)
        self.assertEqual(transport.bodies[0]["state"], transport.bodies[1]["state"])
        self.assertEqual(summary["rows"][0]["asks"][1]["options"], ["c", "b", "a"])

    def test_a_content_based_backend_never_flips(self) -> None:
        for order in ("reverse", "shuffle:5"):
            with self.subTest(order=order):
                summary = _run(ScriptedTransport(choose="b"), order)
                self.assertEqual((summary["flips"], summary["flip_rate"]), (0, 0.0))
                self.assertEqual((summary["accuracy"], summary["reordered_accuracy"]), (1.0, 1.0))
                self.assertEqual(summary["verdict"], "fits")

    def test_same_is_the_control_position_bias_alone_never_flips_but_noise_does(self) -> None:
        biased = _run(ScriptedTransport(), "same:2")  # always the first key: biased, but deterministic
        self.assertEqual((biased["flips"], biased["reordered_asks"]), (0, 16))
        # a backend that disagrees with itself on every other call flips even though the order never changed
        noisy = _run(_AlternatingTransport(), "same:1")
        self.assertEqual((noisy["flips"], noisy["flip_rate"]), (8, 1.0))
        self.assertEqual({tuple(row["asks"][1]["options"]) for row in noisy["rows"]}, {("a", "b", "c")})

    def test_shuffle_asks_n_reorders_per_case_and_is_seeded(self) -> None:
        a, b = _run(ScriptedTransport(), "shuffle:3"), _run(ScriptedTransport(), "shuffle:3")
        self.assertEqual(a["reordered_asks"], 24)
        self.assertEqual([row["asks"] for row in a["rows"]], [row["asks"] for row in b["rows"]])
        self.assertEqual(a["flip_rate"], 1.0)  # every shuffle differs from the file order, so the first key moves
        self.assertIn("FLIP", probe.render_order(a))
        self.assertIn("flips: 8/8 (100%)", probe.render_order(a))


def _main(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with patch.dict(os.environ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestProbeCli(TestCase):
    def _cases_file(self, tmp: str) -> Path:
        path = Path(tmp) / "cases.jsonl"
        path.write_text("".join(json.dumps(case) + "\n" for case in _cases()), encoding="utf-8")
        return path

    def test_without_order_the_probe_output_and_exit_code_are_unchanged(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cli, "build_model", lambda *_a, **_k: JevModel(ScriptedTransport(choose="b"))),
        ):
            code, out, _err = _main(["probe", str(self._cases_file(tmp))])
        self.assertEqual(code, 0)
        self.assertNotIn("flips", out)
        self.assertIn("verdict: fits", out)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cli, "build_model", lambda *_a, **_k: JevModel(ScriptedTransport())),
        ):
            self.assertEqual(_main(["probe", str(self._cases_file(tmp))])[0], 1)

    def test_order_prints_the_flip_rate_and_writes_every_order_to_json(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cli, "build_model", lambda *_a, **_k: JevModel(ScriptedTransport())),
        ):
            out_path = Path(tmp) / "order.json"
            code, out, _err = _main(
                ["probe", str(self._cases_file(tmp)), "--order", "reverse", "--json", str(out_path)]
            )
            saved = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(code, 1)  # verdict from the file order: the first key is always wrong here
        self.assertIn("flips: 8/8 (100%)", out)
        self.assertEqual((saved["model"], saved["order"], saved["flip_rate"]), ("jev", "reverse", 1.0))
        self.assertEqual(saved["rows"][0]["asks"][0]["options"], ["a", "b", "c"])

    def test_a_bad_order_is_a_usage_error_before_any_model_loads(self) -> None:
        built: list[str] = []

        def build(name: str, **_kwargs: Any) -> JevModel:
            built.append(name)
            return JevModel(ScriptedTransport())

        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "build_model", build):
            code, out, err = _main(["probe", str(self._cases_file(tmp)), "--order", "sideways"])
        self.assertEqual((code, out, built), (2, "", []))
        self.assertEqual(len(err.strip().splitlines()), 1)

    def test_seed_or_json_without_order_is_a_usage_error_before_any_model_loads(self) -> None:
        built: list[str] = []

        def build(name: str, **_kwargs: Any) -> JevModel:
            built.append(name)
            return JevModel(ScriptedTransport())

        for extra in (["--seed", "1"], ["--json", "out.json"]):
            with (
                self.subTest(extra=extra),
                tempfile.TemporaryDirectory() as tmp,
                patch.object(cli, "build_model", build),
            ):
                code, out, err = _main(["probe", str(self._cases_file(tmp)), *extra])
            self.assertEqual((code, out, built), (2, "", []))
            self.assertIn("only apply with --order", err)
