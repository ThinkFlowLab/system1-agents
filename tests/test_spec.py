# coding: utf-8
"""The agent spec: validation, the registered agents, the shared flags, and a template played through the runner offline."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError, build_error

from s1a import run as agents
from s1a.run import started_runner
from s1a.decision_models import JevModel, ScriptedTransport
from s1a.spec import BrowserAgentSpec, Budget, RailSpec, Series, ToolAgentSpec
from s1a.tool import loop, series
from support import COUNTER, CounterEnv, counter_series


def stride_flag(parser: ArgumentParser) -> None:
    parser.add_argument("--stride", type=int, default=1)


class TestSpecValidation(TestCase):
    def test_budget_rejects_zero_steps_zero_timeout_and_a_negative_stall(self) -> None:
        for bad in ((0, 1, 0), (1, 0, 0), (1, 1, -1), (2.5, 1, 0)):
            with self.assertRaises(ValueError):
                Budget(*bad)

    def test_a_browser_spec_rejects_a_blank_goal_and_a_choice_rail_needs_two_criteria(self) -> None:
        flights = agents.load("flights")
        with self.assertRaises(ValueError):
            replace(flights, goal=" ")
        self.assertIsNone(replace(flights, goal=None).goal)
        guard = agents.load("injection_guard")
        with self.assertRaises(ValueError):
            replace(guard, question="choice", criteria={"only": "one"}, flagged="only")

    def test_spec_rejects_a_bad_name_and_empty_text(self) -> None:
        for change in (
            {"name": "Bad Name"},
            {"name": ""},
            {"name": "with-dash"},
            {"name": "2048"},
            {"rules": " "},
            {"description": ""},
        ):
            with self.assertRaises(ValueError):
                replace(COUNTER, **change)

    def test_every_registered_agent_loads_as_a_spec_named_after_its_module(self) -> None:
        names = agents.names()
        self.assertTrue(names)
        self.assertNotIn("_templates", names)
        for name in names:
            with self.subTest(agent=name):
                try:
                    spec = agents.load(name)
                except agents.MissingPackage as exc:
                    self.skipTest(str(exc))
                self.assertIsInstance(spec, (ToolAgentSpec, BrowserAgentSpec, RailSpec), name)
                self.assertEqual(spec.name, name)

    def test_load_rejects_a_spec_whose_name_differs_from_its_module(self) -> None:
        module = SimpleNamespace(SPEC=replace(COUNTER, name="other"))
        with patch.object(agents, "names", lambda: ["counter"]), patch("importlib.import_module", lambda _: module):
            with self.assertRaises(ValueError):
                agents.load("counter")


class TestSharedFlags(TestCase):
    def test_budget_is_the_default_and_the_agents_flags_come_after(self) -> None:
        spec = replace(COUNTER, flags=stride_flag)
        args = series.parser(spec).parse_args(["--model", "random", "--rethink", "off", "--episodes", "2"])
        self.assertEqual((args.max_steps, args.timeout, args.stride, args.headed, args.seed), (5, 30.0, 1, False, 0))
        args = series.parser(spec).parse_args(
            ["--model", "random", "--rethink", "off", "--episodes", "2", "--max-steps", "9", "--stride", "3"]
        )
        self.assertEqual((args.max_steps, args.stride), (9, 3))

    def test_every_shared_flag_has_help_on_every_front(self) -> None:
        for spec in (COUNTER, agents.load("flights"), agents.load("injection_guard")):
            for row in agents.flags(spec):
                self.assertTrue(row["help"], (spec.name, row["flag"]))

    def test_episodes_max_steps_and_timeout_at_or_below_zero_are_usage_errors(self) -> None:
        for flags in (
            ["--episodes", "0"],
            ["--episodes", "-1"],
            ["--episodes", "1", "--max-steps", "0"],
            ["--episodes", "1", "--timeout", "0"],
            ["--episodes", "1", "--timeout", "-2.5"],
        ):
            with self.assertRaises(SystemExit) as caught:
                series.parser(COUNTER).parse_args(["--model", "random", "--rethink", "off", *flags])
            self.assertEqual(caught.exception.code, 2)


class TestTemplatePlaysThroughTheRunner(IsolatedAsyncioTestCase):
    async def test_random_plays_two_episodes_and_writes_one_job_folder(self) -> None:
        args = series.parser(COUNTER).parse_args(["--model", "random", "--rethink", "off", "--episodes", "2"])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(loop, "WORKSPACE", Path(tmp) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
        ):
            async with started_runner():
                result = await series.play(COUNTER, args, results_dir=Path(tmp) / "results")
            job_dir = Path(result["job_dir"])
            self.assertEqual(job_dir.parent, Path(tmp) / "results" / "counter")
            summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["env"], summary["policy"], summary["episodes"]), ("counter", "random", 2))
        self.assertEqual((result["env"], result["episodes"], result["errors"]), ("counter", 2, 0))
        self.assertLessEqual(summary["mean_steps"], 5)

    async def test_a_series_that_selects_no_seeds_is_a_run_error(self) -> None:
        spec = replace(COUNTER, series=lambda flags: replace(counter_series(flags), seeds=()))
        args = series.parser(spec).parse_args(["--model", "random", "--rethink", "off", "--episodes", "1"])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(series, "optional_chat_model", lambda: None),
            self.assertRaisesRegex(RuntimeError, "no episodes"),
        ):
            async with started_runner():
                await series.play(spec, args, results_dir=Path(tmp) / "results")

    async def test_a_series_without_chat_tokens_never_looks_a_price_up(self) -> None:
        args = series.parser(COUNTER).parse_args(["--model", "rule", "--rethink", "off", "--episodes", "1"])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(loop, "WORKSPACE", Path(tmp) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
            patch.object(series, "chat_prices", _no_lookup),
        ):
            async with started_runner():
                result = await series.play(COUNTER, args, results_dir=Path(tmp) / "results")
        self.assertEqual((result["chat_calls"], result["cost_usd"]), (0, 0.0))

    async def test_series_setup_and_pricing_run_off_the_event_loop_thread(self) -> None:
        on_main: dict[str, bool] = {}

        def slow_series(flags: Any) -> Series:
            on_main["series"] = threading.current_thread() is threading.main_thread()
            return COUNTER.series(flags)

        def slow_pricing(episodes: list[Any]) -> None:
            on_main["pricing"] = threading.current_thread() is threading.main_thread()

        args = series.parser(COUNTER).parse_args(["--model", "rule", "--rethink", "off", "--episodes", "1"])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(loop, "WORKSPACE", Path(tmp) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
            patch.object(series, "price_episodes", slow_pricing),
        ):
            async with started_runner():
                await series.play(replace(COUNTER, series=slow_series), args, results_dir=Path(tmp) / "results")
        self.assertEqual(on_main, {"series": False, "pricing": False})

    async def test_a_series_that_raises_mid_way_still_writes_the_finished_episodes(self) -> None:
        envs: list[CounterEnv] = []

        def flaky_series(flags: Any) -> Series:
            return Series(
                seeds=range(2),
                env_for=lambda seed: envs.append(CounterEnv()) or envs[-1],
                session=contextlib.nullcontext(),
                baseline=("flaky", lambda observation, offered: "inc" if len(envs) < 2 else "bogus"),
                annotate=lambda env, episode: None,
            )

        args = series.parser(COUNTER).parse_args(["--model", "rule", "--rethink", "off", "--episodes", "2"])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(loop, "WORKSPACE", Path(tmp) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
        ):
            async with started_runner():
                with self.assertRaises(RuntimeError) as caught:
                    await series.play(replace(COUNTER, series=flaky_series), args, results_dir=Path(tmp) / "results")
            self.assertIn("bogus", str(caught.exception))
            (job_dir,) = (Path(tmp) / "results" / "counter").iterdir()
            summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["episodes"], summary["mean_score"]), (1, 3.0))

    async def test_missing_keys_and_a_bad_price_are_raised_before_the_series_is_built(self) -> None:
        def never(flags: Any) -> Series:
            raise AssertionError("the series must not be built before the run's keys are checked")

        spec = replace(COUNTER, series=never)
        for model_name, env in (("jev", {}), ("llm", {}), ("rule", {"CHAT_USD_PER_M_INPUT": "0.3"})):
            args = series.parser(COUNTER).parse_args(["--model", model_name, "--rethink", "off", "--episodes", "1"])
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(series, "optional_chat_model", lambda: None),
                self.assertRaises((RuntimeError, ValueError, BaseError)),
            ):
                await series.play(spec, args, results_dir=Path("unused"))

    async def test_a_series_whose_every_episode_fails_writes_its_job_and_raises(self) -> None:
        args = series.parser(COUNTER).parse_args(["--model", "jev", "--rethink", "off", "--episodes", "2"])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(loop, "WORKSPACE", Path(tmp) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
            patch.object(series, "build_model", _refusing),
        ):
            async with started_runner():
                with self.assertRaises(RuntimeError) as caught:
                    await series.play(COUNTER, args, results_dir=Path(tmp) / "results")
            self.assertIn("decision failed", str(caught.exception))
            (job_dir,) = (Path(tmp) / "results" / "counter").iterdir()
            summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["episodes"], summary["errors"], summary["decisions"]), (2, 2, 0))


def _no_lookup(model_name: str) -> None:
    raise AssertionError("a series that spent no chat tokens must not fetch the price catalogue")


def _refusing(model_name: str, **kwargs: Any) -> JevModel:
    """The jev model of a run whose key is wrong or whose endpoint is down."""
    error = build_error(StatusCode.MODEL_CALL_FAILED, error_msg="decisions endpoint returned HTTP 401")
    return JevModel(ScriptedTransport(error=error))
