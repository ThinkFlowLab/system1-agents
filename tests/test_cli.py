# coding: utf-8
"""``s1a``: list, run (offline, on the counter spec) and decide, plus the launch args the entry point sets."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import TestCase, skipUnless
from unittest.mock import patch

from s1a import cli
from s1a import run as agents
from s1a.decision_models import JevModel, ScriptedTransport
from s1a.spec import Budget
from s1a.tool import loop, series
from support import COUNTER

RUN = ["run", "counter", "--model", "random", "--rethink", "off", "--episodes", "1", "--showcase"]
NO_KEYS = {"TYPESAFE_API_KEY": "", "OPENROUTER_API_KEY": "", "TYPESAFE_API_URL": ""}
HAVE_RLCARD = importlib.util.find_spec("rlcard") is not None  # the one tool agent that runs offline


def _main(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with patch.dict(os.environ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):  # main loads .env
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestList(TestCase):
    def test_lists_every_registered_agent(self) -> None:
        code, out, _err = _main(["list"])
        self.assertEqual(code, 0)
        self.assertEqual(out.split(), agents.names())
        self.assertIn("injection_guard", out)


class TestRun(TestCase):
    def test_an_unknown_agent_exits_2_with_the_names(self) -> None:
        code, _out, err = _main(["run", "nope"])
        self.assertEqual(code, 2)
        self.assertIn("blackjack", err)

    def test_run_help_lists_the_registered_agents(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as caught:
            cli.parser().parse_args(["run", "--help"])
        self.assertEqual(caught.exception.code, 0)
        for name in agents.names():
            self.assertIn(name, out.getvalue())

    def test_an_os_error_inside_a_run_is_a_run_error_and_exits_1(self) -> None:
        def denied(name: str) -> None:
            raise PermissionError("[Errno 13] Permission denied: 'results'")

        with patch.object(agents, "load", denied):
            code, out, err = _main(["run", "counter", "--model", "random", "--rethink", "off", "--episodes", "1"])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("Permission denied", err)

    def test_an_index_error_inside_a_run_is_not_an_unknown_agent(self) -> None:
        def broken(name: str) -> None:
            raise IndexError("list index out of range")

        with patch.object(agents, "load", broken), self.assertRaises(IndexError):
            _main(["run", "counter", "--model", "random", "--rethink", "off", "--episodes", "1"])

    def test_a_tool_agent_runs_offline_and_prints_its_summary_with_the_job_folder(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(agents, "load", lambda name: COUNTER),
            patch.object(agents, "SHOWCASE_DIR", Path(tmp) / "showcase"),
            patch.object(loop, "WORKSPACE", Path(tmp) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
        ):
            code, out, _err = _main(RUN)
            self.assertEqual(code, 0)
            (job_dir,) = (Path(tmp) / "showcase" / "counter").iterdir()
        summary = json.loads(next(line for line in out.splitlines() if line.startswith("{")))
        self.assertEqual((summary["env"], summary["policy"], summary["episodes"]), ("counter", "random", 1))
        self.assertEqual(Path(summary["job_dir"]), job_dir)

    def test_a_missing_key_is_one_line_on_stderr_and_exit_1(self) -> None:
        with (
            patch.dict(os.environ, NO_KEYS),
            patch.object(agents, "load", lambda name: COUNTER),
            patch.object(series, "optional_chat_model", lambda: None),
        ):
            code, out, err = _main(["run", "counter", "--model", "jev", "--rethink", "off", "--episodes", "1"])
        self.assertEqual((code, out), (1, ""))
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertIn("TYPESAFE_API_KEY or OPENROUTER_API_KEY", err)


STALLING = replace(COUNTER, budget=Budget(max_steps=5, timeout_s=30, stall_after=3))


class TestRethinkNeedsTheChatModel(TestCase):
    def test_rethink_on_without_the_chat_model_is_one_line_on_stderr_and_exit_1(self) -> None:
        with (
            patch.dict(os.environ, NO_KEYS),
            patch.object(agents, "load", lambda name: STALLING),
            patch.object(series, "optional_chat_model", lambda: None),
        ):
            code, out, err = _main(["run", "counter", "--model", "random", "--rethink", "on", "--episodes", "1"])
        self.assertEqual((code, out), (1, ""))
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertIn("--rethink on needs the chat model", err)

    def test_rethink_on_needs_no_chat_model_when_the_agent_never_stalls(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, NO_KEYS),
            patch.object(agents, "load", lambda name: COUNTER),
            patch.object(agents, "SHOWCASE_DIR", Path(tmp) / "showcase"),
            patch.object(loop, "WORKSPACE", Path(tmp) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
        ):
            code, _out, _err = _main([*RUN[:4], "--rethink", "on", *RUN[6:]])
        self.assertEqual(code, 0)


DECIDE = [
    "decide",
    "--state",
    '{"player_total": 18}',
    "--option",
    "hit=take a card",
    "--option",
    "stand=keep",
    "--rules",
    "stand on 17 or more",
]


class TestDecide(TestCase):
    def test_prints_the_validated_answer(self) -> None:
        transport = ScriptedTransport(choose="stand")
        built: list[str] = []

        def build(model_name: str, **kwargs: Any) -> JevModel:
            built.append(model_name)
            return JevModel(transport)

        with patch.object(cli, "build_model", build):
            code, out, _err = _main(DECIDE)
        self.assertEqual((code, built), (0, ["jev"]))
        answer = json.loads(out)
        self.assertEqual(sorted(answer), ["choice", "confidence", "ms", "probabilities"])
        self.assertEqual((answer["choice"], answer["probabilities"]["stand"], answer["ms"]), ("stand", 1.0, 9))
        question = transport.bodies[0]["questions"]["pick"]
        self.assertEqual(
            (question["criteria"], question["instructions"]["rules"]),
            ({"hit": "take a card", "stand": "keep"}, "stand on 17 or more"),
        )
        self.assertEqual(transport.bodies[0]["state"], {"player_total": 18})

    def test_the_model_flag_picks_the_model_and_the_model_is_closed(self) -> None:
        closed: list[str] = []

        class Closing(JevModel):
            async def close(self) -> None:
                closed.append(self.name)

        built: list[str] = []

        def build(model_name: str, **kwargs: Any) -> JevModel:
            built.append(model_name)
            return Closing(ScriptedTransport())

        with patch.object(cli, "build_model", build):
            code, _out, _err = _main([*DECIDE, "--model", "laya"])
        self.assertEqual((code, built, closed), (0, ["laya"], ["jev"]))
        with self.assertRaises(SystemExit):
            cli.parser().parse_args([*DECIDE, "--model", "random"])

    def test_malformed_input_is_one_line_on_stderr_and_exit_2_before_any_key_check(self) -> None:
        option = ["--option", "a=one", "--option", "b=two", "--rules", "none"]
        with patch.dict(os.environ, NO_KEYS), tempfile.TemporaryDirectory() as tmp:
            for argv in (
                ["decide", "--state", "{nope", *option],
                ["decide", "--state", f"@{tmp}/missing.json", *option],
                ["decide", "--state", "[1]", *option],
                ["decide", "--state", "{}", "--option", "nokey", "--rules", "none"],
                ["probe", f"{tmp}/missing.jsonl"],
            ):
                with self.subTest(argv=argv):
                    code, out, err = _main(argv)
                    self.assertEqual((code, out), (2, ""))
                    self.assertEqual(len(err.strip().splitlines()), 1, err)
                    self.assertNotIn("TYPESAFE_API_KEY", err)
                    if argv[0] == "decide" and argv[2] != "{}":  # the three malformed --state values
                        self.assertTrue(err.startswith("--state"), err)

    def test_state_from_a_file_and_bad_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text('{"n": 1}', encoding="utf-8")
            self.assertEqual(cli.parse_state(f"@{path}"), {"n": 1})
        with self.assertRaises(ValueError):
            cli.parse_options(["no-equals-sign"])
        with self.assertRaises(ValueError):
            cli.parse_state("[1, 2]")


class TestStdoutIsForResults(TestCase):
    """The console entry, in a subprocess: stdout holds the names or the one result line and nothing else."""

    def test_the_entry_writes_utf8_whatever_the_console_code_page(self) -> None:
        """A Windows pipe defaults to the ANSI code page; a page's superscript two or a CJK answer must still print."""
        code = "from s1a import console; console.utf8_console(); print('km² 苏黎世')"
        done = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            env={**os.environ, "PYTHONIOENCODING": "cp1252"},  # a code page that cannot encode either character
            timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr.decode("utf-8", "replace")[-500:])
        self.assertEqual(done.stdout.strip(), "km² 苏黎世".encode("utf-8"))

    def test_the_module_entry_prints_only_the_names_on_stdout(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "s1a", "list"], capture_output=True, text=True, timeout=120, check=True
        )
        self.assertEqual(done.stdout.split(), agents.names())
        self.assertNotIn("Registered", done.stdout + done.stderr)

    def test_importing_the_cli_module_alone_prints_nothing(self) -> None:
        """A host that imports ``s1a.cli`` without the console entry gets the same routing of the harness logs."""
        done = subprocess.run(
            [sys.executable, "-c", "import s1a.cli"], capture_output=True, text=True, timeout=120, check=True
        )
        self.assertEqual(done.stdout, "")

    def test_running_the_cli_module_prints_only_the_names_on_stdout(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "s1a.cli", "list"], capture_output=True, text=True, timeout=120, check=True
        )
        self.assertEqual(done.stdout.split(), agents.names())

    def test_s1a_home_from_the_dotenv_file_places_the_logs_and_the_runtime_root(self) -> None:
        """``S1A_HOME`` set in the checkout's .env, with no shell variable, is read before the log folder is fixed."""
        code = (
            "import sys; from s1a import entry; from s1a import config; print(config.HOME, file=sys.stderr); "
            "sys.argv = ['s1a', 'list']; entry.s1a()"
        )
        # a copy of the package beside its own .env stands in for the checkout
        with tempfile.TemporaryDirectory() as tmp:
            checkout, home = Path(tmp) / "checkout", Path(tmp) / "home"
            shutil.copytree(Path(cli.__file__).parent, checkout / "s1a", ignore=shutil.ignore_patterns("__pycache__"))
            (checkout / ".env").write_text(f"S1A_HOME={home}\n", encoding="utf-8")
            (checkout / "pyproject.toml").touch()  # the checkout marker console.py looks for
            done = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=checkout,
                env={name: value for name, value in os.environ.items() if name != "S1A_HOME"},
            )
            self.assertEqual(done.returncode, 0, done.stderr[-500:])
            self.assertEqual(done.stdout.split(), agents.names())
            self.assertEqual(done.stderr.strip(), str(home.resolve()))
            self.assertTrue((home / "runs" / "logs").is_dir())

    def test_outside_a_checkout_the_import_is_one_line_on_stderr_exit_1_and_writes_nothing(self) -> None:
        """A wheel install puts the package under site-packages, with no pyproject.toml above it and no evals/ data."""
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(Path(cli.__file__).parent, Path(tmp) / "s1a", ignore=shutil.ignore_patterns("__pycache__"))
            done = subprocess.run(
                [sys.executable, "-c", "import s1a.console"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=tmp,
                env={**{k: v for k, v in os.environ.items() if k != "S1A_HOME"}, "PYTHONPATH": tmp},
            )
            self.assertEqual((done.returncode, done.stdout), (1, ""))
            self.assertEqual(len(done.stderr.strip().splitlines()), 1, done.stderr)
            self.assertIn("git checkout", done.stderr)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["s1a"])

    def test_an_unusable_s1a_home_is_one_line_on_stderr_and_exit_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocked = Path(tmp) / "file"
            blocked.write_text("not a directory", encoding="utf-8")
            done = subprocess.run(
                [sys.executable, "-c", "import s1a.cli"],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "S1A_HOME": str(blocked / "x")},
            )
        self.assertEqual((done.returncode, done.stdout), (1, ""))
        self.assertEqual(len(done.stderr.strip().splitlines()), 1, done.stderr)
        self.assertIn("S1A_HOME", done.stderr)

    @skipUnless(HAVE_RLCARD, "the blackjack extra (rlcard) is not installed")
    def test_a_blackjack_series_prints_one_json_line(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "s1a", "run", "blackjack", "--model", "rule", *RUN[4:]],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
            env={**os.environ, **NO_KEYS, "MODEL_NAME": "", "OPENAI_API_KEY": "", "LLM_API_KEY": ""},
        )
        (line,) = done.stdout.strip().splitlines()
        summary = json.loads(line)
        self.assertEqual((summary["env"], summary["policy"], summary["episodes"]), ("blackjack", "basic", 1))
        self.assertTrue(Path(summary["job_dir"]).is_dir())
        self.assertTrue(Path(summary["job_dir"]).is_relative_to(os.environ["S1A_HOME"]), summary["job_dir"])
        self.assertNotIn("WARNING", done.stderr)  # the harness warnings go to the log files, never to the console
