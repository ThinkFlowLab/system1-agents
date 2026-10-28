"""Content workflow: fixture ingest, human gates, offline rules, and the Omni client contract."""

from __future__ import annotations

import contextlib
import json
import os
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import httpx

import s1a.content.__main__ as content_cli
from s1a.content.ingest import (
    LIVE_X,
    UnconfiguredXAdapter,
    load_inputs,
    load_jsonl,
    load_rss_file,
    load_rss_url,
    load_x_fixture,
    normalize_item,
    parse_feed,
    posts_to_items,
)
from s1a.content.latency import gate, omni_call_latencies, percentile
from s1a.content.omni_client import (
    OmniClient,
    check_base_url,
    decider_from_env,
    require_worker,
    worker_record,
)
from s1a.content.rules import SCREEN_CHOICES, RuleDecider, fact_state, lexical_fact, lexical_screen, screen_state
from s1a.content.workflow import GateRequired, run_workflow

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "content-workflow"
ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Desk</title>
  <entry>
    <title>Agent research note</title>
    <link href="https://example.invalid/atom-1"/>
    <id>atom-1</id>
    <summary>A research note about an agent.</summary>
  </entry>
</feed>
"""


def _keep(ident: str, **extra: object) -> dict:
    row = {
        "id": ident,
        "title": f"Model note {ident}",
        "summary": "A model note for the agent desk.",
        "source": "fixture",
        "label": "keep",
    }
    row.update(extra)
    return row


def _drop(ident: str) -> dict:
    return {
        "id": ident,
        "title": "Rain expected overnight",
        "summary": "Showers move in after midnight.",
        "source": "fixture",
        "label": "drop",
    }


class TestIngest(TestCase):
    def test_rss_file_atom_and_x_fixture_stay_offline(self) -> None:
        rss = load_rss_file(RECIPE / "sample.rss")
        self.assertEqual([row["id"] for row in rss], ["rss-keep-1", "rss-drop-1"])
        self.assertEqual(rss[0]["source"], "rss")
        self.assertNotIn("label", rss[0])
        atom = parse_feed(ATOM, origin="atom")
        self.assertEqual(atom[0]["id"], "atom-1")
        self.assertEqual(atom[0]["url"], "https://example.invalid/atom-1")
        posts = load_x_fixture(RECIPE / "x_posts.jsonl")
        self.assertEqual([row["id"] for row in posts], ["x-local-1", "x-local-2"])
        self.assertEqual(posts[0]["source"], "x")
        self.assertIn("agent", posts[0]["summary"])

    def test_rss_url_uses_the_given_client_and_rejects_a_huge_body(self) -> None:
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, content=(RECIPE / "sample.rss").read_bytes())

        client = httpx.Client(transport=httpx.MockTransport(handler))
        rows = load_rss_url("https://example.invalid/feed", client=client)
        self.assertEqual(len(rows), 2)
        self.assertEqual(seen, ["https://example.invalid/feed"])
        client.close()

        def huge(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"x" * (2_000_001))

        with self.assertRaises(ValueError):
            load_rss_url("https://example.invalid/big", client=httpx.Client(transport=httpx.MockTransport(huge)))

    def test_bad_jsonl_names_the_line_and_hides_the_text(self) -> None:
        secret = "private-ticket-body"
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            path.write_text("{not-json " + secret + "\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_jsonl(path)
        self.assertIn("line 1", str(caught.exception))
        self.assertNotIn(secret, str(caught.exception))

    def test_unknown_fields_and_the_x_adapter_do_not_travel(self) -> None:
        item = normalize_item(
            {"id": "a", "title": "Model", "summary": "A model.", "source": "fixture", "SECRET": "nope", "label": "keep"}
        )
        self.assertNotIn("SECRET", item)
        with self.assertRaises(ValueError) as caught:
            UnconfiguredXAdapter().fetch()
        self.assertEqual(str(caught.exception), LIVE_X)
        self.assertNotIn("api.x.com", json.dumps(posts_to_items([{"id": "p", "text": "A model note."}])))

    def test_labelled_sample_is_fifty_items_and_matches_the_lexical_rule(self) -> None:
        rows = load_jsonl(RECIPE / "items.jsonl")
        self.assertEqual(len(rows), 50)
        self.assertEqual(sum(row["label"] == "keep" for row in rows), 32)
        self.assertEqual(sum(row["label"] == "drop" for row in rows), 18)
        drafts = [row for row in rows if row.get("draft")]
        self.assertEqual(len(drafts), 8)
        for row in rows:
            self.assertEqual(lexical_screen(row), row["label"])
            self.assertTrue(row["url"].startswith("https://example.invalid/"))
        for row in drafts:
            self.assertEqual(lexical_fact(row["draft"]), row["draft"]["label"])
        self.assertEqual(lexical_screen({"title": "Supermodel cafe", "summary": "Downtown."}), "drop")
        self.assertEqual(lexical_screen({"title": "Chipotle sauce", "summary": "Lunch."}), "drop")
        self.assertEqual(lexical_screen({"title": "Agency hours", "summary": "Friday."}), "drop")
        self.assertEqual(lexical_screen({"title": "A model card", "summary": "Research."}), "keep")


class TestGates(TestCase):
    def _three(self) -> list[dict]:
        return [_keep("a"), _keep("b"), _keep("c"), _drop("d")]

    def test_missing_pick_stops_before_fact_check_and_publishes_nothing(self) -> None:
        class Spy(RuleDecider):
            def __init__(self) -> None:
                self.facts = 0

            def fact_check(self, item: dict) -> dict:
                self.facts += 1
                return super().fact_check(item)

        spy = Spy()
        with self.assertRaises(GateRequired) as caught:
            run_workflow(self._three(), decider=spy)
        self.assertEqual(spy.facts, 0)
        packet = caught.exception.packet
        assert packet is not None
        self.assertEqual(packet["stopped_at"], "human_pick")
        self.assertEqual(packet["human_publish"]["executed"], [])
        self.assertFalse(packet["auto_publish"])
        self.assertIsNone(packet["fact_check"])

    def test_pick_must_be_three_to_five_kept_items(self) -> None:
        rows = self._three()
        with self.assertRaises(ValueError):
            run_workflow(rows, decider=RuleDecider(), picks=["a", "b"], publish={"a": "hold", "b": "hold"})
        with self.assertRaises(ValueError):
            run_workflow(rows, decider=RuleDecider(), picks=["a", "b", "d"], publish={})
        with self.assertRaises(GateRequired) as caught:
            run_workflow(rows, decider=RuleDecider(), picks=["a", "b", "c"])
        packet = caught.exception.packet
        assert packet is not None
        self.assertEqual(packet["stopped_at"], "human_publish")
        self.assertEqual([row["id"] for row in packet["fact_check"]["records"]], ["a", "b", "c"])
        self.assertEqual(packet["human_publish"]["executed"], [])

    def test_explicit_publish_is_recorded_and_not_executed(self) -> None:
        report = run_workflow(
            self._three(),
            decider=RuleDecider(),
            picks=["a", "b", "c"],
            publish={"a": "publish", "b": "hold", "c": "publish"},
        )
        self.assertEqual(report["human_publish"]["decisions"]["a"], "publish")
        self.assertEqual(report["human_publish"]["executed"], [])
        self.assertFalse(report["auto_publish"])
        self.assertFalse(report["unsafe_skip"])
        self.assertEqual(report["stages"], ["ingest", "screen", "human_pick", "fact_check", "human_publish"])

    def test_unsafe_skip_invents_a_hold_and_still_does_not_publish(self) -> None:
        report = run_workflow(self._three(), decider=RuleDecider(), allow_unsafe_skip_gates=True)
        self.assertTrue(report["unsafe_skip"])
        self.assertEqual(report["human_pick"]["source"], "unsafe-skip")
        self.assertEqual(report["human_pick"]["ids"], ["a", "b", "c"])
        self.assertEqual(set(report["human_publish"]["decisions"].values()), {"hold"})
        self.assertEqual(report["human_publish"]["executed"], [])
        self.assertFalse(report["auto_publish"])

    def test_explicit_decisions_win_over_the_unsafe_flag_and_hooks(self) -> None:
        def boom(_rows: list[dict]) -> list[str]:
            raise AssertionError("hook should not run")

        report = run_workflow(
            self._three(),
            decider=RuleDecider(),
            picks=["a", "b", "c"],
            publish={"a": "hold", "b": "hold", "c": "hold"},
            pick_hook=boom,
            allow_unsafe_skip_gates=True,
        )
        self.assertEqual(report["human_pick"]["source"], "explicit")
        self.assertFalse(report["unsafe_skip"])

    def test_a_hook_is_a_real_gate_and_a_missing_draft_is_not_invented(self) -> None:
        def pick(kept: list[dict]) -> list[str]:
            self.assertNotIn("label", kept[0])
            return [row["id"] for row in kept]

        def publish(checked: list[dict]) -> dict[str, str]:
            return {row["id"]: "hold" for row in checked}

        class Spy(RuleDecider):
            def __init__(self) -> None:
                self.fact_ids: list[str] = []

            def fact_check(self, item: dict) -> dict:
                self.fact_ids.append(item["id"])
                if "label" in item or "label" in item["draft"]:
                    raise AssertionError("fact-check saw a label")
                return super().fact_check(item)

        draft = {
            "claims": ["The model note is ready."],
            "sources": [{"id": "s", "text": "The model note is ready for the desk."}],
            "label": "supported",
        }
        rows = [_keep("a", draft=draft), _keep("b"), _drop("c")]
        spy = Spy()
        report = run_workflow(rows, decider=spy, pick_hook=pick, publish_hook=publish)
        self.assertEqual(report["human_pick"]["source"], "hook")
        self.assertEqual(spy.fact_ids, ["a"])
        notes = {row["id"]: row.get("note") for row in report["fact_check"]["records"]}
        self.assertEqual(notes["b"], "no_draft")
        self.assertEqual(report["fact_check"]["records"][0]["choice_correct"], True)
        self.assertNotIn("SECRET", json.dumps(report))

    def test_screen_does_not_receive_labels_or_drafts(self) -> None:
        seen = []

        class Spy(RuleDecider):
            def screen(self, item: dict) -> dict:
                seen.append(item)
                return super().screen(item)

        draft = {
            "claims": ["StartLux Instant 4B is the default demo worker for this desk."],
            "sources": [{"id": "s", "text": "The default demo worker for this desk is StartLux Instant 4B."}],
        }
        rows = [_keep("a", draft=draft, SECRET="hidden"), _drop("b"), _keep("c")]
        run_workflow(
            rows,
            decider=Spy(),
            picks=["a", "c"],
            publish={"a": "hold", "c": "hold"},
        )
        self.assertTrue(seen)
        for item in seen:
            self.assertNotIn("label", item)
            self.assertNotIn("draft", item)
            self.assertNotIn("SECRET", item)
        self.assertNotIn("StartLux", screen_state(seen[0]))


class TestOmniClient(TestCase):
    def test_clef_is_refused_and_pplx_needs_its_own_url(self) -> None:
        with self.assertRaises(ValueError) as caught:
            require_worker("clef", fact_check=True)
        self.assertIn("165", str(caught.exception))
        with self.assertRaises(ValueError):
            decider_from_env({"OMNI_BASE_URL": "http://127.0.0.1:8000"}, fact_check_worker="pplx")
        with self.assertRaises(ValueError):
            check_base_url("http://127.0.0.1:8000/v1/systemone")
        with self.assertRaises(ValueError):
            require_worker("pplx", fact_check=False)

    def test_short_prefill_posts_one_question_and_hides_the_error_body(self) -> None:
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            body = json.loads(request.content)
            bodies.append((str(request.url), body))
            self.assertEqual(len(body["questions"]), 1)
            self.assertLessEqual(len(body["state"]), 1500)
            self.assertNotIn("super-secret-key", request.content.decode())
            self.assertNotIn("label", body)
            question_id, question = next(iter(body["questions"].items()))
            if "8001" in str(request.url):
                self.assertEqual(body["thinking"], {"enabled": False})
                self.assertNotIn("think_mode", body)
                self.assertEqual(body["model"], "perplexity-ai/pplx-decider-v1.1-27b")
                choice, level, keys = "supported", 2, list(question["criteria"])
            else:
                self.assertEqual(body["think_mode"], "never")
                self.assertNotIn("thinking", body)
                self.assertEqual(body["model"], "StartLux-Decision-v1.1-4B-Instant")
                choice, level, keys = "keep", 1, list(question["criteria"])
            if question["type"] == "choice":
                answer = {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": {key: (1.0 if key == choice else 0.0) for key in keys},
                }
            else:
                answer = {
                    "type": "score",
                    "score": float(level),
                    "probabilities": {str(i): (1.0 if i == level else 0.0) for i in range(len(question["criteria"]))},
                }
            return httpx.Response(200, json={"model": body["model"], "answers": {question_id: answer}})

        env = {
            "OMNI_BASE_URL": "http://127.0.0.1:8000",
            "OMNI_FACTCHECK_BASE_URL": "http://127.0.0.1:8001",
            "OMNI_API_KEY": "super-secret-key",
        }
        decider = decider_from_env(env, fact_check_worker="pplx", handler=handler)
        self.assertEqual(decider.screen_client.health()["status"], "ok")
        draft = {
            "claims": ["StartLux Instant 4B is the default demo worker for this desk."],
            "sources": [{"id": "s", "text": "The default demo worker for this desk is StartLux Instant 4B."}],
            "label": "supported",
        }
        rows = [
            _keep("a", draft=draft, SECRET="super-secret-key"),
            _drop("b"),
            _keep("c"),
        ]
        try:
            report = run_workflow(
                rows,
                decider=decider,
                picks=["a", "b", "c"],
                publish={"a": "publish", "b": "hold", "c": "hold"},
                workers={
                    "screen": worker_record(
                        decider.screen_client.profile, decider.screen_client.base_url, decider.screen_client.model
                    ),
                    "fact_check": worker_record(
                        decider.fact_client.profile, decider.fact_client.base_url, decider.fact_client.model
                    ),
                },
            )
        finally:
            decider.close()
        self.assertEqual(report["backend"], "omni")
        self.assertEqual(report["workers"]["screen"]["pr"], 164)
        self.assertEqual(report["workers"]["fact_check"]["pr"], 166)
        self.assertFalse(report["workers"]["fact_check"]["hot_path"])
        self.assertNotIn("super-secret-key", json.dumps(report))
        self.assertEqual(report["human_publish"]["executed"], [])
        self.assertEqual(report["human_publish"]["decisions"]["a"], "publish")
        self.assertFalse(report["latency_gate"]["attached"])
        screen_calls = [url for url, _ in bodies if ":8000/" in url and url.endswith("/v1/systemone")]
        fact_calls = [url for url, _ in bodies if ":8001/" in url]
        self.assertEqual(len(screen_calls), 6)
        self.assertEqual(len(fact_calls), 2)
        self.assertEqual(report["screen"]["choice_correct"], 2)
        self.assertEqual(report["fact_check"]["records"][0]["choice"], "supported")
        self.assertEqual(report["fact_check"]["records"][1]["note"], "no_draft")
        self.assertTrue(report["fact_check"]["records"][0]["calls"])

    def test_a_failed_status_does_not_echo_the_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "SECRET-BODY"})

        client = OmniClient(
            "http://127.0.0.1:8000",
            profile=require_worker("startlux", fact_check=False),
            transport=httpx.MockTransport(handler),
        )
        try:
            with self.assertRaises(RuntimeError) as caught:
                client.ask(
                    "Title: Model\nSummary: A model.\nSource: fixture",
                    "screen",
                    {"type": "choice", "instructions": "Keep or drop.", "criteria": dict(SCREEN_CHOICES)},
                )
        finally:
            client.close()
        self.assertNotIn("SECRET-BODY", str(caught.exception))
        self.assertIn("HTTP 500", str(caught.exception))


class TestWorkflowFixture(TestCase):
    def test_recipe_files_walk_all_five_stages_on_the_rule_backend(self) -> None:
        rows = load_jsonl(RECIPE / "items.jsonl")
        picks = json.loads((RECIPE / "picks.json").read_text(encoding="utf-8"))["ids"]
        publish = json.loads((RECIPE / "publish.json").read_text(encoding="utf-8"))["decisions"]
        report = run_workflow(rows, decider=RuleDecider(), picks=picks, publish=publish)
        self.assertEqual(report["screen"]["total"], 50)
        self.assertEqual(report["screen"]["kept"], 32)
        self.assertEqual(report["screen"]["choice_accuracy"], 1.0)
        self.assertEqual(report["screen"]["score_accuracy"], 1.0)
        self.assertEqual(report["human_pick"]["ids"], ["n01", "n02", "n03", "n04"])
        self.assertEqual(
            [row["choice"] for row in report["fact_check"]["records"]],
            ["supported", "unsupported", "needs_human", "supported"],
        )
        self.assertEqual(report["fact_check"]["choice_accuracy"], 1.0)
        self.assertEqual(report["fact_check"]["score_accuracy"], 1.0)
        self.assertEqual(report["human_publish"]["decisions"], publish)
        self.assertIn("publish", report["human_publish"]["decisions"].values())
        self.assertEqual(report["human_publish"]["executed"], [])
        self.assertFalse(report["auto_publish"])
        self.assertEqual(omni_call_latencies(report), [])
        state = fact_state(rows[0])
        self.assertNotIn("Label:", state)
        self.assertLessEqual(len(state), 1500)

    def test_local_rss_and_x_fixtures_can_be_combined_without_a_network(self) -> None:
        items = load_inputs(rss_files=[RECIPE / "sample.rss"], x_files=[RECIPE / "x_posts.jsonl"])
        report = run_workflow(items, decider=RuleDecider(), allow_unsafe_skip_gates=True)
        self.assertEqual(report["ingest"]["count"], 4)
        self.assertEqual(report["ingest"]["sources"], {"rss": 2, "x": 2, "fixture": 0})
        self.assertEqual(report["screen"]["kept"], 2)
        self.assertEqual(report["human_publish"]["executed"], [])
        self.assertTrue(report["unsafe_skip"])
        self.assertEqual(set(report["human_publish"]["decisions"].values()), {"hold"})


class TestCli(TestCase):
    def test_recipe_command_and_the_latency_hook(self) -> None:
        out = self._run(
            [
                "run",
                "--items",
                str(RECIPE / "items.jsonl"),
                "--picks",
                str(RECIPE / "picks.json"),
                "--publish",
                str(RECIPE / "publish.json"),
                "--backend",
                "rule",
            ]
        )
        self.assertEqual(out["code"], 0, out["err"])
        report = json.loads(out["stdout"])
        self.assertEqual(report["screen"]["choice_correct"], 50)
        self.assertEqual(report["human_publish"]["executed"], [])
        self.assertFalse(report["latency_gate"]["attached"])
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(out["stdout"], encoding="utf-8")
            idle = self._run(["latency-report", "--report", str(path)])
            self.assertEqual(idle["code"], 0, idle["err"])
            self.assertFalse(json.loads(idle["stdout"])["enforced"])
            blocked = self._run(["latency-report", "--report", str(path), "--p95-max-ms", "10", "--min-samples", "100"])
            self.assertEqual(blocked["code"], 1)
            body = json.loads(blocked["stdout"])
            self.assertFalse(body["passed"])
            self.assertEqual(body["summary"]["n"], 0)

    def test_gates_cannot_be_skipped_without_the_unsafe_flag(self) -> None:
        refused = self._run(["run", "--items", str(RECIPE / "items.jsonl"), "--backend", "rule"])
        self.assertEqual(refused["code"], 2)
        self.assertIn("human pick gate", refused["err"])
        self.assertIn("allow-unsafe-skip-gates", refused["err"])
        skipped = self._run(
            ["run", "--items", str(RECIPE / "items.jsonl"), "--backend", "rule", "--allow-unsafe-skip-gates"]
        )
        self.assertEqual(skipped["code"], 0, skipped["err"])
        report = json.loads(skipped["stdout"])
        self.assertTrue(report["unsafe_skip"])
        self.assertEqual(report["human_publish"]["executed"], [])
        self.assertEqual(set(report["human_publish"]["decisions"].values()), {"hold"})

    def test_omni_without_a_worker_fails_before_any_publish(self) -> None:
        refused = self._run(
            [
                "run",
                "--rss",
                str(RECIPE / "sample.rss"),
                "--picks",
                str(RECIPE / "picks.json"),
                "--publish",
                str(RECIPE / "publish.json"),
                "--backend",
                "omni",
            ],
            env={"OMNI_BASE_URL": "http://127.0.0.1:9", "OMNI_API_KEY": "super-secret-key", "OMNI_TIMEOUT_S": "0.2"},
        )
        self.assertEqual(refused["code"], 1)
        self.assertNotIn("super-secret-key", refused["err"])
        self.assertNotIn("publish", refused["stdout"])

    def _run(self, argv: list[str], env: dict[str, str] | None = None) -> dict[str, str | int]:
        merged = {"OMNI_BASE_URL": "", "OMNI_FACTCHECK_BASE_URL": "", "OMNI_API_KEY": "", "OMNI_TIMEOUT_S": ""}
        if env:
            merged.update(env)
        stdout, stderr = StringIO(), StringIO()
        with (
            patch.dict(os.environ, merged, clear=False),
            patch.object(content_cli.sys.stdin, "isatty", return_value=False),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = content_cli.main(argv)
        return {"code": code, "stdout": stdout.getvalue(), "err": stderr.getvalue()}


class TestLatencyMath(TestCase):
    def test_nearest_rank_and_an_unattached_gate(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(percentile([1, 2, 3, 4, 5], 95), 5)
        idle = gate([1.0, 2.0])
        self.assertFalse(idle["enforced"])
        self.assertIsNone(idle["passed"])
        closed = gate([], p95_max_ms=10, min_samples=5)
        self.assertFalse(closed["passed"])
        self.assertIn("Public231 C1", closed["procedure"])
