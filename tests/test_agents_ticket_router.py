"""Ticket privacy, deterministic batches, scoring, and the shared tool-loop contract."""

from __future__ import annotations

import importlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from s1a import run as agents
from s1a.run import started_runner
from s1a.tool import loop, series


def router():
    assert "ticket_router" in agents.names(), "ticket_router must be discoverable by CLI and MCP"
    return importlib.import_module("s1a.agents.ticket_router")


TICKETS = [
    {"id": "a", "title": "Delivery", "description": "Track my parcel", "label": "logistics", "note": "SECRET_A"},
    {"id": "b", "title": "Charge", "description": "Payment failed", "label": "payment", "order_status": "unpaid"},
    {"id": "c", "title": "After sales", "description": "Request a refund", "label": "returns"},
    {"id": "d", "title": "Access", "description": "Cannot log in", "label": "account"},
    {"id": "e", "title": "Help", "description": "Please transfer me to a human", "label": "human"},
]


class TestTicketRouterEnv(IsolatedAsyncioTestCase):
    async def test_only_current_public_ticket_fields_are_observed(self):
        module = router()
        env = module.TicketRouterEnv(seed=0, tickets=TICKETS)
        await env.reset()
        seen = []
        expected = {row["id"]: row["label"] for row in TICKETS}
        while not env.done:
            state = await env.observe()
            self.assertEqual(set(state), {"ticket", "progress"})
            ticket = state["ticket"]
            self.assertLessEqual(set(ticket), {"id", "title", "description", "order_status"})
            self.assertEqual(state["progress"], {"ticket_id": ticket["id"]})
            self.assertNotIn("SECRET_A", json.dumps(state))
            seen.append(ticket["id"])
            # Caller mutations must not corrupt observations or the candidate set.
            ticket["description"] = "modified"
            self.assertNotEqual((await env.observe())["ticket"]["description"], "modified")
            candidates = await env.candidates()
            self.assertEqual(set(candidates), {"logistics", "payment", "returns", "account", "human"})
            candidates.clear()
            await env.step(expected[ticket["id"]])
        self.assertEqual(set(seen), {"a", "b", "c", "d", "e"})
        self.assertEqual(env.score, 5)
        self.assertEqual(await env.candidates(), {})
        self.assertIsNone((await env.observe())["ticket"])
        self.assertEqual(env.report()["accuracy"], 1)

    async def test_seed_and_reset_reproduce_order_and_clear_score(self):
        module = router()

        async def order(env):
            await env.reset()
            ids = []
            while not env.done:
                ids.append((await env.observe())["ticket"]["id"])
                await env.step("human")
            return ids

        env = module.TicketRouterEnv(seed=0, tickets=TICKETS)
        first = await order(env)
        self.assertEqual(first, await order(env))
        self.assertEqual(first, await order(module.TicketRouterEnv(seed=0, tickets=TICKETS)))
        self.assertNotEqual(first, await order(module.TicketRouterEnv(seed=1, tickets=TICKETS)))
        await env.reset()
        self.assertEqual(env.score, 0)
        self.assertEqual(env.report()["processed"], 0)

    async def test_bad_action_cannot_consume_ticket_and_completed_batch_rejects_steps(self):
        env = router().TicketRouterEnv(seed=0, tickets=TICKETS[:1])
        await env.reset()
        before = await env.observe()
        with self.assertRaises(ValueError):
            await env.step("refund")
        self.assertEqual(before, await env.observe())
        self.assertEqual(env.score, 0)
        await env.step("logistics")
        with self.assertRaises(ValueError):
            await env.step("logistics")
        self.assertEqual(env.score, 1)

    async def test_partial_batch_metrics_count_unprocessed_as_incorrect(self):
        env = router().TicketRouterEnv(seed=0, tickets=TICKETS)
        await env.reset()
        # Random(0) puts fixture c first, but derive the action from its private test label, not env.report.
        current_id = (await env.observe())["ticket"]["id"]
        expected = {row["id"]: row["label"] for row in TICKETS}
        await env.step(expected[current_id])
        report = env.report()
        self.assertEqual((report["total"], report["processed"], report["correct"]), (5, 1, 1))
        self.assertEqual(report["accuracy"], 0.2)
        self.assertEqual(report["coverage"], 0.2)
        self.assertEqual(report["per_class_recall"][expected[current_id]], 1)
        self.assertEqual(sum(report["per_class_recall"].values()), 1)
        self.assertEqual(len(report["unprocessed_ids"]), 4)

    async def test_custom_batch_size_is_reproducible(self):
        module = router()
        env = module.TicketRouterEnv(seed=2, tickets=TICKETS, batch_size=2)
        await env.reset()
        self.assertEqual(env.report()["total"], 2)
        ids = env.report()["ticket_ids"]
        await env.reset()
        self.assertEqual(env.report()["ticket_ids"], ids)

    async def test_wrong_valid_route_advances_without_reward_and_recall_uses_class_support(self):
        tickets = [
            {"id": "a", "title": "Delivery", "description": "Track my parcel", "label": "logistics"},
            {"id": "b", "title": "Delivery", "description": "Check the tracking number", "label": "logistics"},
        ]
        env = router().TicketRouterEnv(seed=0, tickets=tickets)
        await env.reset()
        first = (await env.observe())["ticket"]["id"]
        await env.step("payment")
        self.assertEqual(env.score, 0)
        self.assertNotEqual((await env.observe())["ticket"]["id"], first)
        await env.step("logistics")
        report = env.report()
        self.assertTrue(env.done)
        self.assertEqual((env.score, report["processed"], report["accuracy"]), (1, 2, 0.5))
        self.assertEqual(report["per_class_recall"]["logistics"], 0.5)
        self.assertIsNone(report["per_class_recall"]["payment"])

    async def test_answer_labels_do_not_change_observations(self):
        module = router()
        a = module.TicketRouterEnv(seed=0, tickets=TICKETS)
        b = module.TicketRouterEnv(seed=0, tickets=[{**row, "label": "human"} for row in TICKETS])
        await a.reset()
        await b.reset()
        while not a.done:
            self.assertEqual(await a.observe(), await b.observe())
            self.assertEqual(await a.candidates(), await b.candidates())
            await a.step("logistics")
            await b.step("logistics")


class TestTicketRouterData(TestCase):
    def test_rejects_ambiguous_or_malformed_labelled_data(self):
        module = router()
        for rows in (
            [],
            [TICKETS[0], TICKETS[0]],
            [{**TICKETS[0], "label": "refund"}],
            [{**TICKETS[0], "description": ""}],
            [{**TICKETS[0], "order_status": {"label": "logistics"}}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                module.TicketRouterEnv(seed=0, tickets=rows)
        for size in (0, -1, 6):
            with self.subTest(size=size), self.assertRaises(ValueError):
                module.TicketRouterEnv(seed=0, tickets=TICKETS, batch_size=size)

    def test_probe_and_evaluation_are_separate_and_all_queues_are_covered(self):
        module = router()
        rows = module.load_tickets(module.DEFAULT_DATASET)
        probe_path = Path(__file__).resolve().parents[1] / "evals/ticket_router/probe.jsonl"
        probe = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(probe), 12)
        self.assertEqual(len(rows), 30)
        self.assertEqual(Counter(r["label"] for r in rows), dict.fromkeys(module.QUEUES, 6))
        self.assertEqual({r["label"] for r in rows}, set(module.QUEUES))
        self.assertFalse({r["description"] for r in rows} & {r["state"]["ticket"]["description"] for r in probe})
        self.assertTrue(all(p["rules"] == module.RULES and p["options"] == module.QUEUES for p in probe))

    def test_keywords_baseline_handles_unique_matches_and_ambiguity(self):
        module = router()
        for description, want in [
            ("Track my parcel", "logistics"),
            ("Duplicate charge", "payment"),
            ("Request a refund", "returns"),
            ("Cannot log in", "account"),
            ("A human should track my parcel", "human"),
            ("Track my parcel and issue a refund", "human"),
            ("Something is wrong", "human"),
            ("PAYMENT FAILED", "payment"),
            ("The furniture arrived", "human"),
        ]:
            with self.subTest(description=description):
                self.assertEqual(
                    module.keyword_rule({"ticket": {"title": "Ticket", "description": description}}, module.QUEUES),
                    want,
                )


class TestTicketRouterThroughTheLoop(IsolatedAsyncioTestCase):
    async def _play(self, model_name, max_steps=5):
        module = router()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "tickets.jsonl"
            data.write_text("\n".join(json.dumps(row) for row in TICKETS), encoding="utf-8")
            args = series.parser(module.SPEC).parse_args(
                [
                    "--model",
                    model_name,
                    "--rethink",
                    "off",
                    "--episodes",
                    "3",
                    "--dataset",
                    str(data),
                    "--max-steps",
                    str(max_steps),
                ]
            )
            with (
                patch.object(loop, "WORKSPACE", root / "ws"),
                patch.object(series, "optional_chat_model", lambda: None),
            ):
                async with started_runner():
                    result = await series.play(module.SPEC, args, results_dir=root / "results")
            # Filesystem enumeration has no guaranteed order; exercise different orders for the two models.
            paths = sorted(Path(result["job_dir"]).glob("*/agent/episode.json"), reverse=model_name == "random")
            episodes = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
            return result, episodes

    async def test_rule_and_random_process_identical_batches_through_shared_loop(self):
        rule, rule_episodes = await self._play("rule")
        self.assertEqual((rule["mean_score"], rule["mean_steps"], rule["errors"], rule["invalid_keys"]), (5, 5, 0, 0))
        chance, random_episodes = await self._play("random")
        self.assertEqual(
            (chance["episodes"], chance["mean_steps"], chance["invalid_keys"], chance["errors"]), (3, 5, 0, 0)
        )
        self.assertEqual((len(rule_episodes), len(random_episodes)), (3, 3))
        rule_by_seed = {episode["extra"]["ticket_router"]["seed"]: episode for episode in rule_episodes}
        random_by_seed = {episode["extra"]["ticket_router"]["seed"]: episode for episode in random_episodes}
        self.assertEqual(set(rule_by_seed), {0, 1, 2})
        self.assertEqual(set(random_by_seed), set(rule_by_seed))
        for seed, left in rule_by_seed.items():
            right = random_by_seed[seed]
            a, b = left["extra"]["ticket_router"], right["extra"]["ticket_router"]
            self.assertEqual(a["ticket_ids"], b["ticket_ids"])
            self.assertEqual(a["batch_sha256"], b["batch_sha256"])
            self.assertEqual(a["accuracy"], 1)
            self.assertEqual(b["processed"], 5)
            self.assertEqual(left["extra"]["rethinks"], [])
            for view in left["views"]:
                self.assertNotIn("label", json.dumps(view["state"]))
                self.assertNotIn("SECRET_A", json.dumps(view["state"]))

    async def test_step_budget_is_reported_as_partial_not_full_success(self):
        result, episodes = await self._play("rule", max_steps=2)
        self.assertEqual(result["mean_steps"], 2)
        for episode in episodes:
            report = episode["extra"]["ticket_router"]
            self.assertEqual((report["processed"], report["accuracy"], report["coverage"]), (2, 0.4, 0.4))

    def test_rethink_on_is_rejected_for_independent_tickets(self):
        module = router()
        flags = series.parser(module.SPEC).parse_args(["--model", "rule", "--rethink", "on", "--episodes", "1"])
        with self.assertRaises(ValueError):
            module.make_series(flags)
