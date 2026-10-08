# coding: utf-8
"""The replay: trial folders in, a pair page and a GIF out. No benchmark code runs here."""

from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

from s1a.jobs import Episode, write_job
from evals.replay.page import render_page
from evals.replay.trial import Trial, read_trial, taken, time_axis, views_of

try:
    from PIL import Image
    from playwright.sync_api import Error as PlaywrightError

    from evals.replay.gif import frames_to_gif, strip_columns, strip_gif, time_ticks, write_gif

    HAVE_GIF = True
except ImportError:  # pillow or playwright missing: the report extra is not installed
    HAVE_GIF = False

try:
    from evals.replay.thor_replay import resolve_task_file

    HAVE_THOR = True
except SystemExit:  # the alfworld-visual extra is not installed
    HAVE_THOR = False


def _blackjack_episode(policy: str, seed: int, *, with_views: bool, rejected_first: bool = False) -> Episode:
    stamp = datetime.now().isoformat()
    probabilities = {"hit": 0.8, "stand": 0.2} if policy == "jev" else {}
    refused = [
        {
            "step": 1,
            "key": "double",
            "confidence": 0.0,
            "probabilities": {},
            "ms": 900,
            "source": policy,
            "accepted": False,
        }
    ]
    views = [
        {
            "state": {"player_cards": ["H5", "D6"], "player_total": 11, "soft_hand": False, "dealer_showing": ["S9"]},
            "candidates": {"hit": "hit (blackjack action)", "stand": "stand (blackjack action)"},
            "done": False,
            "score": 0.0,
        },
        {
            "state": {
                "player_cards": ["H5", "D6", "CK"],
                "player_total": 21,
                "soft_hand": False,
                "dealer_showing": ["S9"],
            },
            "candidates": {"hit": "hit (blackjack action)", "stand": "stand (blackjack action)"},
            "done": False,
            "score": 0.0,
        },
        {
            "state": {
                "player_cards": ["H5", "D6", "CK"],
                "player_total": 21,
                "soft_hand": False,
                "dealer_showing": ["S9"],
                "dealer_cards": ["S9", "H8"],
            },
            "candidates": {},
            "done": True,
            "score": 1.0,
        },
    ]
    return Episode(
        env="blackjack",
        policy=policy,
        seed=seed,
        score=1.0,
        steps=2,
        elapsed_s=1.0,
        started_at=stamp,
        finished_at=stamp,
        final_state=views[-1]["state"],
        chat_calls=0 if policy == "jev" else 2,
        chat_input_tokens=0,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        jev_input_tokens=900 if policy == "jev" else 0,
        invalid_keys=0,
        cost_usd=0.00004,
        decisions=(refused if rejected_first else [])
        + [
            {"step": 1, "key": "hit", "confidence": 0.8, "probabilities": probabilities, "ms": 300, "source": policy},
            {"step": 2, "key": "stand", "confidence": 0.9, "probabilities": probabilities, "ms": 300, "source": policy},
        ],
        views=views if with_views else [],
    )


def _page_data(html: str) -> dict:
    return json.loads(html.split('<script id="replay-data" type="application/json">', 1)[1].split("</script>", 1)[0])


def _write_pair(root: Path, *, with_views: bool = True) -> tuple[Path, Path]:
    jev = write_job("blackjack", [_blackjack_episode("jev", 0, with_views=with_views)], results_dir=root)
    llm = write_job("blackjack", [_blackjack_episode("llm", 0, with_views=with_views)], results_dir=root)
    return next(p for p in jev.iterdir() if p.is_dir()), next(p for p in llm.iterdir() if p.is_dir())


def _write_browser_pair(root: Path, *, png: bytes) -> tuple[Path, Path]:
    """A browser run per model as ``s1a run <agent> --model ...`` and ``evals.replay.cast`` leave them: answer.json, the
    model's records and two stamped frames each."""
    jev, llm = root / "jev-1", root / "llm-1"
    for folder, model_name, elapsed_ms, cost in ((jev, "jev", 3000, 0.01), (llm, "llm", 4000, 0.5)):
        (folder / "frames").mkdir(parents=True)
        answer = {
            "ok": True,
            "final": "Easy Vegetarian Spinach Lasagna, 4.6 stars, 117 ratings, 6 servings.",
            "error": None,
            "elapsed_ms": elapsed_ms,
            "usage": {"decisions": 2, "cost_usd": cost},
            "terminal": {
                "url": "https://www.allrecipes.com/recipe/229764/",
                "title": "Easy Vegetarian Spinach Lasagna",
            },
            "agent": "allrecipes",
            "model": model_name,
        }
        (folder / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
        for name in ("t00000500-0001-browser_navigate.png", "t00001500-0002-browser_run_code_unsafe.png"):
            (folder / "frames" / name).write_bytes(png)
    ticks = [
        {
            "tick": 1,
            "operation": "TYPE_TEXT",
            "target": "Search the site",
            "decision_ms": 400,
            "confidence": 0.97,
            "elapsed_ms": 1000,
            "probabilities": {"2": 1.0},
            "candidates": {"2": "Search the site"},
        },
        {
            "tick": 2,
            "operation": "DONE",
            "target": None,
            "decision_ms": 300,
            "confidence": 0.3,
            "elapsed_ms": 2000,
            "probabilities": {"DONE": 0.6, "WAIT": 0.4},
            "candidates": {"DONE": "The page shows every requirement of the task met.", "WAIT": "Give the page time."},
        },
    ]
    history = [{"action": "Search the site", "kind": "fill", "text": "vegetarian lasagna", "page_changed": True}]
    (jev / "decision_ticks.json").write_text(
        json.dumps({"report": {"history": history}, "ticks": ticks}), encoding="utf-8"
    )
    calls = [
        {"ms": 500, "input_tokens": 28, "output_tokens": 4, "tool_calls": [], "tool_args": [], "content_chars": 3},
        {
            "ms": 1000,
            "input_tokens": 10000,
            "output_tokens": 100,
            "tool_calls": ["mcp_playwright-official_browser_navigate"],
            "tool_args": ['{"url": "https://www.allrecipes.com/search?q=lasagna"}'],
            "content_chars": 0,
        },
        {
            "ms": 900,
            "input_tokens": 20000,
            "output_tokens": 300,
            "tool_calls": [],
            "tool_args": [],
            "content_chars": 400,
        },
    ]
    (llm / "chat_calls.json").write_text(json.dumps(calls), encoding="utf-8")
    return jev, llm


class TestBrowserRun(TestCase):
    def test_a_decision_model_run_reads_its_ticks_on_the_policy_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, _ = _write_browser_pair(Path(tmp), png=b"png")
            trial = read_trial(jev_dir)
        self.assertEqual((trial.eval_name, trial.model, trial.score, trial.elapsed_s), ("allrecipes", "jev", 1.0, 3.0))
        self.assertEqual((trial.steps, trial.cost_usd, trial.rejected), (2, 0.01, 0))
        self.assertEqual([d["key"] for d in trial.decisions], ["TYPE_TEXT · Search the site", "DONE"])
        self.assertEqual([d["ms"] for d in trial.decisions], [400, 300])
        self.assertEqual(trial.decisions[1]["probabilities"], {"DONE": 0.6, "WAIT": 0.4})
        self.assertEqual(trial.views[0]["candidates"], {"2": "Search the site"})
        self.assertEqual((len(trial.views), trial.views[-1]["done"]), (3, True))
        self.assertEqual(time_axis(trial), [0, 1000, 3000], "the first tick's clock, then the run's end")
        self.assertEqual([f.name[:10] for f in trial.frames], ["t00000500-", "t00001500-"])
        self.assertEqual((trial.extra["front"], len(trial.extra["history"])), ("browser", 1))

    def test_a_run_written_by_0_1_0_names_the_model_under_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, _ = _write_browser_pair(Path(tmp), png=b"png")
            answer_file = jev_dir / "answer.json"
            answer = json.loads(answer_file.read_text(encoding="utf-8"))
            answer["slot"] = answer.pop("model")
            answer_file.write_text(json.dumps(answer), encoding="utf-8")
            trial = read_trial(jev_dir)
        self.assertEqual((trial.model, trial.steps), ("jev", 2))

    def test_a_chat_model_run_counts_the_calls_that_issued_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, llm_dir = _write_browser_pair(Path(tmp), png=b"png")
            trial = read_trial(llm_dir)
        self.assertEqual((trial.model, trial.steps, trial.elapsed_s, trial.cost_usd), ("llm", 1, 4.0, 0.5))
        self.assertEqual(
            [d["key"] for d in trial.decisions], ["navigate · https://www.allrecipes.com/search?q=lasagna"]
        )
        self.assertEqual(trial.decisions[0]["source"], "llm")
        self.assertEqual(time_axis(trial), [0, 4000], "the call's latency plus the spread overhead")

    def test_the_pair_page_draws_the_frame_at_the_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, llm_dir = _write_browser_pair(Path(tmp), png=b"png")
            out = Path(tmp) / "out"
            html = render_page([read_trial(llm_dir), read_trial(jev_dir)], out_dir=out)
            copied = sorted(p.name for p in (out / "frames-jev").iterdir())
        data = _page_data(html)
        self.assertEqual([t["model"] for t in data["trials"]], ["jev", "llm"])
        self.assertEqual([t["front"] for t in data["trials"]], ["browser", "browser"])
        self.assertEqual(data["trials"][0]["frames"], [f"frames-jev/{name}" for name in copied])
        self.assertEqual(data["trials"][0]["times"], [0, 1000, 3000])
        self.assertIn("drawBrowser", html)


class TestTrial(TestCase):
    def test_read_trial_reads_the_harbor_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, _ = _write_pair(Path(tmp))
            trial = read_trial(jev_dir)
            self.assertIsInstance(trial, Trial)
            self.assertEqual((trial.eval_name, trial.model, trial.seed, trial.score), ("blackjack", "jev", 0, 1.0))
            self.assertEqual((trial.steps, trial.elapsed_s, trial.cost_usd), (2, 1.0, 0.00004))
            self.assertEqual([d["key"] for d in trial.decisions], ["hit", "stand"])
            self.assertEqual(len(trial.views), 3)
            self.assertEqual(trial.frames, [])

    def test_read_trial_lists_the_frames_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, _ = _write_pair(Path(tmp))
            frames = jev_dir / "agent" / "frames"
            frames.mkdir()
            for name in ("001.png", "000.png", "002.png"):
                (frames / name).write_bytes(b"png")
            self.assertEqual([p.name for p in read_trial(jev_dir).frames], ["000.png", "001.png", "002.png"])

    def test_time_axis_spreads_the_loop_overhead_over_the_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, _ = _write_pair(Path(tmp))
            trial = read_trial(jev_dir)
        # 1.0 s elapsed, two decisions of 300 ms: 400 ms of overhead, 200 per step.
        self.assertEqual(time_axis(trial), [0, 500, 1000])

    def test_a_rejected_llm_key_is_no_step_of_the_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = write_job(
                "blackjack", [_blackjack_episode("llm", 0, with_views=True, rejected_first=True)], results_dir=Path(tmp)
            )
            trial = read_trial(next(p for p in job.iterdir() if p.is_dir()))
            page = render_page([trial], out_dir=Path(tmp) / "out")
        self.assertEqual(([d["key"] for d in trial.decisions], trial.rejected, trial.steps), (["hit", "stand"], 1, 2))
        self.assertEqual(time_axis(trial), [0, 500, 1000], "the refused call's latency is loop overhead")
        self.assertEqual(len(taken(trial.decisions)), 2)
        data = json.loads(
            page.split('<script id="replay-data" type="application/json">', 1)[1].split("</script>", 1)[0]
        )
        self.assertEqual(
            ([d["key"] for d in data["trials"][0]["decisions"]], data["trials"][0]["rejected"]), (["hit", "stand"], 1)
        )

    def test_a_decision_whose_act_never_ran_is_no_step_of_the_replay(self) -> None:
        episode = _blackjack_episode("jev", 0, with_views=True)
        episode.decisions.append(
            {"step": 3, "key": "hit", "confidence": 0.7, "probabilities": {}, "ms": 300, "source": "jev"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            job = write_job("blackjack", [episode], results_dir=Path(tmp))
            trial = read_trial(next(p for p in job.iterdir() if p.is_dir()))
            data = _page_data(render_page([trial], out_dir=Path(tmp) / "out"))
        self.assertEqual(([d["key"] for d in trial.decisions], trial.rejected, trial.steps), (["hit", "stand"], 1, 2))
        self.assertEqual(len(trial.views), 3)
        self.assertEqual(time_axis(trial), [0, 500, 1000])
        self.assertEqual(len(data["trials"][0]["decisions"]), 2)

    def test_zero_steps_with_one_tick_replays_the_start_view_only(self) -> None:
        episode = _blackjack_episode("jev", 0, with_views=True)
        episode.steps = 0
        episode.decisions = episode.decisions[:1]
        episode.views = episode.views[:1]
        with tempfile.TemporaryDirectory() as tmp:
            job = write_job("blackjack", [episode], results_dir=Path(tmp))
            trial = read_trial(next(p for p in job.iterdir() if p.is_dir()))
            data = _page_data(render_page([trial], out_dir=Path(tmp) / "out"))
        self.assertEqual((trial.decisions, trial.rejected, trial.steps), ([], 1, 0))
        self.assertEqual(time_axis(trial), [0])
        self.assertEqual((data["trials"][0]["steps"], data["trials"][0]["times"]), (0, [0]))

    def test_a_record_without_steps_counts_every_tick(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, _ = _write_pair(Path(tmp))
            result_file = jev_dir / "result.json"
            result = json.loads(result_file.read_text(encoding="utf-8"))
            del result["agent_result"]["metadata"]["steps"]
            result_file.write_text(json.dumps(result), encoding="utf-8")
            trial = read_trial(jev_dir)
        self.assertEqual((trial.steps, [d["key"] for d in trial.decisions]), (2, ["hit", "stand"]))

    def test_blackjack_views_are_rebuilt_from_the_final_hand_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, _ = _write_pair(Path(tmp), with_views=False)
            trial = read_trial(jev_dir)
        self.assertEqual(trial.views, [])
        views = views_of(trial)
        self.assertEqual(len(views), 3)
        self.assertEqual(views[0]["state"]["player_cards"], ["H5", "D6"])
        self.assertEqual(views[1]["state"]["player_cards"], ["H5", "D6", "CK"])
        self.assertEqual(views[0]["candidates"], {"hit": "", "stand": ""})
        self.assertEqual(views[2]["state"]["dealer_cards"], ["S9", "H8"])
        self.assertTrue(views[2]["done"])


class TestPage(TestCase):
    def test_pair_page_holds_both_models_and_every_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, llm_dir = _write_pair(Path(tmp))
            html = render_page([read_trial(jev_dir), read_trial(llm_dir)], out_dir=Path(tmp) / "out")
            data = json.loads(
                html.split('<script id="replay-data" type="application/json">', 1)[1].split("</script>", 1)[0]
            )
        self.assertEqual(data["eval"], "blackjack")
        self.assertEqual([t["model"] for t in data["trials"]], ["jev", "llm"])
        self.assertEqual([t["times"] for t in data["trials"]], [[0, 500, 1000], [0, 500, 1000]])
        self.assertEqual(len(data["trials"][0]["views"]), 3)
        self.assertEqual(data["trials"][0]["decisions"][0]["probabilities"], {"hit": 0.8, "stand": 0.2})
        self.assertIn("replay.setTime", html)
        self.assertIn("replay.setStep", html)
        self.assertIn('"System 1 · " + trial.model', html)

    def test_jev_is_always_left_and_llm_right(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, llm_dir = _write_pair(Path(tmp))
            html = render_page([read_trial(llm_dir), read_trial(jev_dir)], out_dir=Path(tmp) / "out")
            data = json.loads(
                html.split('<script id="replay-data" type="application/json">', 1)[1].split("</script>", 1)[0]
            )
        self.assertEqual([t["model"] for t in data["trials"]], ["jev", "llm"])
        self.assertIn("<title>blackjack: jev vs llm</title>", html)

    def test_page_copies_frames_next_to_itself(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, llm_dir = _write_pair(Path(tmp))
            frames = jev_dir / "agent" / "frames"
            frames.mkdir()
            (frames / "000.png").write_bytes(b"png")
            out = Path(tmp) / "out"
            html = render_page([read_trial(jev_dir), read_trial(llm_dir)], out_dir=out)
            data = json.loads(
                html.split('<script id="replay-data" type="application/json">', 1)[1].split("</script>", 1)[0]
            )
            self.assertEqual(data["trials"][0]["frames"], ["frames-jev/000.png"])
            self.assertEqual(data["trials"][1]["frames"], [])
            self.assertTrue((out / "frames-jev" / "000.png").is_file())


@skipUnless(HAVE_THOR, "the alfworld-visual extra is not installed")
class TestThorTaskFile(TestCase):
    def test_a_recorded_game_file_is_looked_up_under_this_machines_alfworld_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            game = Path(tmp) / "json_2.1.1" / "valid_unseen" / "task-1" / "trial_1"
            game.mkdir(parents=True)
            (game / "traj_data.json").write_text("{}")
            recorded = "/Users/someone/.cache/alfworld/json_2.1.1/valid_unseen/task-1/trial_1/game.tw-pddl"
            with patch.dict(os.environ, {"ALFWORLD_DATA": tmp}):
                self.assertEqual(resolve_task_file(recorded), game / "traj_data.json")
            with patch.dict(os.environ, {"ALFWORLD_DATA": tmp}), self.assertRaises(SystemExit):
                resolve_task_file(recorded.replace("trial_1", "trial_9"))


@skipUnless(HAVE_GIF, "the report extra (pillow, playwright) is not installed")
class TestGif(TestCase):
    def test_time_mode_frames_are_whole_centiseconds_at_the_asked_speed(self) -> None:
        for total_ms, speed, max_frames in ((3400, 4.0, 120), (60_000, 8.0, 120), (500, 1.0, 120), (90_000, 3.0, 40)):
            tick_ms, frame_ms = time_ticks(total_ms, speed=speed, max_frames=max_frames)
            self.assertEqual(frame_ms % 10, 0)
            self.assertGreaterEqual(frame_ms, 20)
            self.assertAlmostEqual(tick_ms / frame_ms, speed, places=2)
            self.assertLessEqual(len(range(0, total_ms + tick_ms, tick_ms)), max_frames + 1)

    def test_step_mode_writes_one_frame_per_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, llm_dir = _write_pair(Path(tmp))
            out = Path(tmp) / "out"
            out.mkdir()
            page = out / "replay.html"
            page.write_text(render_page([read_trial(jev_dir), read_trial(llm_dir)], out_dir=out), encoding="utf-8")
            try:
                gif = write_gif(page, out / "replay.gif", mode="step", width=480, speed=1.0, max_frames=120)
            except PlaywrightError as exc:
                if "Executable doesn't exist" not in str(exc):
                    raise
                self.skipTest("Playwright's Chromium is not installed: playwright install chromium")
            with Image.open(gif) as image:
                self.assertEqual(image.n_frames, 3)
                self.assertEqual(image.width, 480)


@skipUnless(HAVE_GIF, "the report extra (pillow, playwright) is not installed")
class TestStripGif(TestCase):
    def test_the_strip_plays_both_runs_under_a_header_at_the_asked_speed(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (400, 200), (200, 30, 30)).save(buffer, format="PNG")
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, llm_dir = _write_browser_pair(Path(tmp), png=buffer.getvalue())
            trials = [read_trial(llm_dir), read_trial(jev_dir)]
            columns = strip_columns(trials)
            gif = strip_gif(columns, Path(tmp) / "strip.gif", speed=4.0, width=400, max_frames=10)
            with Image.open(gif) as image:
                frames, size = image.n_frames, image.size
        self.assertEqual([c.badge for c in columns], ["SYSTEM 1 · JEV", "LLM"])
        self.assertEqual(columns[0].facts, "3.0 s · 2 steps · 2 decisions · $0.0100")
        self.assertEqual([c.total_ms for c in columns], [3000, 4000])
        # 4.0 s at 4x with at most 10 frames: a 480 ms tick, 10 frames of 120 ms
        self.assertEqual(frames, 10)
        self.assertEqual(size[0], 400)
        self.assertGreater(size[1], 100, "each column is its scaled frame under the header band")

    def test_the_strip_needs_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jev_dir, _ = _write_browser_pair(Path(tmp), png=b"png")
            for frame in (jev_dir / "frames").iterdir():
                frame.unlink()
            with self.assertRaises(FileNotFoundError):
                strip_columns([read_trial(jev_dir)])


@skipUnless(HAVE_GIF, "the report extra (pillow, playwright) is not installed")
class TestFramesToGif(TestCase):
    def test_two_folders_play_side_by_side_and_the_shorter_holds_its_last_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / "jev", Path(tmp) / "llm"
            # Distinct colours per frame: Pillow merges identical consecutive GIF frames into one.
            for folder, count, channel in ((left, 2, 1), (right, 5, 0)):
                folder.mkdir()
                for index in range(count):
                    color = [20, 20, 20]
                    color[channel] = 160 + 15 * index
                    Image.new("RGB", (40, 20), tuple(color)).save(folder / f"{index:04d}.png")
            gif = frames_to_gif([left, right], Path(tmp) / "pair.gif", fps=2.0, width=80)
            with Image.open(gif) as image:
                self.assertEqual((image.n_frames, image.width, image.height), (5, 80, 20))
                image.seek(4)
                rgb = image.convert("RGB")
                self.assertEqual(rgb.getpixel((10, 10))[1] > 150, True)  # the left side still shows its last frame
                self.assertEqual(rgb.getpixel((70, 10))[0] > 150, True)
