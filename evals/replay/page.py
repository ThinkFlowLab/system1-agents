# coding: utf-8
"""The replay page: one or two trials side by side, a clock, the drawn state per step and Jev's probabilities.

A browser run draws the cast frame taken at the clock's time; every other trial draws its recorded state.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from evals.replay.trial import Trial, time_axis, views_of

DATA_TAG = '<script id="replay-data" type="application/json">'


def trial_data(trial: Trial, frames: list[str]) -> dict[str, Any]:
    return {
        "model": trial.model,
        "seed": trial.seed,
        "score": trial.score,
        "elapsed_s": trial.elapsed_s,
        "steps": trial.steps,
        "cost_usd": trial.cost_usd,
        "decisions": trial.decisions,
        "rejected": trial.rejected,
        "views": views_of(trial),
        "final_state": trial.final_state,
        "history": trial.extra.get("history") or [],
        "rethinks": trial.extra.get("rethinks") or [],
        "task": str(trial.final_state.get("task") or ""),
        "frames": frames,
        "times": time_axis(trial),
        "front": str(trial.extra.get("front") or "tool"),
    }


def copy_frames(trials: list[Trial], out_dir: Path) -> list[list[str]]:
    """Each trial's frames copied under ``out_dir/frames-<model>/``; the page references them by that relative path."""
    copied: list[list[str]] = []
    seen: dict[str, int] = {}
    for trial in trials:
        seen[trial.model] = seen.get(trial.model, 0) + 1
        folder = f"frames-{trial.model}" + (f"-{seen[trial.model]}" if seen[trial.model] > 1 else "")
        paths: list[str] = []
        if trial.frames:
            target = out_dir / folder
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True)
            for frame in trial.frames:
                shutil.copyfile(frame, target / frame.name)
                paths.append(f"{folder}/{frame.name}")
        copied.append(paths)
    return copied


MODEL_ORDER = {"jev": 0, "llm": 1}  # Jev on the left, the chat model on the right, everything else after


def ordered(trials: list[Trial]) -> list[Trial]:
    return sorted(trials, key=lambda trial: MODEL_ORDER.get(trial.model, len(MODEL_ORDER)))


def render_page(trials: list[Trial], *, out_dir: Path) -> str:
    """The page's HTML, Jev left and the chat model right; frames are copied next to it, everything else is inline."""
    trials = ordered(trials)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = copy_frames(trials, out_dir)
    data = {
        "eval": trials[0].eval_name,
        "trials": [trial_data(trial, paths) for trial, paths in zip(trials, frames)],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    title = f"{data['eval']}: {' vs '.join(t['model'] for t in data['trials'])}"
    return TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg: #f6f7f9; --panel: #fff; --ink: #1c1f26; --muted: #6b7280; --line: #e3e6eb; --accent: #2563eb;
          --jev: #0f766e; --llm: #b45309; --good: #15803d; --bad: #b91c1c; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.4 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  header { padding: 12px 16px; background: var(--panel); border-bottom: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  header h1 { font-size: 16px; margin: 0 8px 0 0; }
  header input[type=range] { width: 320px; }
  button, select { font: inherit; padding: 4px 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); cursor: pointer; }
  #clock { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 140px; }
  #stage { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; padding: 16px; }
  .trial { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; }
  .trial h2 { margin: 0; font-size: 15px; display: flex; gap: 10px; align-items: baseline; }
  .trial h2 .model { text-transform: uppercase; letter-spacing: .04em; font-size: 12px; padding: 2px 8px; border-radius: 999px; color: #fff; }
  .model.jev { background: var(--jev); } .model.llm { background: var(--llm); } .model.other { background: var(--muted); }
  .facts { color: var(--muted); font-size: 12px; display: flex; gap: 12px; flex-wrap: wrap; }
  .board { min-height: 180px; display: flex; align-items: center; justify-content: center; }
  .board img { max-width: 100%; max-height: 420px; border: 1px solid var(--line); border-radius: 6px; }
  .action { font-weight: 600; }
  .action .ms { color: var(--muted); font-weight: 400; margin-left: 6px; }
  .bars { display: grid; grid-template-columns: minmax(120px, 40%) 1fr auto; gap: 4px 8px; align-items: center; font-size: 12px; }
  .bars .label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bars .track { height: 10px; background: var(--line); border-radius: 5px; overflow: hidden; }
  .bars .fill { height: 100%; background: var(--accent); }
  .bars .chosen .label { font-weight: 700; } .bars .chosen .fill { background: var(--good); }
  .timebar { height: 6px; background: var(--line); border-radius: 3px; position: relative; }
  .timebar .done { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent); border-radius: 3px; }
  .muted { color: var(--muted); font-size: 12px; }
  .hand { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .card { width: 46px; height: 64px; border: 1px solid #999; border-radius: 6px; background: #fff; display: flex; flex-direction: column; justify-content: space-between; padding: 4px 5px; font-weight: 700; font-size: 15px; box-shadow: 0 1px 2px rgba(0,0,0,.15); }
  .card.red { color: #c81e1e; } .card.back { background: repeating-linear-gradient(45deg, #3b5bdb 0 6px, #2c48b4 6px 12px); border-color: #2c48b4; }
  .card .s { align-self: flex-end; }
  .hands { display: grid; gap: 10px; width: 100%; }
  .hands .who { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .grid { display: grid; grid-template-columns: repeat(4, 64px); gap: 6px; padding: 6px; background: #bbada0; border-radius: 8px; }
  .tile { width: 64px; height: 64px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 20px; background: #cdc1b4; color: #776e65; }
  .quiz { width: 100%; display: flex; flex-direction: column; gap: 8px; }
  .quiz .q { font-weight: 600; }
  .quiz .ans { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .quiz .ans div { padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; font-size: 13px; }
  .quiz .ans .picked { border-color: var(--accent); background: #eef2ff; }
  .quiz .ans .right { border-color: var(--good); background: #ecfdf3; } .quiz .ans .wrong { border-color: var(--bad); background: #fef2f2; }
  .transcript { width: 100%; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; max-height: 360px; overflow: auto; background: #0f172a; color: #e2e8f0; padding: 10px 12px; border-radius: 8px; }
  .transcript .a { color: #7dd3fc; } .transcript .cur { background: #1e293b; display: block; }
  pre.raw { width: 100%; font-size: 12px; white-space: pre-wrap; margin: 0; }
  footer { padding: 8px 16px 20px; color: var(--muted); font-size: 12px; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #0f1115; --panel: #171a21; --ink: #e6e8ee; --muted: #9aa3b2; --line: #2a2f3a; }
    .card { background: #fff; color: #111; }
  }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <button id="play">Play</button>
  <button id="prev">&larr; step</button>
  <button id="next">step &rarr;</button>
  <select id="speed"><option value="1">1x</option><option value="2">2x</option><option value="4" selected>4x</option><option value="8">8x</option></select>
  <input id="slider" type="range" min="0" max="1000" step="50" value="0">
  <span id="clock"></span>
</header>
<main id="stage"></main>
<footer>Times per step are the recorded decision latencies plus the loop's mean overhead per act, so both sides run on the episode's own clock.</footer>
<script id="replay-data" type="application/json">__DATA__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("replay-data").textContent);
  const maxT = Math.max(1, ...data.trials.map(t => t.times[t.times.length - 1]));
  const stage = document.getElementById("stage");
  const slider = document.getElementById("slider");
  const clock = document.getElementById("clock");
  document.getElementById("title").textContent = data.eval + " · seed " + data.trials[0].seed + " · " + data.trials.map(t => t.model).join(" vs ");
  slider.max = maxT;

  const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const money = v => v == null ? "n/a" : "$" + Number(v).toFixed(4);
  const secs = ms => (ms / 1000).toFixed(1) + " s";

  const suits = {H: "♥", D: "♦", C: "♣", S: "♠"};
  function card(code) {
    const suit = code[0], rank = code.slice(1).replace(/^T$/, "10");
    const red = suit === "H" || suit === "D";
    return '<div class="card' + (red ? " red" : "") + '"><span>' + esc(rank) + '</span><span class="s">' + (suits[suit] || suit) + '</span></div>';
  }
  function total(cards) {
    let t = 0, aces = 0;
    for (const c of cards) { const r = c.slice(1); if (r === "A") { t += 11; aces++; } else if ("TJQK".includes(r)) t += 10; else t += Number(r); }
    while (t > 21 && aces) { t -= 10; aces--; }
    return t;
  }
  function drawBlackjack(view) {
    const s = view.state || {};
    const player = s.player_cards || [], dealer = s.dealer_cards || (s.dealer_showing || []);
    const hidden = s.dealer_cards ? "" : '<div class="card back"></div>';
    return '<div class="hands"><div><div class="who">dealer ' + (s.dealer_cards ? "(" + total(dealer) + ")" : "") + '</div><div class="hand">' + dealer.map(card).join("") + hidden + '</div></div>' +
           '<div><div class="who">player (' + total(player) + (s.soft_hand ? ", soft" : "") + ')</div><div class="hand">' + player.map(card).join("") + '</div></div></div>';
  }
  const tileColors = {2:"#eee4da",4:"#ede0c8",8:"#f2b179",16:"#f59563",32:"#f67c5f",64:"#f65e3b",128:"#edcf72",256:"#edcc61",512:"#edc850",1024:"#edc53f",2048:"#edc22e"};
  function draw2048(view, trial, k) {
    if (trial.frames.length) return '<img src="' + esc(trial.frames[Math.min(k, trial.frames.length - 1)]) + '" alt="2048 board">';
    const rows = (view.state || {}).grid_rows_top_to_bottom || [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
    return '<div class="grid">' + rows.flat().map(v => '<div class="tile" style="' + (v ? "background:" + (tileColors[v] || "#3c3a32") + (v > 4 ? ";color:#f9f6f2" : "") : "") + '">' + (v || "") + '</div>').join("") + '</div>';
  }
  function drawMillionaire(view, trial, k) {
    if (trial.frames.length) return '<img src="' + esc(trial.frames[Math.min(k, trial.frames.length - 1)]) + '" alt="quiz">';
    const s = view.state || {}, d = trial.decisions[k] || {}, probs = d.probabilities || {};
    const entry = (trial.history || []).find(h => h.level === s.level);
    const answers = Object.entries(view.candidates || {}).map(([key, text]) => {
      let cls = key === d.key ? " picked" : "";
      if (entry && key === d.key) cls += entry.correct ? " right" : " wrong";
      const p = probs[key] != null ? ' <span class="muted">' + Math.round(probs[key] * 100) + '%</span>' : "";
      return '<div class="' + cls.trim() + '"><b>' + esc(key) + '</b> ' + esc(text) + p + '</div>';
    }).join("");
    return '<div class="quiz"><div class="muted">Q' + esc(s.level) + ' · ' + esc(s.difficulty) + ' · ' + esc(s.category) + ' · for $' + esc(s.prize) + '</div><div class="q">' + esc(s.question) + '</div><div class="ans">' + answers + '</div><div class="muted">winnings $' + esc(s.winnings) + ' · ' + esc(s.status) + '</div></div>';
  }
  function drawAlfworld(view, trial, k) {
    const lines = [];
    for (let j = 0; j <= k; j++) {
      const v = trial.views[j] || {}, st = v.state || {};
      const h = (trial.history && trial.history[j]) || ((st.recent_steps || []).slice(-1)[0]) || {action: j ? (trial.decisions[j - 1] || {}).key : "start", observation: ""};
      const line = (h.action === "start" ? "" : '<span class="a">&gt; ' + esc(h.action) + '</span>\n') + esc(h.observation);
      lines.push(j === k ? '<span class="cur">' + line + '</span>' : line);
    }
    const s = view.state || {};
    const frame = trial.frames.length ? '<img src="' + esc(trial.frames[Math.min(k, trial.frames.length - 1)]) + '" alt="THOR view">' : "";
    return '<div class="quiz">' + frame + '<div class="q">' + esc(trial.task || s.task) + '</div><div class="muted">holding: ' + esc(s.holding || "nothing") + ' · steps used: ' + esc(s.steps_used == null ? k : s.steps_used) + '</div><div class="transcript">' + lines.slice(-10).join("\n\n") + '</div></div>';
  }
  const stampOf = (path) => { const m = /\/t(\d+)-/.exec(path); return m ? Number(m[1]) : null; };
  function frameAt(trial, ms) {  // the latest cast frame taken at or before ms; the first before any was taken
    let pick = trial.frames.length ? trial.frames[0] : null;
    for (const frame of trial.frames) { const s = stampOf(frame); if (s == null || s > ms) break; pick = frame; }
    return pick;
  }
  function drawBrowser(view, trial, k) {
    const frame = frameAt(trial, t);
    return frame ? '<img src="' + esc(frame) + '" alt="the page">' : '<div class="muted">no frame</div>';
  }
  function drawGeneric(view) { return '<pre class="raw">' + esc(JSON.stringify(view.state, null, 1)) + '</pre>'; }
  const drawers = {blackjack: drawBlackjack, game2048: draw2048, "2048": draw2048, millionaire: drawMillionaire, alfworld: drawAlfworld};

  function bars(decision, view) {
    if (!decision) return '<div class="muted">episode over</div>';
    const entries = Object.entries(decision.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 6);
    if (!entries.length) return '<div class="muted">' + esc(decision.source === "llm" ? "chat model wrote the call" : decision.source) + '</div>';
    const candidates = (view && view.candidates) || {};
    const label = key => candidates[key] && candidates[key] !== key ? key + " · " + candidates[key] : key;
    return '<div class="bars">' + entries.map(([key, p]) => '<div class="' + (key === decision.key ? "chosen" : "") + '" style="display:contents"><span class="label" title="' + esc(label(key)) + '">' + esc(label(key)) + '</span><span class="track"><span class="fill" style="width:' + Math.round(p * 100) + '%"></span></span><span>' + Math.round(p * 100) + '%</span></div>').join("") + '</div>';
  }

  const step = data.trials.map(() => 0);
  let t = 0;
  function render() {
    stage.innerHTML = data.trials.map((trial, i) => {
      const k = step[i], view = trial.views[Math.min(k, trial.views.length - 1)] || {state: {}, candidates: {}};
      const decision = trial.decisions[k];
      const draw = trial.front === "browser" ? drawBrowser : (drawers[data.eval] || drawGeneric);
      const modelClass = trial.model === "jev" || trial.model === "llm" ? trial.model : "other";
      const badge = trial.model === "llm" ? "LLM" : "System 1 · " + trial.model;
      const action = decision ? '<span class="action">step ' + (k + 1) + ' of ' + trial.steps + ': ' + esc(decision.key) + '<span class="ms">' + esc(decision.ms) + ' ms' + (decision.confidence ? ", confidence " + decision.confidence.toFixed(2) : "") + '</span></span>'
                              : '<span class="action">final · score ' + esc(trial.score) + '</span>';
      return '<section class="trial"><h2><span class="model ' + modelClass + '">' + esc(badge) + '</span> score ' + esc(trial.score) + '</h2>' +
        '<div class="facts"><span>' + trial.elapsed_s + ' s</span><span>' + trial.steps + ' steps</span><span>' + trial.decisions.length + ' decisions</span><span>' + money(trial.cost_usd) + '</span></div>' +
        '<div class="board">' + draw(view, trial, k) + '</div>' + action + bars(decision, view) +
        '<div class="timebar"><div class="done" style="width:' + Math.min(100, 100 * trial.times[k] / maxT) + '%"></div></div><div class="muted">t = ' + secs(trial.times[k]) + ' of ' + secs(trial.times[trial.times.length - 1]) + '</div></section>';
    }).join("");
    clock.textContent = secs(t) + " / " + secs(maxT);
    slider.value = t;
  }
  const replay = {
    setStep(k) { data.trials.forEach((trial, i) => { step[i] = Math.max(0, Math.min(k, trial.steps)); }); t = Math.max(...data.trials.map((trial, i) => trial.times[step[i]])); render(); },
    setTime(ms) { t = Math.max(0, Math.min(ms, maxT)); data.trials.forEach((trial, i) => { let k = 0; while (k + 1 < trial.times.length && trial.times[k + 1] <= t) k++; step[i] = k; }); render(); },
    get time() { return t; }, get steps() { return step.slice(); }
  };
  window.replay = replay;
  slider.addEventListener("input", () => replay.setTime(Number(slider.value)));
  document.getElementById("prev").addEventListener("click", () => replay.setStep(Math.max(...step) - 1));
  document.getElementById("next").addEventListener("click", () => replay.setStep(Math.max(...step) + 1));
  let timer = null;
  const playButton = document.getElementById("play");
  playButton.addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = null; playButton.textContent = "Play"; return; }
    if (t >= maxT) replay.setTime(0);
    playButton.textContent = "Pause";
    timer = setInterval(() => { replay.setTime(t + 100 * Number(document.getElementById("speed").value)); if (t >= maxT) { clearInterval(timer); timer = null; playButton.textContent = "Play"; } }, 100);
  });
  replay.setTime(0);
})();
</script>
</body>
</html>
"""
