"""Annotated video of one recorded s1a run: an input slide, the browser frames in real time with the decision band,
and an outcome slide. Usage: annotate_video.py RUN_DIR OUT.mp4 TITLE"""

import json
import os
import re
import subprocess
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

run, out_mp4, title = sys.argv[1], sys.argv[2], sys.argv[3]
W, H = 1280, 900
BAND = 120


def F(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)


def MONO(size):
    return ImageFont.truetype("C:/Windows/Fonts/consola.ttf", size)


DARK, LIGHT, RED, GREEN = (35, 37, 39), (244, 244, 245), (207, 10, 44), (21, 122, 82)

ticks = json.load(open(f"{run}/decision_ticks.json", encoding="utf-8"))["ticks"]
calls = [json.loads(line) for line in open(f"{run}/laya_calls.jsonl", encoding="utf-8")]
answer = json.load(open(f"{run}/answer.json", encoding="utf-8"))
frames = sorted(f for f in os.listdir(f"{run}/frames") if f.endswith(".png"))
times = [int(re.match(r"t(\d+)", f).group(1)) for f in frames]
work = f"{run}/annot"
os.makedirs(work, exist_ok=True)


def slide(lines, name):
    img = Image.new("RGB", (W, H + BAND), LIGHT)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 60], fill=DARK)
    d.text((24, 18), title, font=F(22, True), fill="white")
    y = 90
    for text, font, color in lines:
        d.text((40, y), text, font=font, fill=color)
        y += font.size + 10
    img.save(f"{work}/{name}")
    return name


# 1. what the model receives (the first call's shaped state and operation question)
first = calls[0]
state = first["state"]
rows = state["elements"] if isinstance(state, dict) else []
op = first["questions"].get("operation", {})
lines = [
    ("What Laya receives at step 1 (shaped by this PR)", F(30, True), DARK),
    (
        f"{len(rows)} page elements, one short line each, the page text dropped; {ticks[0].get('input_tokens')} input tokens over {len(first['questions'])} questions",
        F(20),
        DARK,
    ),
    ("", F(8), DARK),
]
for r in rows[:16]:
    lines.append(("  " + str(r)[:95], MONO(19), DARK))
if len(rows) > 16:
    lines.append((f"  ... {len(rows) - 16} more", MONO(19), DARK))
lines.append(("", F(8), DARK))
lines.append(("Question 'operation': " + textwrap.shorten(str(op.get("instructions", "")), 95), F(20, True), DARK))
lines.append(("Options: " + ", ".join(op.get("criteria", {}).keys()), MONO(19), DARK))
seq = [(slide(lines, "s0_input.png"), 9.0)]


# 2. the browser frames in real time, with the decision band
def band_text(i):
    t = ticks[0]
    if i < len(frames) - 1 and len(ticks) == 1 and times[i] < times[-1]:
        decided = i >= len(frames) - 1
    else:
        decided = True
    if not decided:
        return [("Step 1: Laya reads the page and the question...", F(26, True), "white")]
    probs = t.get("probabilities") or {}
    top = "   ".join(f"{k} {v:.2f}" for k, v in list(probs.items())[:5])
    return [
        (
            f"Step 1 decision: {t['operation']}  (confidence {t['confidence']:.2f}, {t['decision_ms'] / 1000:.1f} s, CPU)",
            F(26, True),
            "white",
        ),
        (f"operation probabilities:  {top}", MONO(20), (220, 222, 224)),
    ]


for i, f in enumerate(frames):
    shot = Image.open(f"{run}/frames/{f}").convert("RGB")
    shot = shot.resize((W, int(shot.height * W / shot.width)))
    img = Image.new("RGB", (W, H + BAND), DARK)
    d = ImageDraw.Draw(img)
    d.text((24, 18), title, font=F(22, True), fill="white")
    img.paste(shot.crop((0, 0, W, min(shot.height, H - 60))), (0, 60))
    y = H + 12
    for text, font, color in band_text(i):
        d.text((24, y), text, font=font, fill=color)
        y += font.size + 12
    name = f"f{i:03d}.png"
    img.save(f"{work}/{name}")
    dur = ((times[i + 1] if i + 1 < len(times) else times[i] + 4000) - times[i]) / 1000
    seq.append((name, max(dur, 0.1)))

# 3. the outcome
final = answer.get("terminal") or {}
seq.append(
    (
        slide(
            [
                ("Outcome: failed task", F(34, True), RED),
                (
                    f"Laya answered DONE at step 1 (confidence {ticks[0]['confidence']:.2f}); nothing was filled.",
                    F(24),
                    DARK,
                ),
                (f"Final page: {final.get('title', '')}", F(22), DARK),
                (
                    f"Status {answer.get('status')}, {len(ticks)} decision(s), {answer.get('elapsed_ms', 0) / 1000:.0f} s wall clock.",
                    F(22),
                    DARK,
                ),
                ("", F(10), DARK),
                ("Browser frames are shown in real time, no cuts; slides added before and after.", F(20), (90, 94, 98)),
            ],
            "s9_outcome.png",
        ),
        6.0,
    )
)

with open(f"{work}/concat.txt", "w") as fh:
    for name, dur in seq:
        fh.write(f"file '{name}'\nduration {dur:.3f}\n")
    fh.write(f"file '{seq[-1][0]}'\n")
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        f"{work}/concat.txt",
        "-vf",
        "format=yuv420p",
        "-r",
        "10",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "22",
        out_mp4,
    ],
    check=True,
)
print("wrote", out_mp4, "slides+frames", len(seq))
