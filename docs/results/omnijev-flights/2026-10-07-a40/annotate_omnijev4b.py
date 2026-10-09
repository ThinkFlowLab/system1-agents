"""Annotated video of one OmniJev-4B GPU run: intro slide, every frame in real time with the latest decision under it,
outcome slide. Usage: annotate_omnijev4b.py RUN_DIR OUT.mp4"""

import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

run, out_mp4 = sys.argv[1], sys.argv[2]
TITLE = "s1a flights agent, --model omnijev (OmniJev-4B v1.1, in process, NVIDIA A40) - Google Flights, Zurich to London one way"
W, H, BAND = 1280, 900, 130
DARK, LIGHT, RED, GREEN, GREY = (35, 37, 39), (244, 244, 245), (207, 10, 44), (21, 122, 82), (220, 222, 224)


def F(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)


def MONO(size):
    return ImageFont.truetype("C:/Windows/Fonts/consola.ttf", size)


rec = json.load(open(f"{run}/decision_ticks.json", encoding="utf-8"))
ticks, report = rec["ticks"], rec["report"]
answer = json.load(open(f"{run}/answer.json", encoding="utf-8"))
frames = sorted(f for f in os.listdir(f"{run}/frames") if f.endswith(".png"))
times = [int(re.match(r"t(\d+)", f).group(1)) for f in frames]
work = f"{run}/annot"
os.makedirs(work, exist_ok=True)
seq = []


def save(img, dur):
    name = f"a{len(seq):03d}.png"
    img.save(f"{work}/{name}")
    seq.append((name, dur))


def header(img):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 60], fill=DARK)
    d.text((24, 20), TITLE[:125], font=F(18, True), fill="white")
    return d


def slide(lines):
    img = Image.new("RGB", (W, H + BAND), LIGHT)
    d = header(img)
    y = 90
    for text, font, color in lines:
        d.text((40, y), text, font=font, fill=color)
        y += font.size + 12
    return img


def describe(t):
    what = t["operation"]
    if t.get("target"):
        what += f' "{t["target"][:60]}"'
    if t.get("text"):
        what += f' text "{t["text"]}"'
    return what


save(
    slide(
        [
            ("OmniJev-4B on Google Flights, on GPU", F(30, True), DARK),
            ("Task: one-way flight from Zurich to London.", F(22), DARK),
            ("At each step OmniJev gets a screenshot of the page and the choice questions", F(22), DARK),
            ("(which operation, which element, which text) and answers with probabilities.", F(22), DARK),
            ("", F(10), DARK),
            ("Below each frame: the latest decision, step number, confidence and decision time.", F(20), (90, 94, 98)),
            ("Browser frames in real time, no cuts.", F(20), (90, 94, 98)),
        ]
    ),
    7.0,
)

for i, f in enumerate(frames):
    done = [t for t in ticks if t["elapsed_ms"] <= times[i] + 300]
    img = Image.new("RGB", (W, H + BAND), DARK)
    s = Image.open(f"{run}/frames/{f}").convert("RGB")
    s = s.resize((W, int(s.height * W / s.width)))
    img.paste(s.crop((0, 0, W, min(s.height, H - 60))), (0, 60))
    d = header(img)
    if done:
        t = done[-1]
        color = "white" if t["tick"] <= 9 else (255, 190, 120)
        d.text((24, H + 12), f"Step {t['tick']}: {describe(t)}", font=F(24, True), fill=color)
        d.text(
            (24, H + 50),
            f"confidence {t['confidence']:.2f}, {t['decision_ms'] / 1000:.1f} s on the A40   |   run clock {times[i] / 1000:.0f} s",
            font=MONO(18),
            fill=GREY,
        )
        if t["tick"] >= 10:
            d.text(
                (24, H + 82),
                "the form is complete, but the Search button is never clicked",
                font=F(18),
                fill=(255, 190, 120),
            )
    else:
        d.text((24, H + 12), "Opening Google Flights, first decision...", font=F(24, True), fill="white")
    nxt = times[i + 1] if i + 1 < len(times) else times[i] + 3000
    save(img, max((nxt - times[i]) / 1000, 0.1))

br = json.loads(answer["final"])["browser_result"]
save(
    slide(
        [
            ("Outcome: failed task (no search results)", F(32, True), RED),
            ("Steps 1-9 (47 s): Zurich, London, one way, Sun Nov 8, Done: the form is filled correctly.", F(22), GREEN),
            ("Steps 10-25: it never clicks Search; it presses Enter, scrolls, and leaves the form.", F(22), DARK),
            (
                f"openJiuwen ended the run after {report['decisions']} decisions ({', '.join(br['blockers'])}), {answer['elapsed_ms'] / 1000:.0f} s wall clock.",
                F(22),
                DARK,
            ),
            (
                f"Median decision time {report['median_decision_ms'] / 1000:.1f} s on one A40 (136 s for the 0.8B on CPU).",
                F(22),
                DARK,
            ),
            ("", F(10), DARK),
            (
                "3 runs recorded, all kept: the same outcome in each (runs 1 and 3 made identical decisions).",
                F(20),
                (90, 94, 98),
            ),
        ]
    ),
    8.0,
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
print("wrote", out_mp4, len(seq), "images", round(sum(d for _, d in seq)), "s")
