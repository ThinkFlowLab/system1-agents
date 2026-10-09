"""Video of the PR #36 deadline runs on a real server: the default 5 s, S1A_DECISION_TIMEOUT_S=30 and =1, one after
the other, each browser frame in real time with the latest decision (or the wait) under it.
Usage: annotate_timeout.py OUT_DIR OUT.mp4"""

import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

root, out_mp4 = sys.argv[1], sys.argv[2]
TITLE = os.environ.get(
    "VIDEO_TITLE",
    "s1a flights --model jev -> system1-omni frontend (omni-jev) -> Open-Jev-9B, one NVIDIA A40 - Zurich to London one way",
)
W, H, BAND = 1280, 900, 130
DARK, LIGHT, RED, GREEN, GREY = (35, 37, 39), (244, 244, 245), (207, 10, 44), (21, 122, 82), (220, 222, 224)
ARMS = [
    ("default-5s", 5, "default deadline (S1A_DECISION_TIMEOUT_S unset): 5 s"),
    ("timeout-30s", 30, "S1A_DECISION_TIMEOUT_S=30"),
    ("timeout-1s", 1, "S1A_DECISION_TIMEOUT_S=1 (a deadline the server always exceeds)"),
]


def F(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)


def MONO(size):
    return ImageFont.truetype("C:/Windows/Fonts/consola.ttf", size)


work = f"{root}/annot"
os.makedirs(work, exist_ok=True)
seq = []


def save(img, dur):
    name = f"a{len(seq):03d}.png"
    img.save(f"{work}/{name}")
    seq.append((name, dur))


def header(img):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 60], fill=DARK)
    d.text((24, 20), TITLE[:128], font=F(17, True), fill="white")
    return d


def slide(lines, dur):
    img = Image.new("RGB", (W, H + BAND), LIGHT)
    d = header(img)
    y = 90
    for text, font, color in lines:
        d.text((40, y), text, font=font, fill=color)
        y += font.size + 12
    save(img, dur)


def frame(path, lines, dur):
    img = Image.new("RGB", (W, H + BAND), DARK)
    s = Image.open(path).convert("RGB")
    s = s.resize((W, int(s.height * W / s.width)))
    img.paste(s.crop((0, 0, W, min(s.height, H - 60))), (0, 60))
    d = header(img)
    y = H + 12
    for text, font, color in lines:
        d.text((24, y), text, font=font, fill=color)
        y += font.size + 12
    save(img, dur)


slide(
    [
        ("PR #36: the decision deadline on a real server", F(30, True), DARK),
        ("Same task, same server, three runs one after the other; only the deadline changes.", F(22), DARK),
        *[
            (line, F(20), DARK)
            for line in os.environ.get(
                "VIDEO_PATH",
                "Path: s1a (main at 22e685b, the #36 merge) -> system1-omni's Rust frontend omni-jev (99865743)|-> Open-Jev's reference server (3308a15) with Open-Jev-9B, on one NVIDIA A40 46 GB.",
            ).split("|")
        ],
        (
            "Open-Jev-27B through system1-omni's native worker does not fit this GPU (cudaMalloc out of memory).",
            F(20),
            (90, 94, 98),
        ),
        ("", F(10), DARK),
        ("Browser frames in real time, no cuts; a slide before each run and at the end.", F(20), (90, 94, 98)),
    ],
    8.0,
)

summary = []
for arm, deadline, label in ARMS:
    d = f"{root}/{arm}"
    answer = json.load(open(f"{d}/answer.json", encoding="utf-8"))
    ticks = json.load(open(f"{d}/decision_ticks.json", encoding="utf-8"))["ticks"]
    frames = sorted(f for f in os.listdir(f"{d}/frames") if f.endswith(".png"))
    times = [int(re.match(r"t(\d+)", f).group(1)) for f in frames]
    slide([(f"Run: {label}", F(28, True), DARK)], 3.0)
    for i, f in enumerate(frames):
        done = [t for t in ticks if t["elapsed_ms"] <= times[i] + 300]
        if done:
            t = done[-1]
            what = (
                t["operation"]
                + (f' "{t["target"][:40]}"' if t.get("target") else "")
                + (f' text "{t["text"]}"' if t.get("text") else "")
            )
            lines = [
                (f"Decision {t['tick']}: {what}", F(24, True), "white"),
                (
                    f"answered in {t['decision_ms'] / 1000:.1f} s (deadline {deadline} s)   |   run clock {times[i] / 1000:.0f} s",
                    MONO(18),
                    GREY,
                ),
            ]
        else:
            lines = [
                ("Reading the page, first decision asked...", F(24, True), "white"),
                (f"deadline {deadline} s   |   run clock {times[i] / 1000:.0f} s", MONO(18), GREY),
            ]
        nxt = times[i + 1] if i + 1 < len(times) else answer["elapsed_ms"]
        frame(f"{d}/frames/{f}", lines, max((nxt - times[i]) / 1000, 0.2))
    last = f"{d}/frames/{frames[-1]}"
    if answer["ok"]:
        frame(
            last,
            [
                (
                    f"The run ended after {len(ticks)} decisions, none cut by the deadline ({answer['elapsed_ms'] / 1000:.0f} s)",
                    F(24, True),
                    (140, 230, 180),
                ),
                ("decision times " + ", ".join(f"{t['decision_ms'] / 1000:.0f}" for t in ticks) + " s", MONO(18), GREY),
            ],
            4.0,
        )
        times_s = [t["decision_ms"] / 1000 for t in ticks]
        summary.append(
            (
                arm,
                f"{len(ticks)} decisions of {min(times_s):.1f} to {max(times_s):.1f} s, none cut; run finished at {answer['elapsed_ms'] / 1000:.0f} s",
            )
        )
    else:
        frame(
            last,
            [
                (
                    f"Deadline exceeded: no answer within {deadline} s, the run stops cleanly at {answer['elapsed_ms'] / 1000:.1f} s",
                    F(24, True),
                    (255, 140, 150),
                ),
                (
                    "BLOCKED, decision failed [181001] model call failed (decisions connection failed); no hang",
                    MONO(18),
                    GREY,
                ),
            ],
            4.0,
        )
        summary.append(
            (
                arm,
                f"stopped cleanly at {answer['elapsed_ms'] / 1000:.1f} s, first decision not answered within {deadline} s",
            )
        )

slide(
    [
        ("Outcome", F(32, True), DARK),
        (f"Default 5 s: {summary[0][1]}.", F(21), RED),
        (f"30 s: {summary[1][1]}.", F(21), GREEN),
        (f"1 s: {summary[2][1]}.", F(21), RED),
        ("", F(10), DARK),
        *[
            (line, F(21), DARK)
            for line in os.environ.get(
                "VIDEO_OUTCOME",
                "The task itself is not solved in the 30 s run: the agent filled Zurich and London, then read a price|from the round-trip calendar (EUR 102, Nov 8) without a one-way search. The deadline let it run; the|9B model's choices did not finish the task.",
            ).split("|")
        ],
    ],
    9.0,
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
