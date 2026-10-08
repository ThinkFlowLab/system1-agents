# coding: utf-8
"""GIFs: the replay page screenshotted per step or per tick of episode time, folders of PNGs stitched, or cast
frames side by side under a header band (the strip)."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

from evals.replay.page import ordered
from evals.replay.trial import Trial

VIEWPORT_WIDTH = 1180
VIEWPORT_HEIGHT = 1600
MIN_TICK_MS = 100
STEP_DURATION_MS = 700
MIN_FRAME_MS = 20
HEADER_HEIGHT = 84  # the strip's header band at a column width of 800 px; scales with the column
MODEL_COLORS = {"jev": "#0f766e", "llm": "#b45309"}  # the replay page's badge colours
INK, MUTED, LINE, DONE, PANEL, OTHER = "#1c1f26", "#6b7280", "#e3e6eb", "#2563eb", "#ffffff", "#6b7280"


@dataclass(frozen=True)
class Column:
    """One side of a strip GIF: its cast frames and the header band's text."""

    frames: list[Path]
    badge: str
    color: str
    facts: str
    total_ms: int  # the run's wall clock; the column holds its last frame until the clock passes it


def time_ticks(total_ms: int, *, speed: float, max_frames: int) -> tuple[int, int]:
    """Episode milliseconds per frame and the frame's display time: a whole number of centiseconds (the GIF unit),
    so the GIF plays at exactly ``speed`` times real time with at most ``max_frames`` frames."""
    raw_tick_ms = max(MIN_TICK_MS, math.ceil(total_ms / max(1, max_frames - 1)))
    frame_ms = max(MIN_FRAME_MS, math.ceil(raw_tick_ms / speed / 10) * 10)
    return round(frame_ms * speed), frame_ms


def write_gif(page: Path, out: Path, *, mode: str, width: int, speed: float, max_frames: int) -> Path:
    """Screenshot ``page`` once per step (``mode="step"``) or per tick of episode time (``mode="time"``).

    In time mode the GIF plays the episode at ``speed`` times real time, with at most ``max_frames`` frames.
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        tab = browser.new_page(viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
        tab.goto(page.resolve().as_uri())
        trials = tab.evaluate("JSON.parse(document.getElementById('replay-data').textContent).trials")
        max_steps = max(int(trial["steps"]) for trial in trials)
        total_ms = max(int(trial["times"][-1]) for trial in trials)
        match mode:
            case "step":
                ticks = [(f"replay.setStep({k})", STEP_DURATION_MS) for k in range(max_steps + 1)]
            case "time":
                tick_ms, frame_ms = time_ticks(total_ms, speed=speed, max_frames=max_frames)
                ticks = [(f"replay.setTime({t})", frame_ms) for t in range(0, total_ms + tick_ms, tick_ms)]
            case _:
                raise ValueError(f"mode must be 'step' or 'time', not {mode!r}")
        frames, durations = [], []
        stage = tab.locator("#stage")
        for script, duration in ticks:
            tab.evaluate(script)
            box = stage.bounding_box()
            if box is None:
                raise RuntimeError("the replay page has no #stage element")
            # A full-page shot clipped to the stage: an element shot of a stage taller than the viewport tears.
            shot = tab.screenshot(type="png", full_page=True, clip=box)
            frames.append(_scaled(Image.open(io.BytesIO(shot)), width))
            durations.append(duration)
        browser.close()
    return _save(frames, durations, out)


def frames_to_gif(directories: list[Path], out: Path, *, fps: float, width: int) -> Path:
    """The PNGs of one or two folders at ``fps``; two folders play side by side, left then right.

    Frames named ``t<ms>-<n>.png`` (evals.replay.cast) sit on the wall clock: every tick shows each side's latest
    frame at that time, so both sides run at real speed and the finished side holds its last frame. Other names
    play in order, one frame per tick.
    """
    columns = []
    for directory in directories:
        paths = sorted(directory.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"no PNG frames in {directory}")
        columns.append(paths)
    tick_ms = max(MIN_FRAME_MS, int(1000 / fps))
    if all(_stamp(path) is not None for column in columns for path in column):
        last_ms = max(_stamp(column[-1]) or 0 for column in columns)
        ticks = range(0, last_ms + tick_ms, tick_ms)
        picks = [[_latest(column, tick) for column in columns] for tick in ticks]
    else:
        count = max(len(column) for column in columns)
        picks = [[column[min(index, len(column) - 1)] for column in columns] for index in range(count)]
    cache: dict[Path, Image.Image] = {}
    frames = []
    for row in picks:
        images = []
        for path in row:
            if path not in cache:
                cache[path] = _scaled(Image.open(path), width // len(directories))
            images.append(cache[path])
        frames.append(_beside(images))
    return _save(frames, [tick_ms] * len(frames), out)


def strip_columns(trials: list[Trial]) -> list[Column]:
    """The strip's columns for one or two browser trials, Jev left: badge, facts and clock from the trial."""
    columns = []
    for trial in ordered(trials):
        if not trial.frames:
            raise FileNotFoundError(
                f"no frames under {trial.path}: the strip needs a run cast through evals.replay.cast"
            )
        cost = "n/a" if trial.cost_usd is None else f"${trial.cost_usd:.4f}"
        columns.append(
            Column(
                frames=trial.frames,
                badge="LLM" if trial.model == "llm" else f"SYSTEM 1 · {trial.model.upper()}",
                color=MODEL_COLORS.get(trial.model, OTHER),
                facts=f"{trial.elapsed_s:.1f} s · {trial.steps} steps · {len(trial.decisions)} decisions · {cost}",
                total_ms=round(trial.elapsed_s * 1000),
            )
        )
    return columns


def strip_gif(columns: list[Column], out: Path, *, speed: float, width: int, max_frames: int) -> Path:
    """The columns' frames side by side on the wall clock under a header band (the badge, the facts, a running
    clock and a progress bar), at ``speed`` times real time with at most ``max_frames`` frames."""
    total_ms = max(max(column.total_ms, _stamp(column.frames[-1]) or 0) for column in columns)
    tick_ms, frame_ms = time_ticks(total_ms, speed=speed, max_frames=max_frames)
    column_width = width // len(columns)
    cache: dict[Path, Image.Image] = {}
    frames = []
    for tick in range(0, total_ms + tick_ms, tick_ms):
        panels = []
        for column in columns:
            path = _latest(column.frames, tick)
            if path not in cache:
                cache[path] = _scaled(Image.open(path), column_width)
            panels.append(_with_header(cache[path], column, tick))
        frames.append(_beside(panels))
    return _save(frames, [frame_ms] * len(frames), out)


def _with_header(page: Image.Image, column: Column, tick_ms: int) -> Image.Image:
    scale = page.width / 800
    px = lambda n: max(1, round(n * scale))  # noqa: E731 - one scaling helper for the band's geometry
    height = px(HEADER_HEIGHT)
    canvas = Image.new("RGB", (page.width, height + page.height), PANEL)
    canvas.paste(page, (0, height))
    draw = ImageDraw.Draw(canvas)
    big, small = ImageFont.load_default(size=px(15)), ImageFont.load_default(size=px(12))
    pad = px(12)
    badge_width = round(draw.textlength(column.badge, font=big)) + 2 * pad
    draw.rounded_rectangle([pad, pad, pad + badge_width, pad + px(24)], radius=px(12), fill=column.color)
    draw.text((2 * pad, pad + px(4)), column.badge, fill=PANEL, font=big)
    draw.text((badge_width + 2 * pad, pad + px(6)), column.facts, fill=MUTED, font=small)
    shown = min(tick_ms, column.total_ms)
    draw.text(
        (pad, pad + px(34)), f"t = {shown / 1000:.1f} s of {column.total_ms / 1000:.1f} s", fill=MUTED, font=small
    )
    top, bottom = pad + px(56), pad + px(62)
    draw.rounded_rectangle([pad, top, page.width - pad, bottom], radius=px(3), fill=LINE)
    done = round((page.width - 2 * pad) * shown / max(1, column.total_ms))
    if done > 0:
        draw.rounded_rectangle([pad, top, pad + done, bottom], radius=px(3), fill=DONE)
    return canvas


def _stamp(path: Path) -> int | None:
    """Elapsed milliseconds from a cast frame name ``t<ms>-<n>``; None for any other name."""
    digits = path.stem[1:].split("-", 1)[0]
    return int(digits) if path.stem.startswith("t") and digits.isdigit() else None


def _latest(column: list[Path], tick_ms: int) -> Path:
    """The last frame taken at or before ``tick_ms``, or the first frame before any was taken."""
    latest = column[0]
    for path in column:
        if (_stamp(path) or 0) <= tick_ms:
            latest = path
        else:
            break
    return latest


def _beside(images: list[Image.Image]) -> Image.Image:
    if len(images) == 1:
        return images[0]
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (sum(image.width for image in images), height), images[0].getpixel((0, 0)))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _scaled(image: Image.Image, width: int) -> Image.Image:
    rgb = image.convert("RGB")
    if rgb.width == width:
        return rgb
    return rgb.resize((width, max(1, round(rgb.height * width / rgb.width))), Image.LANCZOS)


def _save(frames: list[Image.Image], durations: list[int], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    # One GIF canvas for frames whose stage grows as transcripts fill: pad every frame to the largest.
    width, height = max(frame.width for frame in frames), max(frame.height for frame in frames)
    padded = []
    for frame in frames:
        canvas = Image.new("RGB", (width, height), frame.getpixel((0, 0)))
        canvas.paste(frame, (0, 0))
        padded.append(canvas)
    palette = [frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for frame in padded]
    palette[0].save(out, save_all=True, append_images=palette[1:], duration=durations, loop=0, disposal=1)
    return out
