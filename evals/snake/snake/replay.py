"""Render the same terminal cells from real timestamped runs, at original speed."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from bisect import bisect_right
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .ui import BG, DIM, MUTED, compose


def load_record(path):
    metadata = None
    frames = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["type"] == "metadata":
            metadata = event
        elif event["type"] == "frame":
            frames.append(event)
    if metadata is None or not frames:
        raise ValueError("Recording must contain metadata and real decision frames")
    times = [frame["at"] for frame in frames]
    if any(not math.isfinite(t) or t < 0 for t in times) or times != sorted(times) or len(set(times)) != len(times):
        raise ValueError("Recording timestamps must strictly increase")
    return metadata, frames


class TerminalRaster:
    def __init__(self, columns, rows, *, width=1920, height=1080, font=None):
        choices = (
            [Path(font)]
            if font
            else [
                Path("/System/Library/Fonts/Menlo.ttc"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
                Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts/consola.ttf",
            ]
        )
        font_path = next((p for p in choices if p.is_file()), None)
        if font_path is None:
            raise FileNotFoundError("A monospace font is required; pass --font /path/to/font.ttf")
        size = min(int((width - 120) / (columns * 0.61)), int((height - 110) / rows) - 1)
        if size < 1:
            raise ValueError("Canvas is too large for the output image; use fewer games or a larger image")
        self.font = ImageFont.truetype(str(font_path), size)
        self.cw = max(1, round(self.font.getlength("M")))
        self.ch = size + 1
        self.width, self.height = width, height
        self.x = (width - columns * self.cw) // 2
        self.y = (height - rows * self.ch) // 2 + 16
        self.cache = {}
        self.base = Image.new("RGB", (width, height), "#05090c")
        draw = ImageDraw.Draw(self.base)
        box = (
            self.x - 22,
            self.y - 42,
            self.x + columns * self.cw + 22,
            self.y + rows * self.ch + 14,
        )
        draw.rounded_rectangle(box, radius=18, fill=BG, outline=DIM, width=2)
        for index, color in enumerate(("#ed6a67", "#eeb65a", "#5ec486")):
            x = self.x + index * 22
            draw.ellipse((x, self.y - 25, x + 11, self.y - 14), fill=color)
        title_font = ImageFont.truetype(str(font_path), max(12, size - 9))
        draw.text(
            (width // 2, self.y - 24),
            "laya-snake  /  real recorded decisions",
            font=title_font,
            fill=MUTED,
            anchor="mt",
        )

    def glyph(self, character, color):
        key = character, color
        if key not in self.cache:
            glyph = Image.new("RGBA", (self.cw, self.ch), (0, 0, 0, 0))
            draw = ImageDraw.Draw(glyph)
            if character == "█":
                draw.rectangle((0, 0, self.cw, self.ch), fill=color)
            elif character == "▀":
                draw.rectangle((0, 0, self.cw, self.ch // 2), fill=color)
            elif character == "▄":
                draw.rectangle((0, self.ch // 2, self.cw, self.ch), fill=color)
            elif character == "━":
                draw.rectangle((0, self.ch // 2, self.cw, self.ch // 2 + 1), fill=color)
            else:
                draw.text((0, -1), character, font=self.font, fill=color, anchor="la")
            self.cache[key] = glyph
        return self.cache[key]

    def render(self, canvas):
        frame = self.base.copy()
        for row, (characters, colors) in enumerate(zip(canvas.chars, canvas.styles)):
            for column, (character, color) in enumerate(zip(characters, colors)):
                if character != " ":
                    glyph = self.glyph(character, color)
                    frame.paste(glyph, (self.x + column * self.cw, self.y + row * self.ch), glyph)
        return frame


def ffmpeg_mp4_cmd(ffmpeg, out_path, fps, width=1920, height=1080):
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True, help=".mp4 or .png")
    parser.add_argument("--start", type=float, default=0, help="Source recording time in seconds")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate; playback remains 1×")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--font")
    parser.add_argument("--gif", type=Path, help="Also create a shareable GIF from the MP4")
    parser.add_argument(
        "--gif-seconds",
        type=float,
        default=15,
        help="GIF excerpt length at original speed (default: 15)",
    )
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.start)
        or not math.isfinite(args.seconds)
        or not math.isfinite(args.gif_seconds)
        or args.start < 0
        or args.seconds <= 0
        or args.gif_seconds <= 0
        or args.fps < 1
        or args.width < 640
        or args.height < 480
        or args.width % 2
        or args.height % 2
    ):
        parser.error("Use finite nonnegative start, positive seconds/FPS and even dimensions >= 640×480")
    if args.output.exists():
        parser.error("Output already exists; choose a new filename")
    if args.output.suffix not in (".mp4", ".png"):
        parser.error("--output must end in .mp4 or .png")
    if args.gif and args.gif.exists():
        parser.error("GIF output already exists; choose a new filename")
    metadata, frames = load_record(args.recording)
    times = [f["at"] for f in frames]
    start = max(times[0], args.start)
    end = min(start + args.seconds, times[-1])
    if start > times[-1] or (args.output.suffix == ".mp4" and end <= start):
        parser.error("Requested interval is outside the recording")

    def canvas_at(t):
        entry = frames[max(0, bisect_right(times, t) - 1)]
        return compose(entry["game"], entry["decision"], {**entry["stats"], "replay": True})

    canvas = canvas_at(start)
    raster = TerminalRaster(canvas.width, canvas.height, width=args.width, height=args.height, font=args.font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".png":
        raster.render(canvas).save(args.output)
        print(args.output)
        return 0
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("MP4 export needs ffmpeg (on macOS: brew install ffmpeg)")
    count = max(1, int((end - start) * args.fps))
    process = subprocess.Popen(
        ffmpeg_mp4_cmd(ffmpeg, args.output, args.fps, args.width, args.height),
        stdin=subprocess.PIPE,
    )
    try:
        last_index, pixels = None, None
        for index in range(count):
            t = start + index / args.fps
            source_index = max(0, bisect_right(times, t) - 1)
            if source_index != last_index:
                pixels = raster.render(canvas_at(t)).tobytes()
                last_index = source_index
            process.stdin.write(pixels)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg did not complete the MP4")
    except BaseException:
        process.kill()
        process.wait()
        raise
    if args.gif:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-i",
                str(args.output),
                "-t",
                str(min(args.gif_seconds, count / args.fps)),
                "-filter_complex",
                "fps=10,scale=1040:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer",
                "-loop",
                "0",
                str(args.gif),
            ],
            check=True,
        )
    sidecar = {
        "source_recording": args.recording.name,
        "source_sha256": hashlib.sha256(args.recording.read_bytes()).hexdigest(),
        "model": metadata["model"],
        "playback_speed": 1,
        "source_start_seconds": start,
        "source_end_seconds": start + count / args.fps,
        "video_fps": args.fps,
        "video_frames": count,
        "gif_seconds": min(args.gif_seconds, count / args.fps) if args.gif else None,
        "renderer_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("ui.py").read_bytes()
        ).hexdigest(),
        "note": "Rendered from real inference records using the live terminal layout. Original wall-clock timing; no fabricated probabilities or time compression. Video is sampled at its stated frame rate.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({"video": str(args.output), "seconds": count / args.fps, "playback_speed": 1}))
    return 0
