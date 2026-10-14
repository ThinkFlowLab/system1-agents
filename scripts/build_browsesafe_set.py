# coding: utf-8
"""A seeded, label-stratified BrowseSafe-Bench sample in the injection guard's labelled-set shape (#75).

BrowseSafe-Bench (perplexity-ai/browsesafe-bench, MIT; Zhang et al., arXiv:2511.20597) holds whole HTML pages
from browser-agent use, labelled ``yes`` (carries a prompt injection) or ``no``. Its test split is page-length:
the median row is about 46,000 characters, while the guard reads the first ``MAX_CHARS`` of a tool result.

    # pick the rows (writes evals/labelled/browsesafe-test-sample.json; commit it)
    uv run --with pyarrow python scripts/build_browsesafe_set.py sample --per-label 100 --seed 0
    # rebuild the JSONL from the committed row ids (not committed: the dataset is not vendored)
    uv run --with pyarrow python scripts/build_browsesafe_set.py build
    uv run s1a run injection_guard --model jev --labelled-set .external/browsesafe/test-sample-html.jsonl
    # how much page text the guard inspects, and how much reaches the model under each policy
    uv run --with pyarrow python scripts/build_browsesafe_set.py coverage

``state.text`` is the page in full (the eval applies the runtime clip, see ``RailSpec.labelled_state``):

- ``html`` (default): the row's HTML unchanged, as a raw ``fetch_webpage`` returns it. The label is defined on
  this text, so a hidden injection (comment, attribute, ``display:none`` block) is still there.
- ``text``: the visible text plus comments and the alt/title/aria-label/placeholder attributes, roughly what a
  reader-mode fetch hands the model. Shorter, so more of each page is inspected, but an injection in some other
  attribute or in ``<script>`` is dropped while its row keeps the ``yes`` label.

A Playwright ``browser_snapshot`` of the rendered page would match production best; it needs a browser per row
and is left for a follow-up. pyarrow is needed only to read the parquet file and is not a project dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections.abc import Iterable, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = "perplexity-ai/browsesafe-bench"
REVISION = "b506fb5bc7fd4472c8738055a67a0ef6406afdc9"  # the dataset's main on 2026-10-14 (last modified 2025-12-04)
SPLIT = "test"
PARQUET_SHA256 = "00cbad96b60fee46e016d79af6981fb221384c61f12cf28b4f04b5a6420573d0"  # test.parquet at REVISION
LABELS = ("yes", "no")  # BrowseSafe-Bench's label column; yes: the page carries an injection
TOOL = "fetch_webpage"
MODES = ("html", "text")
IDS = ROOT / "evals" / "labelled" / "browsesafe-test-sample.json"
CACHE = ROOT / ".external" / "browsesafe"
MAX_CHARS = 6000  # s1a.agents.injection_guard.MAX_CHARS; tests check the two agree

_SKIP = {"script", "style", "noscript", "template", "head", "svg"}
_BLOCK = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "fieldset", "figcaption", "figure",
    "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
    "section", "table", "td", "th", "tr", "ul",
}  # fmt: skip
_KEPT_ATTRIBUTES = ("alt", "title", "aria-label", "placeholder")


def parquet_url(revision: str = REVISION) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{SPLIT}.parquet"


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skipping += 1
            return
        if tag in _BLOCK:
            self.parts.append("\n")
        if not self._skipping:
            for name, value in attrs:
                if name in _KEPT_ATTRIBUTES and value and value.strip():
                    self.parts.append(f" {value.strip()} ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _SKIP:
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skipping = max(0, self._skipping - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)

    def handle_comment(self, data: str) -> None:
        if not self._skipping and data.strip():
            self.parts.append(f"\n<!-- {data.strip()} -->\n")


def html_to_text(html: str) -> str:
    """Visible text, comments and reader-visible attributes, one block per line, runs of blanks collapsed."""
    parser = _Text()
    parser.feed(html)
    parser.close()
    lines = (re.sub(r"[ \t\r\f\v\u00a0]+", " ", line).strip() for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line)


def page_text(content: str, mode: str) -> str:
    match mode:
        case "html":
            return content
        case "text":
            return html_to_text(content)
    raise ValueError(f"unknown text mode {mode!r}; one of {MODES}")


def stratified_sample(labels: Sequence[str], per_label: int, seed: int) -> list[int]:
    """``per_label`` row ids of each label, drawn with one seeded generator per label; sorted ids."""
    picked: list[int] = []
    for label in LABELS:
        ids = [i for i, value in enumerate(labels) if value == label]
        if len(ids) < per_label:
            raise ValueError(f"only {len(ids)} {label!r} rows, asked for {per_label}")
        picked.extend(random.Random(f"{seed}:{label}").sample(ids, per_label))
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"unexpected labels {unknown}; expected {LABELS}")
    return sorted(picked)


def record(row_id: int, content: str, label: str, mode: str) -> dict[str, Any]:
    """One labelled-set line: the full page text (the eval clips it), the boolean label, the row it came from."""
    if label not in LABELS:
        raise ValueError(f"row {row_id}: label {label!r} is not one of {LABELS}")
    return {
        "state": {"tool": TOOL, "text": page_text(content, mode)},
        "label": label == "yes",
        "note": f"browsesafe-bench@{REVISION[:7]} {SPLIT} row {row_id} ({mode})",
    }


def records(rows: Sequence[tuple[str, str]], row_ids: Iterable[int], mode: str) -> list[dict[str, Any]]:
    return [record(i, rows[i][0], rows[i][1], mode) for i in row_ids]


def coverage(lengths: Sequence[int], clip: int = MAX_CHARS) -> dict[str, Any]:
    """How much of each page one clipped decision inspects.

    On main every page below the act band reaches the model whole while only its first ``clip`` characters were
    inspected. Under #57's withhold-uninspected policy (PR #73) the model gets the inspected prefix only, so the
    share of page text reaching it equals the inspected share. Both assume the page is allowed or uncertain."""
    if not lengths:
        raise ValueError("no pages")
    total = sum(lengths)
    inspected = sum(min(n, clip) for n in lengths)
    return {
        "pages": len(lengths),
        "clip": clip,
        "median_chars": int(statistics.median(lengths)),
        "fully_inspected_pages": sum(n <= clip for n in lengths),
        "inspected_share": round(inspected / total, 3) if total else 1.0,
        "mean_page_inspected_share": round(statistics.fmean(min(1.0, clip / n) if n else 1.0 for n in lengths), 3),
        "reaches_model_share": {"main": 1.0, "withhold_uninspected": round(inspected / total, 3) if total else 1.0},
    }


def coverage_rows(rows: Sequence[tuple[str, str]], row_ids: Iterable[int] | None, clip: int = MAX_CHARS) -> list[str]:
    """Markdown rows: per text mode and label, the coverage figures above."""
    ids = list(range(len(rows))) if row_ids is None else list(row_ids)
    out = [
        "| text | label | pages | median chars | fully inspected | text inspected | mean page inspected "
        "| reaches model, main | reaches model, withhold |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        texts = {i: page_text(rows[i][0], mode) for i in ids}
        for label in LABELS:
            c = coverage([len(texts[i]) for i in ids if rows[i][1] == label], clip)
            share = c["reaches_model_share"]
            out.append(
                f"| {mode} | {label} | {c['pages']} | {c['median_chars']:,} | {c['fully_inspected_pages']} "
                f"| {c['inspected_share']:.1%} | {c['mean_page_inspected_share']:.1%} | {share['main']:.0%} "
                f"| {share['withhold_uninspected']:.1%} |"
            )
    return out


def read_rows(path: Path) -> list[tuple[str, str]]:
    """``(content, label)`` per row of the split's parquet file, in file order (the row id is the index)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("reading the parquet file needs pyarrow: uv run --with pyarrow python scripts/build_browsesafe_set.py")
    table = pq.read_table(path, columns=["content", "label"])
    return list(zip(table.column("content").to_pylist(), table.column("label").to_pylist()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> Path:
    """``path`` if its bytes are the split's parquet file at REVISION, else a ValueError naming both digests.

    Every command hashes its source before decoding it, so a local ``--parquet`` from another revision cannot be
    stamped with the pinned revision, digest or row ids."""
    digest = sha256(path)
    if digest != PARQUET_SHA256:
        raise ValueError(f"{path}: sha256 {digest}, expected {PARQUET_SHA256} for revision {REVISION}")
    return path


def fetch(cache: Path = CACHE) -> Path:
    """The split's parquet file at REVISION, downloaded once and checked against PARQUET_SHA256."""
    path = cache / f"{SPLIT}-{REVISION[:12]}.parquet"
    if not path.exists():
        import httpx

        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        with httpx.stream("GET", parquet_url(), follow_redirects=True, timeout=300) as response:
            response.raise_for_status()
            with partial.open("wb") as out:
                for chunk in response.iter_bytes():
                    out.write(chunk)
        partial.rename(path)
    return verify(path)


def load_ids(path: Path = IDS) -> dict[str, Any]:
    ids = json.loads(path.read_text(encoding="utf-8"))
    if (ids["dataset"], ids["revision"], ids["split"]) != (DATASET, REVISION, SPLIT):
        raise ValueError(f"{path} names {ids['dataset']}@{ids['revision']} {ids['split']}, not {DATASET}@{REVISION}")
    return ids


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, help=f"a local {SPLIT}.parquet at the pinned revision (else download)")
    parser.add_argument("--ids", type=Path, default=IDS, help="the committed row-id file")
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="draw the row ids and write the id file")
    sample.add_argument("--per-label", type=int, default=100)
    sample.add_argument("--seed", type=int, default=0)
    build = sub.add_parser("build", help="write the labelled-set JSONL for the committed row ids")
    build.add_argument("--text", choices=MODES, default="html")
    build.add_argument("--out", type=Path, help="default .external/browsesafe/test-sample-<text>.jsonl")
    report = sub.add_parser("coverage", help="print the coverage table for the sample and the whole split")
    report.add_argument("--clip", type=int, default=MAX_CHARS)
    args = parser.parse_args(argv)

    rows = read_rows(verify(args.parquet) if args.parquet else fetch())
    if args.command == "sample":
        row_ids = stratified_sample([label for _, label in rows], args.per_label, args.seed)
        args.ids.write_text(
            json.dumps(
                {
                    "dataset": DATASET,
                    "revision": REVISION,
                    "split": SPLIT,
                    "parquet_sha256": PARQUET_SHA256,
                    "seed": args.seed,
                    "per_label": args.per_label,
                    "labels": {label: sum(rows[i][1] == label for i in row_ids) for label in LABELS},
                    "row_ids": row_ids,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{len(row_ids)} row ids in {args.ids}")
        return
    ids = load_ids(args.ids)
    if args.command == "build":
        out = args.out or CACHE / f"{SPLIT}-sample-{args.text}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = records(rows, ids["row_ids"], args.text)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines), encoding="utf-8")
        print(f"{len(lines)} records ({sum(r['label'] for r in lines)} injected) in {out}")
        return
    print(f"Sample of {len(ids['row_ids'])} rows (seed {ids['seed']}), clip {args.clip:,} characters:\n")
    print("\n".join(coverage_rows(rows, ids["row_ids"], args.clip)))
    print(f"\nWhole {SPLIT} split ({len(rows):,} rows):\n")
    print("\n".join(coverage_rows(rows, None, args.clip)))


if __name__ == "__main__":
    main()
