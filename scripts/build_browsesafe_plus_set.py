# coding: utf-8
"""A seeded BrowseSafe++ training-pair sample in the injection guard's labelled-set shape (#87).

BrowseSafe++ (perplexity-ai/browsesafe-plus-plus, train split CC BY 4.0; Chen et al., arXiv:2610.17457) holds
matched attack/control **pairs** (not unpaired HTML rows like BrowseSafe-Bench / #75). Each page is available as
HTML, Markdown, accessibility tree and screenshot. The public **train** split is labelled (29,500 pairs); the
test inputs are leaderboard-only and must not be vendored or scraped for labels.

The Hugging Face dataset is **gated** (request access on the dataset page, then ``HF_TOKEN`` / ``hf auth login``).
Without a token the download commands fail clearly; offline unit tests use tiny local fixtures instead.

    # after HF access: pick pairs (writes evals/labelled/browsesafe-plus-train-sample.json; commit it)
    HF_TOKEN=… uv run --with pyarrow --with huggingface-hub \\
        python scripts/build_browsesafe_plus_set.py sample --per-pair 50 --seed 0
    # rebuild JSONL from committed pair ids (dataset not vendored)
    HF_TOKEN=… uv run --with pyarrow --with huggingface-hub \\
        python scripts/build_browsesafe_plus_set.py build --view markdown
    uv run s1a run injection_guard --model jev --labelled-set .external/browsesafe-plus/train-sample-markdown.jsonl
    uv run --with pyarrow python scripts/build_browsesafe_plus_set.py coverage

``state.text`` is the chosen view in full (the eval applies the runtime clip). Default view is ``markdown``
(reader-like text). ``accessibility`` maps to ``browser_snapshot``; ``html`` / ``markdown`` to ``fetch_webpage``.
Screenshot detection is a follow-up: ``injection_guard`` has no multimodal path today.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = "perplexity-ai/browsesafe-plus-plus"
REVISION = "7674843074bf3b36c0a0209dfd06e03cf47743dd"  # dataset main tip on 2026-11-06
SPLIT = "train"
LABELS = ("yes", "no")  # BrowseSafe++ label column; yes = injection present
VIEWS = ("markdown", "accessibility", "html")
VIEW_COLUMN = {
    "markdown": "content_markdown",
    "accessibility": "content_ax_yaml",
    "html": "content_html",
}
VIEW_TOOL = {
    "markdown": "fetch_webpage",
    "html": "fetch_webpage",
    "accessibility": "browser_snapshot",
}
IDENTITY_COLUMNS = ("entry_id", "arm", "label", "split")
IDS = ROOT / "evals" / "labelled" / "browsesafe-plus-train-sample.json"
CACHE = ROOT / ".external" / "browsesafe-plus"
MAX_CHARS = 6000  # s1a.agents.injection_guard.MAX_CHARS; tests check the two agree
_ENTRY_RE = re.compile(r"^(\d{5})_(malicious|benign)$")


def pair_id_of(entry_id: str) -> str:
    """``NNNNN`` shared by ``NNNNN_malicious`` and ``NNNNN_benign``."""
    match = _ENTRY_RE.fullmatch(entry_id)
    if not match:
        raise ValueError(f"entry_id {entry_id!r} is not NNNNN_malicious / NNNNN_benign")
    return match.group(1)


def arm_of(entry_id: str) -> str:
    match = _ENTRY_RE.fullmatch(entry_id)
    if not match:
        raise ValueError(f"entry_id {entry_id!r} is not NNNNN_malicious / NNNNN_benign")
    return "attack" if match.group(2) == "malicious" else "control"


def page_text(page: Mapping[str, Any], view: str) -> str:
    if view not in VIEW_COLUMN:
        raise ValueError(f"unknown view {view!r}; one of {VIEWS}")
    text = page.get(VIEW_COLUMN[view])
    if not isinstance(text, str) or not text:
        raise ValueError(f"{page.get('entry_id')}: empty {VIEW_COLUMN[view]}")
    return text


def record(page: Mapping[str, Any], view: str) -> dict[str, Any]:
    """One labelled-set line: full view text (the eval clips it), boolean label, source note."""
    label = page.get("label")
    if label not in LABELS:
        raise ValueError(f"{page.get('entry_id')}: label {label!r} is not one of {LABELS}")
    entry_id = page["entry_id"]
    return {
        "state": {"tool": VIEW_TOOL[view], "text": page_text(page, view)},
        "label": label == "yes",
        "note": f"browsesafe-plus-plus@{REVISION[:7]} {SPLIT} {entry_id} ({view})",
    }


def records(
    pages_by_entry: Mapping[str, Mapping[str, Any]], pair_ids: Iterable[str], view: str
) -> list[dict[str, Any]]:
    """Attack then control for each pair id, in the order of ``pair_ids``."""
    out: list[dict[str, Any]] = []
    for pid in pair_ids:
        for suffix in ("malicious", "benign"):
            entry = f"{pid}_{suffix}"
            if entry not in pages_by_entry:
                raise ValueError(f"missing member {entry} for pair {pid}")
            out.append(record(pages_by_entry[entry], view))
    return out


def group_pairs(pages: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    """``pair_id -> {"attack": page, "control": page}`` for complete, consistent train pairs."""
    buckets: dict[str, dict[str, Mapping[str, Any]]] = {}
    for page in pages:
        if page.get("split") not in (None, SPLIT):
            continue
        entry_id = page["entry_id"]
        pid = pair_id_of(entry_id)
        arm = page.get("arm") or arm_of(entry_id)
        label = page.get("label")
        expected_label = "yes" if arm == "attack" else "no"
        if label not in (None, expected_label):
            raise ValueError(f"{entry_id}: arm {arm!r} disagrees with label {label!r}")
        bucket = buckets.setdefault(pid, {})
        if arm in bucket:
            raise ValueError(f"duplicate {arm} member for pair {pid}")
        bucket[arm] = page
    complete = {pid: members for pid, members in buckets.items() if set(members) == {"attack", "control"}}
    return complete


def sample_pairs(pair_ids: Sequence[str], per_pair: int, seed: int) -> list[str]:
    """``per_pair`` pair ids drawn with a seeded generator; sorted."""
    ids = list(pair_ids)
    if len(ids) < per_pair:
        raise ValueError(f"only {len(ids)} pairs, asked for {per_pair}")
    picked = random.Random(seed).sample(ids, per_pair)
    return sorted(picked)


def coverage(lengths: Sequence[int], clip: int = MAX_CHARS) -> dict[str, Any]:
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


def coverage_rows(
    pages_by_entry: Mapping[str, Mapping[str, Any]],
    pair_ids: Iterable[str],
    clip: int = MAX_CHARS,
) -> list[str]:
    """Markdown rows: per view and label, the coverage figures above."""
    out = [
        "| view | label | pages | median chars | fully inspected | text inspected | mean page inspected "
        "| reaches model, main | reaches model, withhold |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    entries = [f"{pid}_{suffix}" for pid in pair_ids for suffix in ("malicious", "benign")]
    for view in VIEWS:
        texts = {e: page_text(pages_by_entry[e], view) for e in entries}
        for label, suffix in (("yes", "malicious"), ("no", "benign")):
            lengths = [len(texts[f"{pid}_{suffix}"]) for pid in pair_ids]
            c = coverage(lengths, clip)
            share = c["reaches_model_share"]
            out.append(
                f"| {view} | {label} | {c['pages']} | {c['median_chars']:,} | {c['fully_inspected_pages']} "
                f"| {c['inspected_share']:.1%} | {c['mean_page_inspected_share']:.1%} | {share['main']:.0%} "
                f"| {share['withhold_uninspected']:.1%} |"
            )
    return out


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        sys.exit("reading parquet needs pyarrow: uv run --with pyarrow python scripts/build_browsesafe_plus_set.py")


def read_pages(paths: Sequence[Path], columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Page dicts from one or more parquet shards (fixture or downloaded train shards)."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    wanted = list(columns) if columns is not None else list(IDENTITY_COLUMNS) + list(VIEW_COLUMN.values())
    pages: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path, columns=[c for c in wanted if c in pq.read_schema(path).names])
        pages.extend(table.to_pylist())
    return pages


def index_pages(pages: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {page["entry_id"]: page for page in pages}


def parquet_paths(source: Path) -> list[Path]:
    """Train parquet shards under ``source`` or ``source/data`` (published layout or SDK fixture)."""
    roots = [source, source / "data"]
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        found.extend(sorted(root.glob("members-train-*.parquet")))
        found.extend(
            sorted(p for p in root.glob("members-*.parquet") if "train" not in p.name and "test" not in p.name)
        )
    # de-dupe while keeping order
    seen: set[Path] = set()
    out: list[Path] = []
    for path in found:
        if path.resolve() not in seen:
            seen.add(path.resolve())
            out.append(path)
    if not out:
        raise FileNotFoundError(
            f"no members-*.parquet under {source} (expected data/members-train-*.parquet or fixture members-*.parquet)"
        )
    return out


def gated_hint() -> str:
    return (
        f"dataset {DATASET} is gated on Hugging Face: request access at "
        f"https://huggingface.co/datasets/{DATASET}, accept the terms, then set HF_TOKEN "
        f"(or run `hf auth login`) and retry. Offline tests use local --source fixtures instead."
    )


def fetch(cache: Path = CACHE, revision: str = REVISION) -> list[Path]:
    """Download the train parquet shards at ``revision`` into ``cache`` (needs HF_TOKEN for the gated repo)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise PermissionError(gated_hint())
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "downloading needs huggingface-hub: "
            "uv run --with pyarrow --with huggingface-hub python scripts/build_browsesafe_plus_set.py"
        )
    cache.mkdir(parents=True, exist_ok=True)
    try:
        local = snapshot_download(
            repo_id=DATASET,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["data/members-train-*.parquet"],
            local_dir=cache / f"rev-{revision[:12]}",
            token=token,
        )
    except Exception as exc:  # noqa: BLE001 — surface hub errors with the gated hint
        raise PermissionError(f"{gated_hint()} Underlying error: {exc}") from exc
    return parquet_paths(Path(local))


def load_ids(path: Path = IDS) -> dict[str, Any]:
    ids = json.loads(path.read_text(encoding="utf-8"))
    if (ids.get("dataset"), ids.get("revision"), ids.get("split")) != (DATASET, REVISION, SPLIT):
        raise ValueError(
            f"{path} names {ids.get('dataset')}@{ids.get('revision')} {ids.get('split')}, "
            f"not {DATASET}@{REVISION} {SPLIT}"
        )
    return ids


def resolve_paths(source: Path | None) -> list[Path]:
    if source is not None:
        return parquet_paths(source)
    return fetch()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        help="local dataset root with data/members-*.parquet (else download the gated train split)",
    )
    parser.add_argument("--ids", type=Path, default=IDS, help="the committed pair-id file")
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="draw pair ids and write the id file")
    sample.add_argument("--per-pair", type=int, default=50, help="number of attack/control pairs (2 pages each)")
    sample.add_argument("--seed", type=int, default=0)
    build = sub.add_parser("build", help="write the labelled-set JSONL for the committed pair ids")
    build.add_argument("--view", choices=VIEWS, default="markdown")
    build.add_argument("--out", type=Path, help="default .external/browsesafe-plus/train-sample-<view>.jsonl")
    report = sub.add_parser("coverage", help="print the coverage table for the sample")
    report.add_argument("--clip", type=int, default=MAX_CHARS)
    args = parser.parse_args(argv)

    if args.command == "sample":
        pages = read_pages(resolve_paths(args.source), columns=list(IDENTITY_COLUMNS))
        pairs = group_pairs(pages)
        pair_ids = sample_pairs(sorted(pairs), args.per_pair, args.seed)
        payload = {
            "dataset": DATASET,
            "revision": REVISION,
            "split": SPLIT,
            "seed": args.seed,
            "per_pair": args.per_pair,
            "view_default": "markdown",
            "gated": True,
            "labels": {"yes": args.per_pair, "no": args.per_pair},
            "pair_ids": pair_ids,
        }
        args.ids.parent.mkdir(parents=True, exist_ok=True)
        args.ids.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"{len(pair_ids)} pair ids ({2 * len(pair_ids)} pages) in {args.ids}")
        return

    ids = load_ids(args.ids)
    pair_ids = ids.get("pair_ids")
    if not pair_ids:
        raise SystemExit(
            f"{args.ids} has no pair_ids yet (gated HF access was unavailable when this file was scaffolded). "
            f"Request access to {DATASET}, set HF_TOKEN, then run: "
            f"uv run --with pyarrow --with huggingface-hub python scripts/build_browsesafe_plus_set.py sample"
        )

    columns = list(IDENTITY_COLUMNS) + list(VIEW_COLUMN.values())
    pages_by_entry = index_pages(read_pages(resolve_paths(args.source), columns=columns))
    missing = [f"{pid}_{s}" for pid in pair_ids for s in ("malicious", "benign") if f"{pid}_{s}" not in pages_by_entry]
    if missing:
        raise ValueError(f"source is missing {len(missing)} sampled members (first: {missing[0]})")

    if args.command == "build":
        out = args.out or CACHE / f"{SPLIT}-sample-{args.view}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = records(pages_by_entry, pair_ids, args.view)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines), encoding="utf-8")
        print(f"{len(lines)} records ({sum(r['label'] for r in lines)} injected) in {out}")
        return

    print(f"Sample of {len(pair_ids)} pairs (seed {ids['seed']}), clip {args.clip:,} characters:\n")
    print("\n".join(coverage_rows(pages_by_entry, pair_ids, args.clip)))


if __name__ == "__main__":
    main()
