# coding: utf-8
"""Convert a pinned TS-Bench revision into the tool-gate labelled-set shape (#122).

TS-Bench (ToolSafe, Apache-2.0) is not vendored. This script downloads the pinned eval splits, samples ids, and
writes JSONL records ``{"state", "label", "split", "id", "note"}`` under ``.external/tsbench/``. Score 1 is unsafe,
0 is safe; 0.5 (controversial) is dropped from the headline labelled set and listed separately.

    uv run python scripts/build_tsbench_set.py sample --per-split-per-label 150 --seed 122
    uv run python scripts/build_tsbench_set.py build --dest .external/tsbench/sample.jsonl
    uv run python scripts/build_tsbench_set.py check-dedup

``sample`` rewrites ``evals/labelled/tsbench-sample.json`` (ids and hashes only). ``build`` needs the downloaded
splits; a ``--source`` directory can reuse a previous fetch. Every command hashes each split before decoding it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

from s1a.agents.injection_guard import text_of
from s1a.agents.tool_gate import labelled_state, parse_call
from s1a.config import ROOT

DATASET = "MurrayTom/TS-Bench"
REVISION = "19047a0c4a54f54cc1ddd4a2c597105d8453b819"
SPLITS = {
    "AgentDojo-Traj": "data/eval/AgentDojo-Traj.jsonl",
    "ASB-Traj": "data/eval/ASB-Traj.jsonl",
}
SPLIT_SHA256 = {
    "AgentDojo-Traj": "740251908ffbac1f397c71f554678ccf6f73aa6d00be4a63925ea491f6c0ed9b",
    "ASB-Traj": "36261a98f6c38969c4c8d84ead1c3ab333e80ae1f6f4cbc3c9d379bf7d0c1f2a",
}
IDS = ROOT / "evals" / "labelled" / "tsbench-sample.json"
CACHE = ROOT / ".external" / "tsbench" / "raw"
INJECTION_PUBLIC = ROOT / "evals" / "labelled" / "injection-public.jsonl"
SAFE, UNSAFE, CONTROVERSIAL = 0.0, 1.0, 0.5


def hf_url(path: str, revision: str = REVISION) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, split: str) -> Path:
    digest = sha256(path)
    expected = SPLIT_SHA256[split]
    if digest != expected:
        raise ValueError(f"{path}: sha256 {digest}, expected {expected} for {split} at {REVISION}")
    return path


def fetch(split: str, cache: Path = CACHE) -> Path:
    dest = cache / Path(SPLITS[split]).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256(dest) == SPLIT_SHA256[split]:
        return dest
    url = hf_url(SPLITS[split])
    with urlopen(url) as response:  # noqa: S310 - pinned Hugging Face revision URL
        dest.write_bytes(response.read())
    return verify(dest, split)


def score_of(row: dict[str, Any]) -> float:
    try:
        return float(row["score"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"row {row.get('id')!r}: score is not a number") from exc


def band_of(score: float) -> str:
    if score == SAFE:
        return "safe"
    if score == UNSAFE:
        return "unsafe"
    if score == CONTROVERSIAL:
        return "controversial"
    raise ValueError(f"unexpected TS-Bench score {score}")


def load_split(path: Path, split: str) -> list[dict[str, Any]]:
    verify(path, split)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: not JSON: {exc}") from exc
        if not isinstance(row, dict) or not row.get("id"):
            raise ValueError(f"{path}:{line_number}: a row needs an id")
        rows.append(row)
    return rows


def record(row: dict[str, Any], split: str) -> dict[str, Any]:
    """One labelled record; callers must drop controversial rows from the headline set."""
    action = str(row.get("current_action") or "")
    tool, arguments = parse_call(action)
    state = labelled_state(
        {
            "instruction": str(row.get("instruction") or ""),
            "history": str(row.get("history") or ""),
            "tool": tool,
            "arguments": arguments,
            "description": str(row.get("env_info") or ""),
        }
    )
    score = score_of(row)
    return {
        "state": state,
        "label": score == UNSAFE,
        "split": split,
        "id": row["id"],
        "note": f"{split} {row['id']} score={score}",
        "score": score,
    }


def records(rows: Iterable[dict[str, Any]], split: str, *, headline: bool = True) -> list[dict[str, Any]]:
    out = [record(row, split) for row in rows]
    if headline:
        return [item for item in out if item["score"] in (SAFE, UNSAFE)]
    return out


def stratified_sample(rows: list[dict[str, Any]], per_label: int, seed: int) -> list[str]:
    """``per_label`` safe and ``per_label`` unsafe ids (controversial rows are never drawn)."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        band = band_of(score_of(row))
        if band in {"safe", "unsafe"}:
            buckets[band].append(row["id"])
    picked: list[str] = []
    for band in ("safe", "unsafe"):
        ids = buckets[band]
        if len(ids) < per_label:
            raise ValueError(f"{band}: only {len(ids)} rows, need {per_label}")
        picked.extend(random.Random(f"{seed}:{band}").sample(ids, per_label))
    return sorted(picked)


def load_ids(path: Path = IDS) -> dict[str, Any]:
    ids = json.loads(path.read_text(encoding="utf-8"))
    if (ids.get("dataset"), ids.get("revision")) != (DATASET, REVISION):
        raise ValueError(f"{path} names {ids.get('dataset')}@{ids.get('revision')}, not {DATASET}@{REVISION}")
    return ids


def headline_jsonl(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        row = {k: v for k, v in item.items() if k != "score"}
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    return "".join(lines)


def injection_texts() -> set[str]:
    texts = set()
    if not INJECTION_PUBLIC.is_file():
        return texts
    for line in INJECTION_PUBLIC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        state = rec.get("state")
        texts.add(text_of(state.get("text") if isinstance(state, dict) else state).strip())
    return texts


def verbatim_overlap(items: list[dict[str, Any]], corpus: set[str]) -> list[str]:
    """Ids whose instruction, history, or arguments appear verbatim in injection-public tool-output text."""
    hits = []
    for item in items:
        state = item["state"]
        blobs = [str(state.get(key) or "").strip() for key in ("instruction", "history", "arguments")]
        if any(blob and blob in corpus for blob in blobs):
            hits.append(item["id"])
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="directory of already-downloaded split jsonl files")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--per-split-per-label", type=int, default=150)
    sample.add_argument("--seed", type=int, default=122)
    sample.add_argument("--out", type=Path, default=IDS)
    build = commands.add_parser("build")
    build.add_argument("--dest", type=Path, default=Path(".external/tsbench/sample.jsonl"))
    commands.add_parser("check-dedup")
    args = parser.parse_args(argv)

    def split_path(name: str) -> Path:
        if args.source:
            candidate = args.source / Path(SPLITS[name]).name
            return verify(candidate, name)
        return fetch(name)

    if args.command == "sample":
        listing: dict[str, Any] = {
            "dataset": DATASET,
            "revision": REVISION,
            "license": "Apache-2.0",
            "seed": args.seed,
            "per_split_per_label": args.per_split_per_label,
            "splits": {},
            "sha256": dict(SPLIT_SHA256),
        }
        for name in SPLITS:
            rows = load_split(split_path(name), name)
            ids = stratified_sample(rows, args.per_split_per_label, args.seed)
            by_band = defaultdict(int)
            wanted = set(ids)
            for row in rows:
                if row["id"] in wanted:
                    by_band[band_of(score_of(row))] += 1
            listing["splits"][name] = {"ids": ids, "counts": dict(by_band)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(listing, indent=1) + "\n", encoding="utf-8")
        print(args.out)
        return 0

    ids = load_ids()
    if args.command == "build":
        items: list[dict[str, Any]] = []
        controversial = 0
        for name, spec in ids["splits"].items():
            wanted = set(spec["ids"])
            for row in load_split(split_path(name), name):
                if row["id"] not in wanted:
                    continue
                item = record(row, name)
                if item["score"] == CONTROVERSIAL:
                    controversial += 1
                    continue
                items.append(item)
        items.sort(key=lambda item: (item["split"], item["id"]))
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(headline_jsonl(items), encoding="utf-8")
        print(f"{args.dest}: {len(items)} headline records, {controversial} controversial dropped")
        return 0

    # check-dedup
    items = []
    for name, spec in ids["splits"].items():
        wanted = set(spec["ids"])
        for row in load_split(split_path(name), name):
            if row["id"] in wanted:
                items.append(record(row, name))
    hits = verbatim_overlap(items, injection_texts())
    print(
        "no verbatim overlap with injection-public.jsonl"
        if not hits
        else f"verbatim overlap with injection-public ({len(hits)}): " + ", ".join(hits[:20])
    )
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
