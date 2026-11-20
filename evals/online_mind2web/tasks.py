# coding: utf-8
"""The committed Online-Mind2Web slice and the official task file it points at.

The slice names a Hugging Face dataset revision, a seed and task ids. Task text is downloaded on demand at that
revision and is not vendored: the official file is gated and released under CC BY 4.0.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SLICE_PATH = Path(__file__).with_name("slice.json")
REVISION = re.compile(r"^[0-9a-f]{40}$")
TASK_ID = re.compile(r"^[0-9a-f]{32}(?:_[0-9a-z]+)?$", re.IGNORECASE)
LEVELS = ("easy", "medium", "hard")
HF_RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{source_file}"

# Login, payment, cart, favorites and other account-writes: these stay out of the public-read pool.
SKIP = re.compile(
    r"("
    r"\b(log[\s-]?in|sign[\s-]?in|sign[\s-]?up|create an account|my account|your account|"
    r"check[\s-]?out|purchase|payment|credit card|home store|set (?:it|this) as|"
    r"subscribe|password|register|book a |reserve |reservation|"
    r"submit a (?:request|form|application)|liked by)\b"
    r"|add .{0,80}to (?:the |my |your )?(?:cart|favorites?|wishlist)"
    r"|to (?:the |my |your )?(?:cart|favorites?|wishlist)"
    r"|\bbuy\b"
    r")",
    re.IGNORECASE,
)


class SliceError(ValueError):
    """The slice file or the official task file is malformed or disagrees with the slice."""


@dataclass(frozen=True)
class SliceTask:
    id: str
    level: str
    website: str
    reference_length: int


@dataclass(frozen=True)
class Slice:
    name: str
    dataset: str
    source_file: str
    revision: str
    expected_tasks: int
    seed: int
    n_easy: int
    n_medium: int
    selection: str
    tasks: tuple[SliceTask, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(task.id for task in self.tasks)


@dataclass(frozen=True)
class OfficialTask:
    id: str
    website: str
    confirmed_task: str
    reference_length: int
    level: str


def _need(data: dict[str, Any], key: str, kind: type) -> Any:
    value = data.get(key)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is int):
        raise SliceError(f"slice field {key!r} must be a {kind.__name__}, got {value!r}")
    return value


def parse_slice(data: Any) -> Slice:
    """A validated ``Slice``: the revision is a full SHA, ids are unique Online-Mind2Web hashes, levels are known."""
    if not isinstance(data, dict):
        raise SliceError("the slice must be a JSON object")
    if not REVISION.match(str(data.get("revision", ""))):
        raise SliceError("slice field 'revision' must be a full 40-character commit SHA")
    raw_tasks = _need(data, "tasks", list)
    if not raw_tasks:
        raise SliceError("the slice names no tasks")
    tasks: list[SliceTask] = []
    for raw in raw_tasks:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not TASK_ID.match(raw["id"]):
            raise SliceError(f"bad task entry {raw!r}: want an id like a 32-character hex hash")
        level = raw.get("level")
        if level not in LEVELS:
            raise SliceError(f"task {raw['id']}: level must be one of {LEVELS}, got {level!r}")
        website = raw.get("website")
        if not isinstance(website, str) or urlparse(website).scheme not in {"http", "https"}:
            raise SliceError(f"task {raw['id']}: website must be an http(s) URL, got {website!r}")
        length = raw.get("reference_length")
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise SliceError(f"task {raw['id']}: reference_length must be a positive int, got {length!r}")
        tasks.append(SliceTask(raw["id"], level, website, length))
    ids = [task.id for task in tasks]
    if len(set(ids)) != len(ids):
        raise SliceError("the slice repeats a task id")
    n_easy, n_medium = _need(data, "n_easy", int), _need(data, "n_medium", int)
    if n_easy < 0 or n_medium < 0 or n_easy + n_medium == 0:
        raise SliceError("n_easy and n_medium must be non-negative and not both zero")
    if sum(1 for task in tasks if task.level == "easy") != n_easy:
        raise SliceError(f"the slice lists {sum(1 for t in tasks if t.level == 'easy')} easy tasks, expected {n_easy}")
    if sum(1 for task in tasks if task.level == "medium") != n_medium:
        raise SliceError(
            f"the slice lists {sum(1 for t in tasks if t.level == 'medium')} medium tasks, expected {n_medium}"
        )
    expected = _need(data, "expected_tasks", int)
    if expected < n_easy + n_medium:
        raise SliceError("expected_tasks is smaller than the slice")
    return Slice(
        name=_need(data, "name", str),
        dataset=_need(data, "dataset", str),
        source_file=_need(data, "source_file", str),
        revision=data["revision"],
        expected_tasks=expected,
        seed=_need(data, "seed", int),
        n_easy=n_easy,
        n_medium=n_medium,
        selection=_need(data, "selection", str),
        tasks=tuple(tasks),
    )


def load_slice(path: Path = SLICE_PATH) -> Slice:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SliceError(f"{path} is not JSON: {exc}") from None
    return parse_slice(data)


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise SliceError(f"task row is missing any of {names}: {row!r}")


def parse_official(data: Any) -> list[OfficialTask]:
    """The official ``Online_Mind2Web.json`` list (or a dict keyed by task id)."""
    if isinstance(data, dict) and all(isinstance(value, dict) for value in data.values()):
        rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        raise SliceError("the official file must be a JSON list of tasks or an object keyed by task id")
    tasks: list[OfficialTask] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise SliceError(f"official task is not an object: {raw!r}")
        task_id = str(_field(raw, "task_id", "id")).strip()
        if not TASK_ID.match(task_id):
            raise SliceError(f"bad official task id {task_id!r}")
        if task_id in seen:
            raise SliceError(f"official file repeats task id {task_id}")
        seen.add(task_id)
        level = str(_field(raw, "level")).lower()
        if level not in LEVELS:
            raise SliceError(f"task {task_id}: level must be one of {LEVELS}, got {level!r}")
        website = str(_field(raw, "website"))
        if urlparse(website).scheme not in {"http", "https"}:
            raise SliceError(f"task {task_id}: website must be an http(s) URL, got {website!r}")
        length = _field(raw, "reference_length")
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise SliceError(f"task {task_id}: reference_length must be a positive int, got {length!r}")
        text = str(_field(raw, "confirmed_task", "task_description", "task")).strip()
        if not text:
            raise SliceError(f"task {task_id} has an empty task description")
        tasks.append(OfficialTask(task_id, website, text, length, level))
    if not tasks:
        raise SliceError("the official file names no tasks")
    return tasks


def load_official(path: Path) -> list[OfficialTask]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SliceError(f"{path} is not JSON: {exc}") from None
    return parse_official(data)


def is_public_read(task: OfficialTask) -> bool:
    """True when the task text does not ask for a login, payment, cart, favorites or similar account write."""
    return SKIP.search(task.confirmed_task) is None


def eligible_ids(tasks: list[OfficialTask], level: str) -> list[str]:
    if level not in LEVELS:
        raise SliceError(f"level must be one of {LEVELS}, got {level!r}")
    return sorted(task.id for task in tasks if task.level == level and is_public_read(task))


def select_ids(tasks: list[OfficialTask], *, seed: int, n_easy: int, n_medium: int) -> list[str]:
    """The documented seed: ``sample`` from the sorted public-read easy ids, then the medium ids; return sorted ids."""
    easy, medium = eligible_ids(tasks, "easy"), eligible_ids(tasks, "medium")
    if len(easy) < n_easy:
        raise SliceError(f"need {n_easy} public-read easy tasks, the official file has {len(easy)}")
    if len(medium) < n_medium:
        raise SliceError(f"need {n_medium} public-read medium tasks, the official file has {len(medium)}")
    rng = random.Random(seed)
    return sorted(rng.sample(easy, n_easy) + rng.sample(medium, n_medium))


def check_selection(slice_: Slice, tasks: list[OfficialTask]) -> None:
    """The official file has the expected size and the seed reproduces the committed ids, levels and websites."""
    if len(tasks) != slice_.expected_tasks:
        raise SliceError(f"official file has {len(tasks)} tasks, slice expected {slice_.expected_tasks}")
    by_id = {task.id: task for task in tasks}
    missing = [task_id for task_id in slice_.ids if task_id not in by_id]
    if missing:
        raise SliceError(f"official file is missing slice ids: {missing}")
    chosen = select_ids(tasks, seed=slice_.seed, n_easy=slice_.n_easy, n_medium=slice_.n_medium)
    if chosen != list(slice_.ids):
        raise SliceError("the seed does not reproduce the committed slice ids from this official file")
    for item in slice_.tasks:
        official = by_id[item.id]
        if (official.level, official.website, official.reference_length) != (
            item.level,
            item.website,
            item.reference_length,
        ):
            raise SliceError(
                f"task {item.id} in the official file is {official.level} {official.website} "
                f"len={official.reference_length}, slice says {item.level} {item.website} len={item.reference_length}"
            )


def official_url(slice_: Slice) -> str:
    return HF_RESOLVE.format(dataset=slice_.dataset, revision=slice_.revision, source_file=slice_.source_file)


def fetch_official(slice_: Slice, dest: Path, *, token: str | None = None, opener=None) -> Path:
    """Download the official JSON to ``dest`` (a file). Needs a Hugging Face token with access to the gated dataset."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    auth = token if token is not None else os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not auth:
        raise SliceError(
            "the official Online-Mind2Web file is gated; set HF_TOKEN after accepting the dataset terms at "
            f"https://huggingface.co/datasets/{slice_.dataset}"
        )
    request = urllib.request.Request(
        official_url(slice_),
        headers={"Authorization": f"Bearer {auth}", "User-Agent": "system1-agents-online-mind2web"},
    )
    try:
        with (opener or urllib.request.urlopen)(request) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise SliceError(f"download failed ({exc.code}): {exc.reason}") from None
    except urllib.error.URLError as exc:
        raise SliceError(f"download failed: {exc.reason}") from None
    dest.write_bytes(body)
    check_selection(slice_, load_official(dest))
    return dest


def goal(task: OfficialTask) -> str:
    """The browse goal: go to the official start URL and finish the official task text, then give a final answer."""
    return (
        f"Go to {task.website} and complete this task:\n\n{task.confirmed_task}\n\n"
        "When finished, provide your final answer as text."
    )


def bind(slice_: Slice, official: list[OfficialTask]) -> list[OfficialTask]:
    """The official records for the slice, in slice order, after ``check_selection``."""
    check_selection(slice_, official)
    by_id = {task.id: task for task in official}
    return [by_id[task_id] for task_id in slice_.ids]
