# coding: utf-8
"""The committed slice and the task metadata it points at.

The slice file names a Hugging Face dataset revision, a split and task ids. The tasks themselves (``task.toml``,
``instruction.md`` and ``tests/``) are downloaded on demand at that revision and never vendored: WebHarbor's reuse
licence is unclear (docs/webharbor.md).
"""

from __future__ import annotations

import json
import random
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SLICE_PATH = Path(__file__).with_name("slice.json")
TASK_ID = re.compile(r"^[a-z0-9_-]+--\d+$")
STATE_CHANGE = ("read_only", "writes", "unchecked")
REVISION = re.compile(r"^[0-9a-f]{40}$")


class SliceError(ValueError):
    """The slice file, the registry or a task's metadata is malformed or disagrees with the slice."""


@dataclass(frozen=True)
class SliceTask:
    id: str
    state_change: str = "unchecked"


@dataclass(frozen=True)
class Slice:
    name: str
    repo: str
    registry_revision: str
    dataset: str
    dataset_version: str
    webharbor_revision: str
    site: str
    image: str
    port: int
    seed: int
    tasks: tuple[SliceTask, ...]

    @property
    def dataset_ref(self) -> str:
        return f"{self.dataset}@{self.dataset_version}"


def _need(data: dict[str, Any], key: str, kind: type) -> Any:
    value = data.get(key)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise SliceError(f"slice field {key!r} must be a {kind.__name__}, got {value!r}")
    return value


def parse_slice(data: Any) -> Slice:
    """A validated ``Slice``: revisions are full SHAs, ids are ``<site>--<n>`` on the slice's site, none repeated."""
    if not isinstance(data, dict):
        raise SliceError("the slice must be a JSON object")
    for key in ("registry_revision", "webharbor_revision"):
        if not REVISION.match(str(data.get(key, ""))):
            raise SliceError(f"slice field {key!r} must be a full 40-character commit SHA")
    site = _need(data, "site", str)
    raw_tasks = _need(data, "tasks", list)
    if not raw_tasks:
        raise SliceError("the slice names no tasks")
    tasks: list[SliceTask] = []
    for raw in raw_tasks:
        entry = {"id": raw} if isinstance(raw, str) else raw
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not TASK_ID.match(entry["id"]):
            raise SliceError(f"bad task entry {raw!r}: want an id like '{site}--0'")
        if entry["id"].split("--")[0] != site:
            raise SliceError(f"task {entry['id']} is not on the slice's site {site!r}")
        state = entry.get("state_change", "unchecked")
        if state not in STATE_CHANGE:
            raise SliceError(f"task {entry['id']}: state_change must be one of {STATE_CHANGE}, got {state!r}")
        tasks.append(SliceTask(entry["id"], state))
    ids = [task.id for task in tasks]
    if len(set(ids)) != len(ids):
        raise SliceError("the slice repeats a task id")
    port = _need(data, "port", int)
    if not 1 <= port <= 65535:
        raise SliceError(f"slice port {port} is out of range")
    return Slice(
        name=_need(data, "name", str),
        repo=_need(data, "repo", str),
        registry_revision=data["registry_revision"],
        dataset=_need(data, "dataset", str),
        dataset_version=_need(data, "dataset_version", str),
        webharbor_revision=data["webharbor_revision"],
        site=site,
        image=_need(data, "image", str),
        port=port,
        seed=_need(data, "seed", int),
        tasks=tuple(tasks),
    )


def load_slice(path: Path = SLICE_PATH) -> Slice:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SliceError(f"{path} is not JSON: {exc}") from None
    return parse_slice(data)


def registry_tasks(registry: Any, dataset: str, version: str) -> list[str]:
    """The task names one dataset of a Harbor ``registry.json`` lists."""
    if not isinstance(registry, list):
        raise SliceError("registry.json must be a list of datasets")
    for entry in registry:
        if isinstance(entry, dict) and entry.get("name") == dataset and str(entry.get("version")) == version:
            return [str(task["name"]) for task in entry.get("tasks") or [] if isinstance(task, dict) and "name" in task]
    raise SliceError(f"registry has no dataset {dataset}@{version}")


def check_against_registry(slice_: Slice, registry: Any) -> None:
    """Every slice task is in the slice's dataset; raises ``SliceError`` naming the missing ones."""
    listed = set(registry_tasks(registry, slice_.dataset, slice_.dataset_version))
    missing = [task.id for task in slice_.tasks if task.id not in listed]
    if missing:
        raise SliceError(f"not in {slice_.dataset_ref}: {', '.join(missing)}")


def select(registry: Any, dataset: str, version: str, site: str, n: int, seed: int) -> list[str]:
    """``n`` task ids on ``site`` drawn with ``random.Random(seed)``, sorted by index: how ``slice.json`` was made."""
    pool = sorted(
        (name for name in registry_tasks(registry, dataset, version) if name.split("--")[0] == site),
        key=lambda name: int(name.split("--")[1]),
    )
    if n < 1 or n > len(pool):
        raise SliceError(f"cannot pick {n} of the {len(pool)} {site} tasks in {dataset}@{version}")
    return sorted(random.Random(seed).sample(pool, n), key=lambda name: int(name.split("--")[1]))


@dataclass(frozen=True)
class TaskSpec:
    """What the runner needs from one downloaded task's ``task.toml`` and ``instruction.md``."""

    id: str
    site: str
    port: int
    start_url: str
    image: str
    verify_module: str
    question: str
    instruction: str
    webharbor_revision: str
    env: tuple[tuple[str, str], ...] = ()  # ``[environment.env]``: the site container's WH_* variables


def parse_task(task_id: str, toml_text: str, instruction: str) -> TaskSpec:
    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as exc:
        raise SliceError(f"{task_id}: task.toml is not TOML: {exc}") from None
    env = (data.get("environment") or {}).get("env") or {}
    meta = data.get("metadata") or {}
    try:
        port = int(env["WH_PORT"])
        spec = TaskSpec(
            id=task_id,
            site=str(env["WH_SITE"]),
            port=port,
            start_url=str(env.get("WH_START_URL") or f"http://localhost:{port}/"),
            image=str(data["environment"]["docker_image"]),
            verify_module=str(env["WH_VERIFY_MODULE"]),
            question=str(env.get("WH_TASK_QUES") or ""),
            instruction=instruction.strip(),
            webharbor_revision=str(meta.get("webharbor_revision") or ""),
            env=tuple((str(key), str(value)) for key, value in env.items()),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SliceError(f"{task_id}: task.toml lacks {exc}") from None
    if not spec.instruction:
        raise SliceError(f"{task_id}: instruction.md is empty")
    return spec


def check_task(slice_: Slice, spec: TaskSpec) -> None:
    """A downloaded task agrees with the slice: same site, port, image and WebHarbor revision."""
    expected = {"site": slice_.site, "port": slice_.port, "image": slice_.image}
    if slice_.webharbor_revision and spec.webharbor_revision:
        expected["webharbor_revision"] = slice_.webharbor_revision
    for key, value in expected.items():
        if getattr(spec, key) != value:
            raise SliceError(f"{spec.id}: {key} is {getattr(spec, key)!r}, the slice pins {value!r}")


def load_task(slice_: Slice, task_dir: Path) -> TaskSpec:
    """A downloaded task directory (``task.toml``, ``instruction.md``, ``tests/``) checked against the slice."""
    toml_path, instruction_path = task_dir / "task.toml", task_dir / "instruction.md"
    for path in (toml_path, instruction_path, task_dir / "tests"):
        if not path.exists():
            raise SliceError(f"{task_dir.name}: missing {path.name}")
    spec = parse_task(
        task_dir.name, toml_path.read_text(encoding="utf-8"), instruction_path.read_text(encoding="utf-8")
    )
    check_task(slice_, spec)
    return spec


def download(slice_: Slice, dest: Path) -> Path:
    """The slice's task folders and ``registry.json`` at the pinned revision, from Hugging Face, into ``dest``."""
    from huggingface_hub import snapshot_download  # an optional, network-only import

    patterns = ["registry.json"] + [f"tasks/{task.id}/**" for task in slice_.tasks]
    root = snapshot_download(
        repo_id=slice_.repo,
        repo_type="dataset",
        revision=slice_.registry_revision,
        allow_patterns=patterns,
        local_dir=dest,
    )
    return Path(root)
