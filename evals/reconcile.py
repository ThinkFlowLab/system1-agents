# coding: utf-8
"""Reconcile planned evaluation attempts with saved trial result artifacts.

Results are matched by the existing ``(eval folder, model label, task_name)`` record fields. The
reconciler reports denominator coverage and artifact mismatches; it does not compute or change scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PlanStatus = Literal["planned", "interrupted", "unsupported"]
AttemptKey = tuple[str, str, str]


@dataclass(frozen=True)
class PlannedAttempt:
    attempt_id: str
    eval_name: str
    model: str
    task_name: str
    status: PlanStatus
    reason: str | None = None

    @property
    def key(self) -> AttemptKey:
        return self.eval_name, self.model, self.task_name


@dataclass(frozen=True)
class ResultArtifact:
    eval_name: str
    model: str
    task_name: str
    errored: bool
    path: str

    @property
    def key(self) -> AttemptKey:
        return self.eval_name, self.model, self.task_name


@dataclass(frozen=True)
class ReconciledAttempt:
    attempt_id: str
    eval_name: str
    model: str
    task_name: str
    status: str
    reason: str | None = None
    artifact: str | None = None


def load_plan(path: Path) -> list[PlannedAttempt]:
    """Load and validate a version-1 attempt plan JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read attempt plan {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("attempt plan must be an object with schema_version: 1")
    raw_attempts = payload.get("attempts")
    if not isinstance(raw_attempts, list) or not raw_attempts:
        raise ValueError("attempt plan must contain a non-empty attempts list")

    attempts: list[PlannedAttempt] = []
    ids: set[str] = set()
    keys: set[AttemptKey] = set()
    for index, item in enumerate(raw_attempts):
        if not isinstance(item, dict):
            raise ValueError(f"attempts[{index}] must be an object")
        text_fields = {name: item.get(name) for name in ("id", "eval", "model", "task_name", "status")}
        if any(not isinstance(value, str) or not value.strip() for value in text_fields.values()):
            raise ValueError(f"attempts[{index}] requires non-empty string id, eval, model, task_name and status")
        attempt_id = str(item["id"]).strip()
        eval_name = str(item["eval"]).strip()
        model = str(item["model"]).strip()
        task_name = str(item["task_name"]).strip()
        status = str(item["status"]).strip()
        if status not in {"planned", "interrupted", "unsupported"}:
            raise ValueError(f"attempts[{index}] has unsupported status {status!r}")
        reason = item.get("reason")
        if status in {"interrupted", "unsupported"} and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(f"attempts[{index}] with status {status!r} requires a non-empty reason")
        if attempt_id in ids:
            raise ValueError(f"duplicate attempt id {attempt_id!r}")
        key = eval_name, model, task_name
        if key in keys:
            raise ValueError(f"duplicate result match key {key!r}; task_name must be unique per eval and model")
        ids.add(attempt_id)
        keys.add(key)
        attempts.append(
            PlannedAttempt(
                attempt_id=attempt_id,
                eval_name=eval_name,
                model=model,
                task_name=task_name,
                status=status,  # type: ignore[arg-type]
                reason=reason.strip() if isinstance(reason, str) and reason.strip() else None,
            )
        )
    return attempts


def _model_label(result: dict[str, Any]) -> str:
    """Use the same label as evals.table, after any ``<suite>/`` prefix."""
    name = str((result.get("agent_info") or {}).get("name") or "")
    return name.partition("/")[2] or name


def read_artifacts(results_root: Path) -> list[ResultArtifact]:
    """Read every saved trial result using the same folder shape as evals.table."""
    if not results_root.is_dir():
        raise ValueError(f"results root is not a directory: {results_root}")
    artifacts = []
    for result_file in sorted(results_root.glob("*/*/*/result.json")):
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read result artifact {result_file}: {exc}") from exc
        if not isinstance(result, dict):
            raise ValueError(f"result artifact must contain an object: {result_file}")
        task_name = result.get("task_name")
        artifacts.append(
            ResultArtifact(
                eval_name=result_file.relative_to(results_root).parts[0],
                model=_model_label(result),
                task_name=str(task_name) if isinstance(task_name, str) else "",
                errored=result.get("exception_info") is not None,
                path=result_file.relative_to(results_root).as_posix(),
            )
        )
    return artifacts


def reconcile(attempts: list[PlannedAttempt], artifacts: list[ResultArtifact]) -> list[ReconciledAttempt]:
    """Match planned attempts to their exact saved task results; retain every extra artifact as unplanned."""
    by_key: dict[AttemptKey, list[ResultArtifact]] = {}
    for artifact in artifacts:
        by_key.setdefault(artifact.key, []).append(artifact)
    rows: list[ReconciledAttempt] = []
    for attempt in attempts:
        matches = by_key.get(attempt.key, [])
        artifact = matches.pop(0) if matches else None
        if artifact is None:
            status = "missing" if attempt.status == "planned" else attempt.status
            rows.append(
                ReconciledAttempt(
                    attempt_id=attempt.attempt_id,
                    eval_name=attempt.eval_name,
                    model=attempt.model,
                    task_name=attempt.task_name,
                    status=status,
                    reason=attempt.reason,
                )
            )
            continue
        if attempt.status == "unsupported":
            status, reason = "unexpected-result", attempt.reason
        else:
            status = "error" if artifact.errored else "recorded"
            reason = f"Manifest says interrupted: {attempt.reason}" if attempt.status == "interrupted" else None
        rows.append(
            ReconciledAttempt(
                attempt_id=attempt.attempt_id,
                eval_name=attempt.eval_name,
                model=attempt.model,
                task_name=attempt.task_name,
                status=status,
                reason=reason,
                artifact=artifact.path,
            )
        )
    for remaining in by_key.values():
        for artifact in remaining:
            rows.append(
                ReconciledAttempt(
                    attempt_id="",
                    eval_name=artifact.eval_name,
                    model=artifact.model,
                    task_name=artifact.task_name,
                    status="unplanned-result",
                    reason="Result has no matching entry in the attempt plan.",
                    artifact=artifact.path,
                )
            )
    return rows


def markdown(rows: list[ReconciledAttempt]) -> str:
    """Render the attempt denominator and artifact status without changing score statistics."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    planned_total = sum(bool(row.attempt_id) for row in rows)
    lines = [
        f"Planned attempts: {planned_total}; recorded outcomes: "
        f"{counts.get('recorded', 0) + counts.get('error', 0) + counts.get('unexpected-result', 0)}; "
        f"missing: {counts.get('missing', 0)}; interrupted: {counts.get('interrupted', 0)}; "
        f"unsupported: {counts.get('unsupported', 0)}; errors: {counts.get('error', 0)}; "
        f"unexpected results: {counts.get('unexpected-result', 0) + counts.get('unplanned-result', 0)}.",
        "",
        "| Attempt | Eval | Model | Task | Status | Artifact | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = (
            row.attempt_id,
            row.eval_name,
            row.model,
            row.task_name,
            row.status,
            row.artifact or "",
            row.reason or "",
        )
        lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    return "\n".join(lines)


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def main() -> None:
    """Print a manifest-to-artifact coverage report."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, help="results root, e.g. evals/results")
    parser.add_argument("attempt_plan", type=Path, help="version-1 JSON file declaring planned attempts")
    args = parser.parse_args()
    print(markdown(reconcile(load_plan(args.attempt_plan), read_artifacts(args.results_root))))


if __name__ == "__main__":
    main()
