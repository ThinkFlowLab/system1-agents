"""Producer-shaped CPU evidence; framework stubs are confined to these tests."""

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from benchmarks.evidence import digest, dump
from benchmarks.quality_ticket import aggregate, load_preparation


@pytest.fixture
def preparation(monkeypatch, tmp_path):
    path = os.environ.get("TICKET_PREPARATION_ROOT")
    if not path:
        pytest.skip("Set TICKET_PREPARATION_ROOT to the frozen producer preparation")
    # Execute the actual validator and environment, without installing or initializing the framework.
    if importlib.util.find_spec("openjiuwen") is None:
        for name in ("openjiuwen.core.common.exception.codes", "openjiuwen.core.common.exception.errors"):
            module = ModuleType(name)
            module.StatusCode = object()
            module.BaseError = ValueError
            module.build_error = lambda *a, **k: ValueError("framework stub")
            monkeypatch.setitem(sys.modules, name, module)
        package = ModuleType("s1a.decision_models")
        package.__path__ = [str(Path(__file__).resolve().parents[2] / "s1a/decision_models")]
        monkeypatch.setitem(sys.modules, "s1a.decision_models", package)
        config = ModuleType("s1a.config")
        config.HOME = tmp_path
        monkeypatch.setitem(sys.modules, "s1a.config", config)
        rail = ModuleType("openjiuwen.core.single_agent.rail.base")
        rail.AgentCallbackContext = type("AgentCallbackContext", (), {})
        rail.AgentCallbackEvent = type(
            "AgentCallbackEvent", (), {"BEFORE_TOOL_CALL": "before", "AFTER_TOOL_CALL": "after"}
        )
        monkeypatch.setitem(sys.modules, rail.__name__, rail)
        logging = ModuleType("openjiuwen.core.common.logging")
        logging.logger = None
        monkeypatch.setitem(sys.modules, logging.__name__, logging)
    return load_preparation(Path(path))


def body(value):
    raw = dump(value).encode()
    return dict(bytes=len(raw), sha256=digest(raw), base64=base64.b64encode(raw).decode(), utf8=raw.decode())


@pytest.fixture
def measurement(tmp_path, preparation):
    prep, binding, producer = preparation
    root = tmp_path / "cpu-ticket-fixture"
    root.mkdir()
    (root / "binding.json").write_bytes((prep / "binding.json").read_bytes())
    (root / "source.json").write_text(
        dump(
            dict(
                binding_sha256=digest((prep / "binding.json").read_bytes()),
                files=binding["source_sha256"],
                runtime_revision=binding["agents_revision"],
            )
        )
    )
    models = [
        dict(name=name, arm=f"/remote/{root.name}/{name}", collector_exit_code=0) for name in binding["service_order"]
    ]
    (root / "status.json").write_text(
        dump(dict(job_id="123", planned_models=binding["service_order"], planned_tickets_per_model=30, models=models))
    )
    from s1a.agents.ticket_router import PUBLIC_FIELDS, QUEUES, RULES, load_tickets
    import random

    rows = load_tickets(prep / "runtime/s1a/agents/_data/ticket_router_eval.jsonl")
    rows = [{key: row[key] for key in (*PUBLIC_FIELDS, "label") if key in row} for row in rows]
    random.Random(0).shuffle(rows)
    name = "open-9b"
    arm = root / name
    arm.mkdir()
    remote = f"/remote/{root.name}/{name}"
    assets = {
        "binary_sha256": {"/remote/worker": "a" * 64, "/remote/frontend": "c" * 64},
        "export": {"model_id": "fixture"},
        "verified_export_files": {"weights": {"sha256": "b" * 64}},
    }
    identity = dict(
        benchmark_run_id=root.name,
        planned_model=name,
        requested_model="open-jev-9b",
        job_id="123",
        source_binding_sha256=digest((prep / "binding.json").read_bytes()),
        actual_assets=assets,
        identity_state="current_exact_assets_recorded",
        gpu={"name": "CPU mock; no GPU"},
        model_identity_sha256=digest(json.dumps(assets, ensure_ascii=False, sort_keys=True).encode()),
    )
    (arm / "identity.json").write_text(dump(identity))
    (arm / "launch.json").write_text(dump(dict(worker="/remote/worker", frontend="/remote/frontend")))
    processes = {}
    for role, sha in (("worker", "a" * 64), ("frontend", "c" * 64)):
        maps = f"1000-2000 r-xp 00000000 00:00 1 /remote/{role}\n".encode()
        processes[role] = dict(
            pid=1 if role == "worker" else 2,
            exe=f"/remote/{role}",
            argv=[f"/remote/{role}"],
            exe_sha256=sha,
            maps_sha256=digest(maps),
            loaded_libraries=[],
        )
        for phase in ("ready", "after"):
            (arm / f"{role}-{phase}.maps").write_bytes(maps)
    for phase in ("ready", "after"):
        (arm / f"process-{phase}.json").write_text(dump(processes))
    planned = [json.loads(line) for line in (prep / "planned-cases.jsonl").read_text().splitlines()]
    (arm / "planned-initial.json").write_text(
        dump(
            dict(
                planned_denominator=30,
                cases=planned,
                identity={
                    "benchmark_run_id": root.name,
                    "planned_model": "open-9b",
                    "identity_state": "planned_contract_only",
                },
            )
        )
    )
    plan_identity = {key: identity[key] for key in ("benchmark_run_id", "model_identity_sha256")}
    plan_identity["record"] = "identity.json"
    (arm / "planned-cases.json").write_text(
        dump(
            dict(
                identity=plan_identity,
                planned_denominator=30,
                cases=[
                    {
                        **case,
                        "logical_key": [
                            root.name,
                            identity["model_identity_sha256"],
                            "ticket_router",
                            0,
                            case["ticket_id"],
                        ],
                    }
                    for case in planned
                ],
            )
        )
    )
    modules = {
        key[8:-3].replace("/", "."): {"path": "/remote/source/" + key[8:], "sha256": value}
        for key, value in binding["source_sha256"].items()
        if key.startswith("runtime/s1a/") and key.endswith(".py")
    }
    for key in list(modules):
        if key.endswith(".__init__"):
            modules[key[:-9]] = modules.pop(key)
    (arm / "python-identity.json").write_text(
        dump(
            dict(
                executable="/remote/python",
                executable_sha256="c" * 64,
                version="fixture",
                modules=modules,
                expected_s1a_root="/remote/source",
                actual_s1a_root="/remote/source/s1a",
                environment={
                    "MODEL_NAME": "",
                    "S1A_HOME": remote + "/s1a-home",
                    "S1A_DECISION_TIMEOUT_S": "120",
                    "LAYA_SERVED_TIMEOUT_S": "120",
                    "TYPESAFE_MODEL": "open-jev-9b",
                },
            )
        )
    )
    framework = b"# synthetic source capture containing result_type\n"
    (arm / "framework.py").write_bytes(framework)
    framework_sha = digest(framework)
    modules["openjiuwen.fixture"] = {"path": "/remote/framework.py", "sha256": framework_sha}
    (arm / "python-loaded-after.json").write_text(dump(modules))
    (arm / "framework-result-type-source.json").write_text(
        dump(
            dict(
                normal_result_type_allowlist=["answer"],
                gate_basis="conservative_provisional_pending_actual_framework_source_verification",
                candidates=[
                    dict(
                        module="openjiuwen.fixture",
                        path="/remote/framework.py",
                        copy=remote + "/framework.py",
                        bytes=len(framework),
                        sha256=framework_sha,
                        loaded_module_record_sha256=framework_sha,
                    )
                ],
            )
        )
    )
    argv = [
        "s1a",
        "run",
        "ticket_router",
        "--model",
        "jev",
        "--rethink",
        "off",
        "--episodes",
        "1",
        "--seed",
        "0",
        "--max-steps",
        "30",
        "--timeout",
        "300",
        "--dataset",
        "/remote/source/s1a/agents/_data/ticket_router_eval.jsonl",
        "--batch-size",
        "30",
        "--log",
    ]
    events = []
    compact = {key: identity[key] for key in ("benchmark_run_id", "model_identity_sha256")}
    compact["record"] = "identity.json"

    def add(kind, **fields):
        events.append(
            dict(
                event_index=len(events) + 1,
                utc="fixture",
                monotonic_ns=len(events) + 1,
                kind=kind,
                identity=compact,
                **fields,
            )
        )
        return len(events)

    def pair(kind, start, terminal):
        index = add(kind + "_start", **start)
        return add(kind + "_terminal", start_event=index, status="returned", **terminal)

    def observation(index):
        if index == 30:
            return {"ticket": None, "progress": {"ticket_id": None}}
        return {
            "ticket": {key: rows[index][key] for key in PUBLIC_FIELDS if key in rows[index]},
            "progress": {"ticket_id": rows[index]["id"]},
        }

    def view(index):
        return dict(
            state=observation(index), candidates={} if index == 30 else QUEUES, done=index == 30, score=float(index)
        )

    add("cli_start", argv=argv)
    for stage in ("Runner.start", "build_model", "make_series", "DecisionModel.warm", "TicketRouterEnv.__init__"):
        pair("lifecycle", {"stage": stage}, {"stage": stage})
    episode_start = add("lifecycle_start", stage="run_episode")
    pair("env_reset", {}, dict(view=observation(0), candidates=QUEUES))
    pair("lifecycle", {"stage": "DeepAgent.ensure_initialized"}, {"stage": "DeepAgent.ensure_initialized"})
    runner_start = add("lifecycle_start", stage="Runner.run_agent")
    ticks, routes = [], []
    for index, row in enumerate(rows):
        context = dict(
            decision_index=index + 1,
            case_id=row["id"],
            planned_sequence=index + 1,
            suite="ticket_router",
            episode_seed=0,
            env_instance=123,
            env_index=index,
            logical_key=[root.name, identity["model_identity_sha256"], "ticket_router", 0, row["id"]],
        )
        key = row["label"]
        probabilities = {queue: float(queue == key) for queue in QUEUES}
        question = dict(type="choice", criteria=QUEUES, instructions={"rules": RULES})
        request = dict(model="open-jev-9b", state=observation(index), questions={"pick": question})
        reply = dict(
            model="fixture",
            answers={"pick": dict(type="choice", choice=key, probabilities=probabilities, confidence=1.0)},
        )
        start_slot = add("slot_start", context=context, observation=observation(index), candidates=QUEUES)
        start_validation = add(
            "decision_validation_start",
            context=context,
            observation={"state": observation(index), "images": []},
            questions={"pick": dict(options=QUEUES, rules=[RULES], goal="", operation="")},
            kwargs={},
        )
        http = add(
            "http_start",
            context=context,
            method="POST",
            url="http://127.0.0.1:18188/v1/systemone",
            category="decision_post",
            post_attempt_index=1,
            request_body=body(request),
        )
        add(
            "http_terminal",
            context=context,
            start_event=http,
            status="response",
            post_attempt_index=1,
            http_status=200,
            response_body=body(reply),
        )
        add(
            "decision_validation_terminal",
            context=context,
            start_event=start_validation,
            status="returned",
            result=dict(
                answers={"pick": {"key": key, "probabilities": probabilities, "confidence": 1.0}},
                model="fixture",
                raw=reply,
            ),
        )
        tick = dict(step=index + 1, key=key, confidence=1.0, probabilities=probabilities, source="jev", model="fixture")
        ticks.append(tick)
        add(
            "slot_terminal",
            context=context,
            start_event=start_slot,
            status="returned",
            new_decisions=[tick],
            state_error=None,
        )
        act_context = dict(context, selected_key=key)
        act = add("act_start", context=act_context, inputs={"key": key}, association_matches=True, before=view(index))
        pair(
            "env_step",
            dict(
                context=act_context,
                key=key,
                before=dict(index=index, done=False, observation=observation(index), candidates=QUEUES),
            ),
            dict(
                context=act_context,
                after=dict(
                    index=index + 1,
                    done=index == 29,
                    observation=observation(index + 1),
                    candidates={} if index == 29 else QUEUES,
                ),
            ),
        )
        add(
            "act_terminal",
            context=act_context,
            start_event=act,
            status="returned",
            act_calls=[{"key": key, "accepted": True}],
            after=view(index + 1),
            raw_tool_result=dump(view(index + 1)),
        )
        routes.append(dict(id=row["id"], predicted=key, expected=key, correct=True))
    add(
        "lifecycle_terminal",
        start_event=runner_start,
        stage="Runner.run_agent",
        status="returned",
        result_type="answer",
    )
    for stage in ("DeepAgent.cleanup_task_resources", "ability_manager.teardown_tools", "Runner.release"):
        pair("lifecycle", {"stage": stage}, {"stage": stage})
    add("lifecycle_terminal", start_event=episode_start, stage="run_episode", status="returned")
    for stage in ("annotate", "DecisionModel.close"):
        pair("lifecycle", {"stage": stage}, {"stage": stage})
    job = arm / "s1a-home/evals/results/ticket_router/fixture__jev"
    trial = job / "ticket_router--0__fixture"
    (trial / "agent").mkdir(parents=True)
    report = dict(
        batch_sha256=binding["batch_sha256"],
        seed=0,
        total=30,
        processed=30,
        correct=30,
        coverage=1,
        ticket_ids=[row["id"] for row in rows],
        unprocessed_ids=[],
        routes=routes,
    )
    episode = dict(
        decisions=ticks,
        final_state=observation(30),
        views=[view(i) for i in range(31)],
        extra={"ticket_router": report, "rethink": False, "rethinks": [], "result_type": "answer"},
    )
    (trial / "agent/episode.json").write_text(dump(episode))
    (trial / "result.json").write_text(
        dump(
            dict(
                task_name="ticket_router/0",
                source="ticket_router",
                exception_info=None,
                agent_result={"metadata": {"steps": 30, "decisions": 30, "chat_calls": 0, "invalid_keys": 0}},
            )
        )
    )
    (job / "config.json").write_text(
        dump(
            dict(
                agents=[{"name": "s1a-evals/jev", "model_name": "jev"}],
                datasets=[{"name": "ticket_router", "n_tasks": 1}],
            )
        )
    )
    for name in ("result.json", "summary.json"):
        (job / name).write_text("{}")
    pair(
        "lifecycle",
        {"stage": "write_job"},
        {"stage": "write_job", "result": remote + "/s1a-home/evals/results/ticket_router/fixture__jev"},
    )
    pair("lifecycle", {"stage": "Runner.stop"}, {"stage": "Runner.stop"})
    add("process_terminal", state="returned", exit_code=0, stage="original_cli_entry")

    def save_events():
        (arm / "journal.jsonl").write_text("".join(dump(event) + "\n" for event in events))

    save_events()
    return root, arm, events, save_events, preparation


def test_complete_routes_are_provisional_and_missing_cells_remain(measurement):
    root, _, _, _, preparation = measurement
    result = aggregate(root, preparation)
    cell = result["cells"][0]
    assert cell["route_correct"] == cell["accepted_processed"] == 30
    assert cell["complete_episode"] and cell["strict_batch_success"]
    assert cell["quality_status"] == "PROVISIONAL"
    assert cell["process_identity_state"] == "verified"
    assert result["planned_routes"] == 90 and result["planned_episodes"] == 3
    assert [row["accepted_processed"] for row in result["cells"]] == [30, 0, 0]


@pytest.mark.parametrize(
    "mutation", ["expected", "probability", "request_label", "selected_key", "after_ticket", "chat", "timeout"]
)
def test_corrupt_or_abnormal_evidence_cannot_pass(measurement, mutation):
    root, arm, events, save_events, preparation = measurement
    episode_path = next((arm / "s1a-home").glob("evals/results/ticket_router/*/*/agent/episode.json"))
    episode = json.loads(episode_path.read_text())
    if mutation == "expected":
        episode["extra"]["ticket_router"]["routes"][0]["expected"] = "tampered"
    elif mutation == "probability":
        next(e for e in events if e["kind"] == "slot_terminal")["new_decisions"][0]["probabilities"] = {"human": 1}
    elif mutation == "request_label":
        event = next(e for e in events if e["kind"] == "http_start")
        request = json.loads(event["request_body"]["utf8"])
        request["state"]["ticket"]["label"] = "human"
        event["request_body"] = body(request)
    elif mutation == "selected_key":
        next(e for e in events if e["kind"] == "act_start")["inputs"]["key"] = "tampered"
    elif mutation == "after_ticket":
        next(e for e in events if e["kind"] == "env_step_terminal")["after"]["observation"]["ticket"]["id"] = "wrong"
    elif mutation == "chat":
        path = episode_path.parents[1] / "result.json"
        trial = json.loads(path.read_text())
        trial["agent_result"]["metadata"]["chat_calls"] = 1
        path.write_text(dump(trial))
    elif mutation == "timeout":
        episode["extra"]["result_type"] = "timeout"
        next(e for e in events if e.get("stage") == "Runner.run_agent" and e["kind"] == "lifecycle_terminal")[
            "result_type"
        ] = "timeout"
    episode_path.write_text(dump(episode))
    save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert not cell["strict_batch_success"] and not cell["complete_episode"]
    assert cell["planned_routes"] == 30 and cell["errors"]
    if mutation == "timeout":
        assert cell["route_correct"] == 30


def test_journal_gap_and_body_hash_fail_closed(measurement):
    root, _, events, save_events, preparation = measurement
    next(e for e in events if e["kind"] == "http_start")["request_body"]["sha256"] = "0" * 64
    events[-1]["event_index"] += 1
    save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert not cell["strict_batch_success"] and cell["errors"]


@pytest.mark.parametrize("value", ["off", True, 0, None])
def test_rethink_requires_original_false_boolean(measurement, value):
    root, arm, _, _, preparation = measurement
    path = next((arm / "s1a-home").glob("evals/results/ticket_router/*/*/agent/episode.json"))
    episode = json.loads(path.read_text())
    episode["extra"]["rethink"] = value
    path.write_text(dump(episode))
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["accepted_processed"] == cell["route_correct"] == 30
    assert not cell["complete_episode"] and not cell["strict_batch_success"]


@pytest.mark.parametrize("damage", ["final_newline", "stop_terminal", "process_terminal"])
def test_tail_damage_retains_closed_routes(measurement, damage):
    root, arm, events, save_events, preparation = measurement
    if damage == "final_newline":
        path = arm / "journal.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
    else:
        events[:] = [
            row
            for row in events
            if not (
                row["kind"] == "process_terminal"
                if damage == "process_terminal"
                else row["kind"] == "lifecycle_terminal" and row.get("stage") == "Runner.stop"
            )
        ]
        renumber(events)
        save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["accepted_processed"] == cell["route_correct"] == 30
    assert not cell["complete_episode"] and not cell["strict_batch_success"] and cell["errors"]


def test_broken_ticket_pair_only_loses_affected_ticket(measurement):
    root, _, events, save_events, preparation = measurement
    first = next(row for row in events if row["kind"] == "act_terminal")
    events.remove(first)
    renumber(events)
    save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert not cell["cases"][0]["accepted"]
    assert cell["accepted_processed"] == cell["route_correct"] == 29
    assert not cell["complete_episode"]


def test_corrupt_health_body_blocks_episode_without_losing_routes(measurement):
    root, _, events, save_events, preparation = measurement
    compact = events[-1]["identity"]
    events[-1:-1] = [
        dict(
            event_index=-1,
            kind="http_start",
            identity=compact,
            context=None,
            method="GET",
            url="http://127.0.0.1:18188/health",
            category="health",
            request_body=body({}),
        ),
        dict(
            event_index=-2,
            kind="http_terminal",
            identity=compact,
            context=None,
            start_event=-1,
            status="response",
            http_status=200,
            response_body=body({"status": "ready"}),
        ),
    ]
    renumber(events)
    save_events()
    assert aggregate(root, preparation)["cells"][0]["complete_episode"]
    events[-2]["response_body"]["sha256"] = "0" * 64
    save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["accepted_processed"] == cell["route_correct"] == 30
    assert not cell["complete_episode"] and not cell["strict_batch_success"]
    assert any("body length/hash" in error for error in cell["errors"])


@pytest.mark.parametrize("damage", ["ready_hash", "after_hash", "launch", "maps", "after_pid"])
def test_process_asset_binding_blocks_episode_preserving_routes(measurement, damage):
    root, arm, _, _, preparation = measurement
    assert aggregate(root, preparation)["cells"][0]["complete_episode"]
    path = arm / ("process-after.json" if damage in ("after_hash", "after_pid") else "process-ready.json")
    capture = json.loads(path.read_text())
    if damage.endswith("hash"):
        capture["worker"]["exe_sha256"] = "0" * 64
        path.write_text(dump(capture))
    elif damage == "after_pid":
        capture["worker"]["pid"] += 1
        path.write_text(dump(capture))
    elif damage == "launch":
        launch = json.loads((arm / "launch.json").read_text())
        launch["worker"] = "/remote/other-worker"
        (arm / "launch.json").write_text(dump(launch))
    else:
        (arm / "worker-ready.maps").write_text("changed")
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["accepted_processed"] == cell["route_correct"] == 30
    assert not cell["complete_episode"] and not cell["strict_batch_success"] and cell["errors"]
    assert cell["process_identity_state"] == "invalid"


def test_framework_copy_corruption_preserves_verified_routes(measurement):
    root, arm, _, _, preparation = measurement
    remote = f"/remote/{root.name}/open-9b"
    copy = arm / "framework.py"
    copy.write_text("# captured framework\n")
    sha = digest(copy.read_bytes())
    row = dict(
        module="openjiuwen.normal",
        path="/remote/framework.py",
        copy=remote + "/framework.py",
        bytes=copy.stat().st_size,
        sha256=sha,
        loaded_module_record_sha256=sha,
    )
    path = arm / "python-loaded-after.json"
    modules = json.loads(path.read_text())
    modules[row["module"]] = {"path": row["path"], "sha256": sha}
    path.write_text(dump(modules))
    (arm / "framework-result-type-source.json").write_text(dump(dict(candidates=[row])))
    assert aggregate(root, preparation)["cells"][0]["complete_episode"]
    copy.write_text("changed")
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["accepted_processed"] == cell["route_correct"] == 30
    assert not cell["complete_episode"] and not cell["strict_batch_success"]
    assert cell["framework_sources"]["state"] == "invalid"
    assert any("framework source copy hash" in error for error in cell["errors"])


@pytest.mark.parametrize("damage", ["missing", "empty", "capture_error", "partial_capture_error"])
def test_missing_or_incomplete_framework_capture_cannot_complete(measurement, damage):
    root, arm, _, _, preparation = measurement
    assert aggregate(root, preparation)["cells"][0]["complete_episode"]
    path = arm / "framework-result-type-source.json"
    capture = json.loads(path.read_text())
    failure = dict(module="openjiuwen.failed", path="/remote/failed.py", capture_error="read denied")
    if damage == "missing":
        path.unlink()
    else:
        capture["candidates"] = (
            [] if damage == "empty" else ([failure] if damage == "capture_error" else capture["candidates"] + [failure])
        )
        path.write_text(dump(capture))
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["route_correct"] == cell["accepted_processed"] == 30
    assert not cell["complete_episode"] and not cell["strict_batch_success"] and cell["errors"]
    assert cell["framework_sources"]["state"] == ("missing" if damage == "missing" else "incomplete")


@pytest.mark.parametrize(
    "damage", ["early_runner_return", "early_release", "early_episode_return", "late_reset", "late_init"]
)
def test_original_runner_and_cleanup_order_is_required(measurement, damage):
    root, _, events, save_events, preparation = measurement
    assert aggregate(root, preparation)["cells"][0]["complete_episode"]
    if damage == "early_runner_return":
        terminal = next(
            row for row in events if row["kind"] == "lifecycle_terminal" and row.get("stage") == "Runner.run_agent"
        )
        events.remove(terminal)
        events.insert(next(i for i, row in enumerate(events) if row.get("stage") == "Runner.run_agent") + 1, terminal)
    elif damage in ("late_reset", "late_init"):
        moved = [
            row
            for row in events
            if (
                row["kind"] in ("env_reset_start", "env_reset_terminal")
                if damage == "late_reset"
                else row.get("stage") == "Runner.start"
            )
        ]
        events[:] = [row for row in events if row not in moved]
        at = next(
            i
            for i, row in enumerate(events)
            if row["kind"] == "lifecycle_terminal" and row.get("stage") == "Runner.run_agent"
        )
        if damage == "late_init":
            at += 1
        events[at:at] = moved
    else:
        stage = "Runner.release" if damage == "early_release" else "run_episode"
        moved = [
            row
            for row in events
            if row.get("stage") == stage and (stage == "Runner.release" or row["kind"] == "lifecycle_terminal")
        ]
        events[:] = [row for row in events if row not in moved]
        at = next(
            i
            for i, row in enumerate(events)
            if row["kind"] == "lifecycle_terminal" and row.get("stage") == "Runner.run_agent"
        )
        events[at:at] = moved
    renumber(events)
    save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["route_correct"] == cell["accepted_processed"] == 30
    assert not cell["complete_episode"] and not cell["strict_batch_success"]
    assert any("lifecycle order" in error or "outside original Runner" in error for error in cell["errors"])


@pytest.mark.parametrize("provenance_source", ["response", "routing", "health"])
def test_served_laya_original_raw_provenance_transform(measurement, provenance_source):
    import asyncio
    import copy

    from benchmarks.quality_ticket import verify_decision
    from s1a.agents.ticket_router import QUEUES, RULES
    from s1a.decision_models.served import ServedLayaModel
    from s1a.decision_models.types import ChoiceQuestion, Observation
    from s1a.decision_models.validation import choice_faults, validate_answers

    _, _, events, _, preparation = measurement
    context = next(row["context"] for row in events if row["kind"] == "slot_start")
    related = copy.deepcopy([row for row in events if (row.get("context") or {}).get("case_id") == context["case_id"]])
    slot = next(row for row in related if row["kind"] == "slot_start")
    terminal = next(row for row in related if row["kind"] == "http_terminal")
    payload = json.loads(terminal["response_body"]["utf8"])
    payload.update(
        model="english",
        served_by={"checkpoint": "laya-english", "revision": "r1", "source": "wire"},
        diagnostic="original wire field",
    )
    health, read_at = {}, None
    if provenance_source != "response":
        payload["served_by"] = None
    if provenance_source == "health":
        payload["routing"] = {"model": "english", "repo": "laya-english", "reason": "explicit model english"}
        health = {"models": {"english": {"checkpoint": "laya-english", "revision": "r1", "device": "cuda"}}}
        read_at = "2026-10-10T09:00:00+00:00"

    class OfflineClient:
        url = "http://127.0.0.1:18188"

        def deadline(self):
            return 1.0

        async def decide(self, request, request_id, deadline):
            assert request == dict(
                model="english",
                state=slot["observation"],
                questions={"pick": dict(type="choice", criteria=QUEUES, instructions={"rules": RULES})},
            )
            return copy.deepcopy(payload), {}, 1.0

    model = ServedLayaModel(OfflineClient(), model="english", clock=lambda: 1.0)
    model._health_tried = 1.0
    model._health, model._health_read_at = health, read_at
    questions = {"pick": ChoiceQuestion(QUEUES, rules=RULES)}
    reply = asyncio.run(model._decide(Observation(slot["observation"]), questions))
    validated = validate_answers(reply, questions)
    captured = preparation[2].value(validated)
    assert captured["raw"]["served_by"]["source"] == provenance_source
    journal = related
    if health:
        start = dict(
            kind="http_start", event_index=slot["event_index"] - 2, method="GET", url="http://127.0.0.1:18188/health"
        )
        end = dict(
            kind="http_terminal",
            event_index=slot["event_index"] - 1,
            start_event=start["event_index"],
            status="response",
            http_status=200,
            response_body=body(health),
        )
        journal = [start, end, *related]
    for row in related:
        if row["kind"] == "http_start":
            request = json.loads(row["request_body"]["utf8"])
            request["model"] = "english"
            row["request_body"] = body(request)
        elif row["kind"] == "http_terminal":
            row["response_body"] = body(payload)
        elif row["kind"] == "decision_validation_terminal":
            row["result"] = captured
        elif row["kind"] == "slot_terminal":
            row["new_decisions"][0].update(model=validated.model, source="laya-served")
    identity = {"requested_model": "english"}
    assert verify_decision(related, slot["observation"], QUEUES, RULES, identity, choice_faults, journal=journal)
    raw = captured["raw"]
    for target, field, wrong in (
        (raw["served_by"], "source", "wrong"),
        (raw["served_by"], "revision", "wrong"),
        (raw, "diagnostic", "wrong"),
        (raw["answers"]["pick"], "confidence", 0.5),
    ):
        original = target[field]
        target[field] = wrong
        with pytest.raises(ValueError, match="Validated raw differs"):
            verify_decision(related, slot["observation"], QUEUES, RULES, identity, choice_faults, journal=journal)
        target[field] = original


def test_unknown_result_type_requests_actual_source(measurement):
    root, arm, events, save_events, preparation = measurement
    path = next((arm / "s1a-home").glob("evals/results/ticket_router/*/*/agent/episode.json"))
    episode = json.loads(path.read_text())
    episode["extra"]["result_type"] = "new-framework-type"
    path.write_text(dump(episode))
    next(e for e in events if e.get("stage") == "Runner.run_agent" and e["kind"] == "lifecycle_terminal")[
        "result_type"
    ] = "new-framework-type"
    save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["route_correct"] == 30 and not cell["strict_batch_success"]
    assert cell["required_input"] == "independent review of actual loaded framework result_type source"


def renumber(events):
    indices = {row["event_index"]: number for number, row in enumerate(events, 1)}
    for row in events:
        old = row["event_index"]
        if "start_event" in row:
            row["start_event"] = indices[row["start_event"]]
        row["event_index"] = indices[old]


def test_retry_is_one_ticket_and_each_request_is_checked(measurement):
    root, _, events, save_events, preparation = measurement
    success = next(row for row in events if row["kind"] == "http_start")
    terminal = next(row for row in events if row.get("start_event") == success["event_index"])
    import copy

    failed_start, failed_end = copy.deepcopy(success), copy.deepcopy(terminal)
    failed_start["event_index"], failed_end["event_index"] = 10000, 10001
    failed_end.update(start_event=10000, http_status=503, response_body=body({"error": "retry fixture"}))
    success["post_attempt_index"] = terminal["post_attempt_index"] = 2
    position = events.index(success)
    events[position:position] = [failed_start, failed_end]
    renumber(events)
    save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["accepted_processed"] == 30 and cell["cases"][0]["post_attempts"] == 2
    request = json.loads(failed_start["request_body"]["utf8"])
    request["state"]["ticket"]["id"] = "another-ticket"
    failed_start["request_body"] = body(request)
    save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert not cell["strict_batch_success"] and cell["errors"]


def test_29_routes_and_missing_job_preserve_denominator(measurement):
    root, arm, events, save_events, preparation = measurement
    last_id = next(row["context"]["case_id"] for row in reversed(events) if row.get("context"))
    events[:] = [row for row in events if (row.get("context") or {}).get("case_id") != last_id]
    renumber(events)
    save_events()
    path = next((arm / "s1a-home").glob("evals/results/ticket_router/*/*/agent/episode.json"))
    episode = json.loads(path.read_text())
    report = episode["extra"]["ticket_router"]
    report.update(processed=29, correct=29, coverage=29 / 30, unprocessed_ids=[last_id], routes=report["routes"][:29])
    episode.update(
        decisions=episode["decisions"][:29], views=episode["views"][:30], final_state=episode["views"][29]["state"]
    )
    path.write_text(dump(episode))
    trial_path = path.parents[1] / "result.json"
    trial = json.loads(trial_path.read_text())
    trial["agent_result"]["metadata"].update(steps=29, decisions=29)
    trial_path.write_text(dump(trial))
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["route_correct"] == cell["accepted_processed"] == 29 and cell["route_accuracy"] == 29 / 30
    assert not cell["complete_episode"] and not cell["strict_batch_success"]
    path.unlink()
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["route_correct"] == 29 and not cell["complete_episode"]


def test_all_wrong_complete_episode_is_not_strict_success(measurement):
    root, arm, events, save_events, preparation = measurement
    from s1a.agents.ticket_router import QUEUES

    slots = [row for row in events if row["kind"] == "slot_terminal"]
    choices = {
        row["context"]["decision_index"]: next(key for key in QUEUES if key != row["new_decisions"][0]["key"])
        for row in slots
    }
    for row in events:
        context = row.get("context")
        if not context:
            continue
        key = choices[context["decision_index"]]
        probabilities = {queue: float(queue == key) for queue in QUEUES}
        if "selected_key" in context:
            context["selected_key"] = key
        if row["kind"] == "http_terminal" and row["status"] == "response":
            reply = json.loads(row["response_body"]["utf8"])
            reply["answers"]["pick"].update(choice=key, probabilities=probabilities)
            row["response_body"] = body(reply)
        elif row["kind"] == "decision_validation_terminal":
            row["result"]["answers"]["pick"].update(key=key, probabilities=probabilities)
            row["result"]["raw"]["answers"]["pick"].update(choice=key, probabilities=probabilities)
        elif row["kind"] == "slot_terminal":
            row["new_decisions"][0].update(key=key, probabilities=probabilities)
        elif row["kind"] == "act_start":
            row["inputs"]["key"] = key
            row["before"]["score"] = 0.0
        elif row["kind"] == "env_step_start":
            row["key"] = key
        elif row["kind"] == "act_terminal":
            row["act_calls"][0]["key"] = key
            row["after"]["score"] = 0.0
            row["raw_tool_result"] = dump(row["after"])
    path = next((arm / "s1a-home").glob("evals/results/ticket_router/*/*/agent/episode.json"))
    episode = json.loads(path.read_text())
    episode["decisions"] = [row["new_decisions"][0] for row in slots]
    for route, key in zip(episode["extra"]["ticket_router"]["routes"], choices.values(), strict=True):
        route.update(predicted=key, correct=False)
    episode["extra"]["ticket_router"]["correct"] = 0
    for view in episode["views"]:
        view["score"] = 0.0
    path.write_text(dump(episode))
    save_events()
    cell = aggregate(root, preparation)["cells"][0]
    assert cell["accepted_processed"] == 30 and cell["route_correct"] == 0
    assert cell["complete_episode"] and not cell["strict_batch_success"]
