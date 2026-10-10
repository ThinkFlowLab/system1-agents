"""Independently verify frozen ticket-router raw evidence offline, with all 3 x 30 cases."""

import argparse
import asyncio
import base64
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlsplit

from benchmarks.client import decode
from benchmarks.evidence import digest, dump, outside_repo, write_json

BINDING_SHA256 = "8bd28a3624842cdaf7fcd35dac187dc0378f029d559daa2f870ed3ca4d7a573a"
SCHEMA_SHA256 = "34b2014a5f62db9801a52eab7831de6ffa4ef6af0cd3ce3bde8cb5b75b7499af"
CELLS = {
    "open-9b": ("jev", "open-jev-9b"),
    "open-27b": ("jev", "open-jev-27b-v1.1"),
    "laya-english": ("laya-served", "english"),
}
NORMAL_BASIS = "answer_only_provisional_pending_actual_framework_source_verification"


def read_json(path):
    return decode(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_preparation(path):
    require(digest((path / "binding.json").read_bytes()) == BINDING_SHA256, "Frozen producer binding hash mismatch")
    require(digest((path / "schema-r1.md").read_bytes()) == SCHEMA_SHA256, "Frozen producer schema hash mismatch")
    binding = read_json(path / "binding.json")
    source = path / "evidence.py"
    require(
        digest(source.read_bytes()) == binding["source_sha256"]["evidence.py"], "Frozen journal parser hash mismatch"
    )
    spec = importlib.util.spec_from_file_location("ticket_raw_journal", source)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    # Imports below use the current checkout, whose relevant contracts must match this producer.
    current = Path(__file__).resolve().parents[1]
    for name in (
        "s1a/agents/ticket_router.py",
        "s1a/decision_models/validation.py",
        "s1a/decision_models/types.py",
        "s1a/decision_models/jev.py",
        "s1a/decision_models/laya.py",
        "s1a/decision_models/served.py",
        "s1a/tool/models.py",
        "s1a/tool/loop.py",
        "s1a/jobs.py",
    ):
        require(
            digest((current / name).read_bytes()) == binding["source_sha256"]["runtime/" + name],
            f"Current contract differs from frozen producer: {name}",
        )
    fixture = path / "runtime/s1a/agents/_data/ticket_router_eval.jsonl"
    require(digest(fixture.read_bytes()) == binding["fixture_sha256"], "Frozen ticket fixture hash mismatch")
    require(
        digest((path / "planned-cases.jsonl").read_bytes()) == binding["planned_cases_sha256"],
        "Frozen ticket plan hash mismatch",
    )
    return path, binding, producer


def body_bytes(record):
    raw = base64.b64decode(record["base64"], validate=True)
    require(len(raw) == record["bytes"] and digest(raw) == record["sha256"], "HTTP body length/hash mismatch")
    require(raw.decode("utf-8", errors="replace") == record["utf8"], "HTTP body viewing text mismatch")
    return raw


def bytes_json(record):
    return decode(body_bytes(record))


def relocated(path, remote_arm, arm):
    suffix = Path(path).relative_to(Path(remote_arm))
    local = (arm / suffix).resolve()
    require(local.is_relative_to(arm.resolve()), "Original artifact path escapes current cell")
    return local


def verify_identity(root, arm, item, binding):
    identity = read_json(arm / "identity.json")
    require(
        identity["benchmark_run_id"] == root.name and identity["planned_model"] == arm.name,
        "Current run/cell identity mismatch",
    )
    require(
        identity["requested_model"] == CELLS[arm.name][1]
        and identity["identity_state"] == "current_exact_assets_recorded",
        "Actual model identity unavailable",
    )
    assets_hash = digest(json.dumps(identity["actual_assets"], sort_keys=True, ensure_ascii=False).encode("utf-8"))
    require(assets_hash == identity["model_identity_sha256"], "Actual asset identity hash mismatch")
    require(identity["source_binding_sha256"] == BINDING_SHA256, "Cell source binding mismatch")
    require(identity["job_id"] == read_json(root / "status.json")["job_id"], "Job identity mismatch")
    python = read_json(arm / "python-identity.json")
    loaded = read_json(arm / "python-loaded-after.json")
    require(
        python["actual_s1a_root"] == str(Path(python["expected_s1a_root"]) / "s1a"),
        "Actual loaded Agents root differs from requested source",
    )
    for modules in (python["modules"], loaded):
        for name, row in modules.items():
            if name == "s1a" or name.startswith("s1a."):
                suffix = Path(row["path"]).relative_to(python["expected_s1a_root"])
                require(
                    row["sha256"] == binding["source_sha256"].get("runtime/" + suffix.as_posix()),
                    f"Loaded source hash mismatch: {name}",
                )
        for name in ("s1a.agents.ticket_router", "s1a.decision_models.validation", "s1a.tool.models", "s1a.tool.loop"):
            require(name in modules, f"Missing actual loaded source: {name}")
    env = python["environment"]
    require(
        env["MODEL_NAME"] == "" and env["S1A_HOME"] == str(Path(item["arm"]) / "s1a-home"),
        "Chat model must be empty and S1A_HOME must belong to this cell",
    )
    require(
        env["S1A_DECISION_TIMEOUT_S"] == "120" and env["LAYA_SERVED_TIMEOUT_S"] == "120",
        "Actual decision deadline differs from 120s",
    )
    model_field = "LAYA_SERVED_MODEL" if arm.name == "laya-english" else "TYPESAFE_MODEL"
    require(env[model_field] == identity["requested_model"], "Actual request model environment mismatch")
    return identity, python


def verify_processes(arm, identity):
    launch = read_json(arm / "launch.json")
    assets = identity["actual_assets"]
    binaries = assets["binaries"] if arm.name == "laya-english" else assets["binary_sha256"]
    ready = read_json(arm / "process-ready.json")
    after = read_json(arm / "process-after.json")
    for phase, capture in (("ready", ready), ("after", after)):
        for role in ("worker", "frontend"):
            process = capture[role]
            require(
                process["exe"] == launch[role]
                and process["argv"] == [launch[role]]
                and process["exe_sha256"] == binaries[launch[role]],
                f"Actual {phase} {role} executable differs from launch/assets",
            )
            require(
                type(process["pid"]) is int and process["pid"] > 0 and process["pid"] == ready[role]["pid"],
                f"Actual {role} process changed during episode",
            )
            maps = (arm / f"{role}-{phase}.maps").read_bytes()
            require(digest(maps) == process["maps_sha256"], f"Actual {phase} {role} maps hash mismatch")
            mapped = {
                line.split(maxsplit=5)[5] for line in maps.decode().splitlines() if len(line.split(maxsplit=5)) == 6
            }
            require(
                process["exe"] in mapped and set(process["loaded_libraries"]).issubset(mapped),
                f"Actual {phase} {role} executable/libraries absent from captured maps",
            )


def verify_decision(events, observation, queues, rules, identity, choice_faults, *, journal=None):
    slots = [row for row in events if row["kind"] == "slot_start"]
    require(len(slots) == 1, "Decision must have exactly one original slot start")
    slot = slots[0]
    context = slot["context"]
    require(
        all(
            {key: value for key, value in row["context"].items() if key != "selected_key"} == context for row in events
        ),
        "Decision context/environment instance differs across original calls",
    )
    require(
        slot["observation"] == observation and slot["candidates"] == queues, "Slot current ticket/candidates mismatch"
    )
    terminals = {row.get("start_event"): row for row in events if row["kind"].endswith("_terminal")}
    posts = [row for row in events if row["kind"] == "http_start" and row["method"] == "POST"]
    require(
        all(type(row["post_attempt_index"]) is int for row in posts)
        and [row["post_attempt_index"] for row in posts] == list(range(1, len(posts) + 1)),
        "POST retry sequence mismatch",
    )
    expected_request = dict(
        model=identity["requested_model"],
        state=observation,
        questions={"pick": {"type": "choice", "criteria": queues, "instructions": {"rules": rules}}},
    )
    replies = []
    for post in posts:
        url = urlsplit(post["url"])
        require(
            post["category"] == "decision_post"
            and url.hostname in ("127.0.0.1", "localhost")
            and url.port == 18188
            and url.path == "/v1/systemone",
            "Unexpected decision POST destination",
        )
        require(
            dump(bytes_json(post["request_body"])) == dump(expected_request),
            "Actual POST differs from current public ticket/queues/rules/model",
        )
        terminal = terminals[post["event_index"]]
        if terminal["status"] == "response":
            # HTTP errors are still decoded/hash checked; non-JSON error bodies remain diagnostic.
            try:
                payload = bytes_json(terminal["response_body"])
            except (UnicodeError, json.JSONDecodeError):
                require(not 200 <= terminal["http_status"] < 300, "Successful response is not JSON")
                continue
            if 200 <= terminal["http_status"] < 300:
                replies.append((post, terminal, payload))
    slot_result = terminals[slot["event_index"]]
    if slot_result["status"] != "returned" or not slot_result.get("new_decisions"):
        return None
    require(
        slot_result.get("state_error") is None and len(slot_result["new_decisions"]) == 1,
        "Returned slot is blocked or has multiple decisions",
    )
    tick = slot_result["new_decisions"][0]
    validations = [row for row in events if row["kind"] == "decision_validation_start"]
    require(len(validations) == 1, "Missing or duplicate original validation start")
    validation = validations[0]
    require(validation["observation"] == {"state": observation, "images": []}, "Validator observation mismatch")
    require(
        validation["questions"] == {"pick": {"options": queues, "rules": [rules], "goal": "", "operation": ""}},
        "Validator offered question mismatch",
    )
    result = terminals[validation["event_index"]]
    require(result["status"] == "returned" and replies, "No successful POST for original validated decision")
    validated = result["result"]
    answer = validated["answers"]["pick"]
    faults = choice_faults(
        {"choice": answer["key"], "probabilities": answer["probabilities"], "confidence": answer["confidence"]},
        list(queues),
    )
    require(not faults, "Invalid original validated probabilities/confidence: " + "; ".join(faults))
    payload = replies[-1][2]
    wire = payload["answers"]["pick"]
    require(
        wire.get("type") == "choice" and not choice_faults(wire, list(queues)),
        "Invalid wire choice probabilities/confidence",
    )
    require(
        (wire["choice"], wire["probabilities"], wire["confidence"])
        == (answer["key"], answer["probabilities"], answer["confidence"]),
        "Wire and original validation result differ",
    )
    expected_raw = payload
    if identity["requested_model"] == "english":
        from s1a.decision_models.served import served_by_from_health

        sent = payload.get("served_by")
        if isinstance(sent, dict):
            served_by = {**sent, "source": "response"}
        else:
            history = events if journal is None else journal
            starts = {row["event_index"]: row for row in history if row["kind"] == "http_start"}
            health = {}
            for row in history:
                if row["event_index"] >= replies[-1][0]["event_index"]:
                    break
                start = starts.get(row.get("start_event"), {})
                url = urlsplit(start.get("url", ""))
                if (
                    row["kind"] == "http_terminal"
                    and row.get("status") == "response"
                    and row.get("http_status") == 200
                    and start.get("method") == "GET"
                    and url.hostname in ("127.0.0.1", "localhost")
                    and url.port == 18188
                    and url.path == "/health"
                ):
                    try:
                        reading = bytes_json(row["response_body"])
                    except (UnicodeError, json.JSONDecodeError):
                        continue
                    if isinstance(reading, dict) and reading:
                        health = reading
            provenance = validated["raw"]["served_by"]
            read_at = provenance.get("read_at")
            require(read_at is None or isinstance(read_at, str), "Invalid original health reading timestamp")
            served_by = served_by_from_health(health, payload.get("routing"), read_at)
        expected_raw = {**payload, "served_by": served_by}
    require(
        all(validated["raw"].get(key) == value for key, value in expected_raw.items()),
        "Validated raw differs from actual HTTP reply",
    )
    require(
        tick["key"] == answer["key"]
        and tick["probabilities"] == answer["probabilities"]
        and tick["confidence"] == round(answer["confidence"], 3)
        and tick["model"] == validated["model"],
        "Slot tick differs from original validation result",
    )
    require(
        slot["event_index"] < validation["event_index"] < posts[0]["event_index"]
        and replies[-1][1]["event_index"] < result["event_index"] < slot_result["event_index"],
        "Decision/POST/validation/slot event order mismatch",
    )
    return tick, context, slot_result


async def score_cell(root, arm, item, binding, producer, fixture, TicketRouterEnv, queues, rules, choice_faults):
    identity, python = verify_identity(root, arm, item, binding)
    events, integrity = producer.read_journal(arm / "journal.jsonl", identity)
    errors = [f"journal: {row['error']}" for row in integrity]
    # Pair/identity faults affect their events; a broken cleanup cannot erase closed ticket evidence.
    damaged_events = {row[key] for row in integrity for key in ("event_index", "start_event") if key in row}
    malformed_lines = [row["line"] for row in integrity if row["code"] == "invalid_json_event"]
    if events and any(row["code"] == "missing_final_newline" for row in integrity):
        damaged_events.add(events[-1]["event_index"])
    terminals = {row.get("start_event"): row for row in events if row["kind"].endswith("_terminal")}
    env = TicketRouterEnv(0, tickets=fixture, batch_size=30)
    await env.reset()
    planned_ids = env.report()["ticket_ids"]
    labels = {row["id"]: row["label"] for row in fixture}
    observations = []
    for ticket_id in planned_ids:
        observations.append(await env.observe())
        await env.step(labels[ticket_id])
    observations.append(await env.observe())
    planned = [
        dict(
            suite="ticket_router",
            episode_seed=0,
            sequence=index + 1,
            ticket_id=ticket_id,
            public_ticket=observations[index]["ticket"],
            public_ticket_sha256=digest(dump(observations[index]["ticket"]).encode("utf-8")),
            expected_next_ticket_id=planned_ids[index + 1] if index < 29 else None,
            planned_denominator=30,
        )
        for index, ticket_id in enumerate(planned_ids)
    ]
    initial = read_json(arm / "planned-initial.json")
    actual_plan = read_json(arm / "planned-cases.json")
    require(
        initial["planned_denominator"] == actual_plan["planned_denominator"] == 30 and initial["cases"] == planned,
        "Prebuilt initial plan differs from fixture-derived seed0 order/public fields",
    )
    require(
        initial["identity"]["benchmark_run_id"] == root.name
        and initial["identity"]["planned_model"] == arm.name
        and initial["identity"]["identity_state"] == "planned_contract_only",
        "Prebuilt plan identity differs from this run/cell",
    )
    require(
        actual_plan["identity"]
        == {key: identity[key] for key in ("benchmark_run_id", "model_identity_sha256")} | {"record": "identity.json"},
        "Current actual plan identity mismatch",
    )
    require(
        actual_plan["cases"]
        == [
            {
                **case,
                "logical_key": [root.name, identity["model_identity_sha256"], "ticket_router", 0, case["ticket_id"]],
            }
            for case in planned
        ],
        "Current actual plan/logical keys mismatch",
    )
    cases = [
        dict(
            ticket_id=ticket_id,
            planned_sequence=index + 1,
            expected=labels[ticket_id],
            attempted=False,
            post_attempts=0,
            accepted=False,
            correct=False,
            predicted=None,
            errors=[],
        )
        for index, ticket_id in enumerate(planned_ids)
    ]
    decisions = {}
    for row in events:
        context = row.get("context")
        if context is None:
            if row["kind"] in ("http_start", "http_terminal"):
                try:
                    for field in ("request_body", "response_body"):
                        if field in row:
                            body_bytes(row[field])
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"Non-decision HTTP evidence: {error}")
            if row["kind"] == "http_start" and row["method"] == "POST":
                errors.append("POST outside original decision context; zero-chat provenance insufficient")
            continue
        ticket_id = context.get("case_id")
        if ticket_id not in planned_ids:
            errors.append("Unknown current ticket in journal context")
            continue
        index = planned_ids.index(ticket_id)
        expected_key = [root.name, identity["model_identity_sha256"], "ticket_router", 0, ticket_id]
        if (
            context.get("logical_key") != expected_key
            or any(
                type(context.get(key)) is not int
                for key in ("decision_index", "planned_sequence", "env_index", "episode_seed")
            )
            or context.get("decision_index", 0) <= 0
            or context.get("planned_sequence") != index + 1
            or context.get("env_index") != index
            or context.get("suite") != "ticket_router"
            or context.get("episode_seed") != 0
        ):
            cases[index]["errors"].append("Current case/seed/logical identity mismatch")
        decisions.setdefault(context.get("decision_index"), []).append(row)
    verified_ticks, verified_acts = [], []
    for related in decisions.values():
        context = related[0]["context"]
        ticket_id = context.get("case_id")
        index = planned_ids.index(ticket_id)
        case = cases[index]
        case["attempted"] = True
        case["post_attempts"] += sum(row["kind"] == "http_start" and row["method"] == "POST" for row in related)
        try:
            require(
                not case["errors"]
                and not any(row["event_index"] in damaged_events for row in related)
                and not any(line <= max(row["event_index"] for row in related) for line in malformed_lines),
                "Journal/current case integrity unavailable",
            )
            verified = verify_decision(
                related, observations[index], queues, rules, identity, choice_faults, journal=events
            )
            acts = [row for row in related if row["kind"] == "act_start"]
            steps = [row for row in related if row["kind"] == "env_step_start"]
            if verified is None:
                require(not acts and not steps, "No valid slot decision for executed act/step")
                continue
            tick, slot_context, slot_result = verified
            verified_ticks.append(tick)
            if not acts:
                require(not steps, "Environment step without original act")
                continue
            require(len(acts) == len(steps) == 1 and not case["accepted"], "Duplicate act/step for planned ticket")
            act, step = acts[0], steps[0]
            act_result, step_result = terminals[act["event_index"]], terminals[step["event_index"]]
            key = tick["key"]
            act_context = {**slot_context, "selected_key": key}
            require(
                act["context"] == step["context"] == act_context
                and act["inputs"] == {"key": key}
                and act["association_matches"] is True
                and step["key"] == key,
                "Selected key/accepted act association mismatch",
            )
            require(
                act_result["status"] == step_result["status"] == "returned"
                and act_result["act_calls"] == [{"key": key, "accepted": True}],
                "Original act/step was not accepted",
            )
            require(
                slot_result["event_index"]
                < act["event_index"]
                < step["event_index"]
                < step_result["event_index"]
                < act_result["event_index"],
                "Original act/step event order mismatch",
            )
            require(
                step["before"] == dict(index=index, done=False, observation=observations[index], candidates=queues)
                and step_result["after"]
                == dict(
                    index=index + 1,
                    done=index == 29,
                    observation=observations[index + 1],
                    candidates={} if index == 29 else queues,
                ),
                "Original environment step before/after differs from seeded fixture",
            )
            for view, expected_index in ((act["before"], index), (act_result["after"], index + 1)):
                require(
                    view["state"] == observations[expected_index]
                    and view["done"] == (expected_index == 30)
                    and view["candidates"] == ({} if expected_index == 30 else queues),
                    "Original act view differs from seeded fixture",
                )
            require(decode(act_result["raw_tool_result"]) == act_result["after"], "Original act output/view mismatch")
            case.update(
                accepted=True,
                correct=key == labels[ticket_id],
                predicted=key,
                accepted_event=act_result["event_index"],
                step_event=step_result["event_index"],
            )
            verified_acts.append((case, act, act_result))
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
            case["errors"].append(str(error))
    for case in cases:
        errors.extend(f"{case['ticket_id']}: {error}" for error in case["errors"])
    process_identity_state = "verified"
    try:
        verify_processes(arm, identity)
    except (AttributeError, OSError, KeyError, TypeError, ValueError) as error:
        process_identity_state = "invalid"
        errors.append(str(error))
    # Partial/timeout episodes retain independently verified accepted prefixes.
    complete, result_type, job_source = False, None, None
    try:
        writes = [
            row
            for row in events
            if row["kind"] == "lifecycle_terminal"
            and row.get("stage") == "write_job"
            and row.get("status") == "returned"
        ]
        require(len(writes) == 1, "Missing unique actual write_job result")
        job = relocated(writes[0]["result"], item["arm"], arm)
        require(
            job.is_relative_to((arm / "s1a-home/evals/results/ticket_router").resolve()),
            "Original job outside current cell",
        )
        paths = list(job.glob("*/agent/episode.json"))
        require(len(paths) == 1, "Missing unique original episode in actual returned job")
        path = paths[0]
        episode, trial, config = (
            read_json(path),
            read_json(path.parents[1] / "result.json"),
            read_json(job / "config.json"),
        )
        for name in ("result.json", "summary.json"):
            read_json(job / name)  # Existence/JSON only: score and DONE are not an oracle.
        job_source = dict(
            job=str(job),
            original_path=writes[0]["result"],
            episode_sha256=digest(path.read_bytes()),
            trial_sha256=digest((path.parents[1] / "result.json").read_bytes()),
        )
        require(
            config["agents"] == [{"name": "s1a-evals/" + CELLS[arm.name][0], "model_name": CELLS[arm.name][0]}]
            and config["datasets"] == [{"name": "ticket_router", "n_tasks": 1}],
            "Original job policy/plan mismatch",
        )
        require(
            trial["task_name"] == "ticket_router/0" and trial["source"] == "ticket_router",
            "Original trial seed/task mismatch",
        )
        extra, metadata = episode["extra"], trial["agent_result"]["metadata"]
        result_type = extra.get("result_type")
        require(
            extra["rethink"] is False
            and not extra.get("rethinks")
            and type(metadata["chat_calls"]) is int
            and metadata["chat_calls"] == 0,
            "Original episode must have rethink off and zero chat",
        )
        require(episode["decisions"] == verified_ticks, "Original job decisions differ from verified slot ticks")
        accepted = [case for case in cases if case["accepted"]]
        report = extra["ticket_router"]
        expected_routes = [
            {
                "id": case["ticket_id"],
                "expected": case["expected"],
                "predicted": case["predicted"],
                "correct": case["correct"],
            }
            for case in accepted
        ]
        require(
            report["seed"] == 0
            and report["batch_sha256"] == binding["batch_sha256"]
            and report["total"] == 30
            and report["processed"] == len(accepted)
            and report["ticket_ids"] == planned_ids
            and report["unprocessed_ids"] == planned_ids[len(accepted) :]
            and report["routes"] == expected_routes
            and report["correct"] == sum(case["correct"] for case in accepted),
            "Original report differs from fixture-derived routes/labels/full denominator",
        )
        require(
            [case["planned_sequence"] for case in accepted] == list(range(1, len(accepted) + 1)),
            "Accepted routes are not a prefix",
        )
        views = episode["views"]
        require(
            len(views) == len(accepted) + 1 and episode["final_state"] == observations[len(accepted)],
            "Original job views/final state mismatch",
        )
        for index, view in enumerate(views):
            require(
                view["state"] == observations[index]
                and view["candidates"] == ({} if index == 30 else queues)
                and view["done"] == (index == 30)
                and view["score"] == sum(case["correct"] for case in accepted[:index]),
                "Original job view differs from independently replayed prefix",
            )
        for index, (_, act, act_result) in enumerate(verified_acts):
            require(
                act["before"] == views[index] and act_result["after"] == views[index + 1],
                "Original job/act views mismatch",
            )
        require(
            metadata["steps"] == len(accepted)
            and metadata["decisions"] == len(verified_ticks)
            and metadata["invalid_keys"] == 0,
            "Original job action counts mismatch",
        )
        lifecycle = [row for row in events if row["kind"] == "lifecycle_terminal"]
        required = {
            stage: [row for row in lifecycle if row.get("stage") == stage] for stage in producer.REQUIRED_LIFECYCLES
        }
        require(
            all(len(rows) == 1 and rows[0]["status"] == "returned" for rows in required.values()),
            "Original lifecycle incomplete or abnormal",
        )
        require(
            required["Runner.run_agent"][0].get("result_type") == result_type
            and required["Runner.stop"][0]["event_index"] > writes[0]["event_index"],
            "Original runner result/stop mismatch",
        )
        runner = required["Runner.run_agent"][0]
        require(
            all(
                runner["start_event"] < row["event_index"] < runner["event_index"]
                for row in events
                if row.get("context") is not None
            ),
            "Decision/act/environment evidence outside original Runner.run_agent",
        )
        resets = [row for row in events if row["kind"] == "env_reset_terminal"]
        require(len(resets) == 1, "Original episode must have exactly one environment reset")
        cli = [row for row in events if row["kind"] == "cli_start"]
        require(len(cli) == 1, "Missing unique original CLI start")
        ordered = [cli[0]["event_index"]]
        for stage in ("Runner.start", "build_model", "make_series", "DecisionModel.warm", "TicketRouterEnv.__init__"):
            row = required[stage][0]
            ordered.extend((row["start_event"], row["event_index"]))
        ordered.extend((required["run_episode"][0]["start_event"], resets[0]["start_event"], resets[0]["event_index"]))
        for stage in (
            "DeepAgent.ensure_initialized",
            "Runner.run_agent",
            "DeepAgent.cleanup_task_resources",
            "ability_manager.teardown_tools",
            "Runner.release",
        ):
            row = required[stage][0]
            ordered.extend((row["start_event"], row["event_index"]))
        ordered.append(required["run_episode"][0]["event_index"])
        for stage in ("annotate", "DecisionModel.close", "write_job", "Runner.stop"):
            row = required[stage][0]
            ordered.extend((row["start_event"], row["event_index"]))
        require(all(left < right for left, right in zip(ordered, ordered[1:])), "Original lifecycle order mismatch")
        processes = [row for row in events if row["kind"] == "process_terminal"]
        expected_argv = [
            "s1a",
            "run",
            "ticket_router",
            "--model",
            CELLS[arm.name][0],
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
            str(Path(python["expected_s1a_root"]) / "s1a/agents/_data/ticket_router_eval.jsonl"),
            "--batch-size",
            "30",
            "--log",
        ]
        require(cli[0]["argv"] == expected_argv, "Actual CLI plan/budget differs from frozen contract")
        require(
            len(processes) == 1
            and events[-1] == processes[0]
            and processes[0].get("state") == "returned"
            and processes[0].get("exit_code") == 0
            and processes[0].get("stage") == "original_cli_entry"
            and not processes[0].get("error")
            and not processes[0].get("received_signal")
            and item.get("collector_exit_code") == 0,
            "Original process/collector did not complete normally",
        )
        require(
            not any(
                row["kind"] == "process_signal" or (row["kind"] != "http_terminal" and row.get("status") == "exception")
                for row in events
            ),
            "Abnormal non-HTTP termination",
        )
        require(
            trial.get("exception_info") is None and result_type == "answer",
            "Non-normal or unknown original result_type",
        )
        require(len(accepted) == 30 and not errors, "Incomplete accepted coverage or evidence errors")
        complete = True
    except (AttributeError, OSError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    try:
        framework = framework_sources(arm, item["arm"])
        if framework["state"] != "verified":
            errors.append("Actual framework source capture missing or incomplete")
            complete = False
    except (AttributeError, OSError, KeyError, TypeError, ValueError) as error:
        framework = {"state": "invalid", "record": str(arm / "framework-result-type-source.json")}
        errors.append(str(error))
        complete = False
    correct = sum(case["correct"] for case in cases if case["accepted"])
    processed = sum(case["accepted"] for case in cases)
    return dict(
        cell=arm.name,
        planned_routes=30,
        planned_episodes=1,
        cases=cases,
        route_correct=correct,
        route_accuracy=correct / 30,
        route_evidence_basis="recorded cell identity and individually verified HTTP/slot/accepted act/environment step",
        accepted_processed=processed,
        accepted_coverage=processed / 30,
        complete_episode=complete,
        strict_batch_success=complete and correct == 30,
        result_type=result_type,
        normal_result_type_basis=NORMAL_BASIS,
        quality_status="PROVISIONAL" if complete else "INCOMPLETE",
        required_input="independent review of actual loaded framework result_type source",
        framework_sources=framework,
        identity=identity,
        identity_basis="exact recorded assets; source-to-build ancestry unverified",
        process_identity_state=process_identity_state,
        original_job=job_source,
        errors=errors,
    )


def framework_sources(arm, remote_arm):
    path = arm / "framework-result-type-source.json"
    if not path.exists():
        return {"state": "missing", "candidates": []}
    record = read_json(path)
    loaded = read_json(arm / "python-loaded-after.json")
    for row in record["candidates"]:
        if "capture_error" in row:
            continue
        raw = relocated(row["copy"], remote_arm, arm).read_bytes()
        require(
            len(raw) == row["bytes"] and digest(raw) == row["sha256"] == row["loaded_module_record_sha256"],
            "Actual framework source copy hash mismatch",
        )
        require(
            loaded.get(row["module"]) == {"path": row["path"], "sha256": row["sha256"]},
            "Framework source capture differs from actually loaded module",
        )
    record["state"] = (
        "verified"
        if record["candidates"] and not any("capture_error" in row for row in record["candidates"])
        else "incomplete"
    )
    return record  # Exact files are supplied for independent enum review, not interpreted as normal here.


def aggregate(root, preparation):
    try:
        from s1a.agents.ticket_router import QUEUES, RULES, TicketRouterEnv, load_tickets
        from s1a.decision_models.validation import choice_faults
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Ticket scoring needs the existing System1-Agents environment (including openjiuwen); "
            "public231 scoring is independent of that dependency"
        ) from error
    prep, binding, producer = preparation
    current = Path(__file__).resolve().parents[1]
    require(
        Path(TicketRouterEnv.__init__.__code__.co_filename).resolve() == current / "s1a/agents/ticket_router.py"
        and Path(choice_faults.__code__.co_filename).resolve() == current / "s1a/decision_models/validation.py",
        "Offline scorer imported a different Agents contract root",
    )
    fixture = load_tickets(prep / "runtime/s1a/agents/_data/ticket_router_eval.jsonl")
    require(len(fixture) == 30 and len({row["id"] for row in fixture}) == 30, "Fixture must contain 30 unique tickets")
    frozen_plan = [decode(line) for line in (prep / "planned-cases.jsonl").read_text(encoding="utf-8").splitlines()]
    cells, root_errors, items = [], [], {}
    try:
        status = read_json(root / "status.json")
        require(
            status["planned_models"] == list(CELLS) and status["planned_tickets_per_model"] == 30,
            "Measurement full planned cells/denominator mismatch",
        )
        require(
            len(status["models"]) == 3 and {item["name"] for item in status["models"]} == set(CELLS),
            "Measurement duplicate or unknown cell",
        )
        require(
            (root / "binding.json").read_bytes() == (prep / "binding.json").read_bytes(),
            "Measurement binding differs from frozen producer",
        )
        source = read_json(root / "source.json")
        require(
            source
            == {
                "binding_sha256": BINDING_SHA256,
                "files": binding["source_sha256"],
                "runtime_revision": binding["agents_revision"],
            },
            "Measurement source identity mismatch",
        )
        items = {item["name"]: item for item in status["models"]}
    except (AttributeError, OSError, KeyError, TypeError, ValueError) as error:
        root_errors.append(str(error))
    for name in CELLS:
        try:
            require(not root_errors and name in items, "Missing or invalid measurement root/cell identity")
            cell = asyncio.run(
                score_cell(
                    root,
                    root / name,
                    items[name],
                    binding,
                    producer,
                    fixture,
                    TicketRouterEnv,
                    QUEUES,
                    RULES,
                    choice_faults,
                )
            )
        except (AttributeError, OSError, KeyError, TypeError, ValueError) as error:
            cases = [
                dict(
                    ticket_id=case["ticket_id"],
                    planned_sequence=case["sequence"],
                    attempted=False,
                    accepted=False,
                    correct=False,
                    predicted=None,
                    errors=[str(error)],
                )
                for case in frozen_plan
            ]
            cell = dict(
                cell=name,
                planned_routes=30,
                planned_episodes=1,
                cases=cases,
                route_correct=0,
                route_accuracy=0.0,
                accepted_processed=0,
                accepted_coverage=0.0,
                complete_episode=False,
                strict_batch_success=False,
                quality_status="MISSING_OR_INVALID",
                errors=[str(error)],
                identity=None,
                process_identity_state="unavailable",
                original_job=None,
                normal_result_type_basis=NORMAL_BASIS,
            )
        cells.append(cell)
    return dict(
        schema="ticket-raw-offline-quality-v1",
        run=str(root.resolve()),
        planned_routes=90,
        planned_episodes=3,
        cells=cells,
        root_errors=root_errors,
        latency_reported=False,
        route_correct=sum(cell["route_correct"] for cell in cells),
        accepted_processed=sum(cell["accepted_processed"] for cell in cells),
        complete_episodes=sum(cell["complete_episode"] for cell in cells),
        strict_batch_successes=sum(cell["strict_batch_success"] for cell in cells),
        producer_binding_sha256=BINDING_SHA256,
        normal_result_type_basis=NORMAL_BASIS,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="whole measurement root, including missing cells")
    parser.add_argument("--preparation", type=Path, required=True, help="frozen ticket producer r3 directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.run, load_preparation(args.preparation))
    output = outside_repo(args.out)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result)
    print(dump(result))


if __name__ == "__main__":
    main()
