"""The browser policy consumes served JEV-VL answers; the page and HTTP are test doubles."""

from __future__ import annotations

import base64
import json
from copy import deepcopy

import httpx
import pytest

from test_browser_policy import _MESSAGES, _TOOLS, _slot_model
from test_decision_models_clm import Server as ClmServer
from test_decision_models_jev_vl_served import MODEL, PROTOCOL, image_bytes
from test_decision_models_served import Server as LayaServer
from s1a.decision_models.jev_vl_served import ServedJevVLModel
from s1a.decision_models.served import ServedLayaClient, ServedLayaModel


class Page:
    def __init__(self, *, capture_fails=False):
        self.completed = []
        self.capture_fails = capture_fails
        self.pictures = [image_bytes(color="white"), image_bytes(color="black")]

    async def probe_for_policy(self, source, params):
        generation = f"g{len(self.completed)}"
        return {
            "url": "http://fixture.test/",
            "title": "Two controls",
            "text": f"Completed controls: {self.completed}",
            "page_key": generation,
            "generation_id": generation,
            "elements": [
                {"node": i, "target_id": f"t_{generation}_{i}", "role": "button", "label": label, "value": ""}
                for i, label in ((1, "First"), (2, "Second"))
            ],
        }

    async def code_executor(self, code):
        if self.capture_fails:
            raise RuntimeError("capture failed")
        encoded = base64.b64encode(self.pictures[len(self.completed)]).decode()
        return {"payload": {"result": json.dumps({"png": encoded})}}

    def apply(self, reply):
        call = reply.tool_calls[0]
        assert call.name == "browser_click"
        target = json.loads(call.arguments)["target_id"]
        expected = f"t_g{len(self.completed)}_{len(self.completed) + 1}"
        assert target == expected
        self.completed.append(target)


def served(page, *, fail_target=False):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        instructions = json.loads(body["question"])
        target = instructions.get("operation") == "CLICK"
        if target and fail_target:
            return httpx.Response(413, json={"error": {"message": "image rejected"}})
        index = len(page.completed) if target else 0
        probabilities = [0.1 / (len(body["options"]) - 1)] * len(body["options"])
        probabilities[index] = 0.9
        return httpx.Response(
            200,
            headers={
                "Server-Timing": "infer;dur=12",
                "X-Request-Id": "worker-id",
                "Set-Cookie": "session=private-cookie",
                "Authorization": "Bearer private-token",
                "X-Proxy-Secret": "private-proxy-value",
            },
            json={
                "model": MODEL,
                "protocol": PROTOCOL,
                "kind": body["kind"],
                "effective_kind": body["kind"],
                "options": body["options"],
                "choice_index": index,
                "choice": body["options"][index],
                "probabilities": probabilities,
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                "num_model_requests": 1,
            },
        )

    return ServedJevVLModel(url="http://jev.test", transport=httpx.MockTransport(handler)), requests


@pytest.mark.asyncio
async def test_two_captures_drive_distinct_controls_and_preserve_serving_trace():
    page = Page()
    model, requests = served(page)
    slot = _slot_model([], runtime=page, decision_model=model, goal_value_cache=False, prefetch_values=False)
    try:
        for _ in range(2):
            page.apply(await slot.invoke(_MESSAGES, tools=_TOOLS))
    finally:
        await model.close()
    assert page.completed == ["t_g0_1", "t_g1_2"]
    assert len(requests) == 4
    assert requests[0]["state"][1] != requests[2]["state"][1]
    assert "Completed controls: []" in requests[0]["state"][0]
    assert "t_g0_1" in requests[2]["state"][0]
    for tick in slot.ticks:
        assert tick["url"] == "http://jev.test"
        assert tick["served_by"] == {"checkpoint": MODEL, "revision": None, "device": None, "source": "response"}
        raw = tick["decision_raw"]
        assert raw["num_model_requests"] == 2 and raw["forced"] is False
        assert [r["question_id"] for r in raw["responses"]] == ["operation", "click_target"]
        assert raw["responses"][0]["payload"]["protocol"] == PROTOCOL
        assert tick["server_timing"]["operation"] == {"infer": 12.0}
        for response in raw["responses"]:
            assert set(response["headers"]) == {"server-timing", "x-request-id", "content-type", "content-length"}
        assert "private-" not in json.dumps(tick)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["laya-served", "clm"])
async def test_existing_served_backends_keep_provenance_without_raw_payload(backend):
    page = Page()
    if backend == "laya-served":
        server = LayaServer()
        model = ServedLayaModel(ServedLayaClient(url="http://laya.test", transport=httpx.MockTransport(server.handler)))
    else:
        model = ClmServer().model()
    slot = _slot_model([], runtime=page, decision_model=model, goal_value_cache=False, prefetch_values=False)
    try:
        page.apply(await slot.invoke(_MESSAGES, tools=_TOOLS))
    finally:
        await model.close()
    assert page.completed == ["t_g0_1"]
    assert len(slot.ticks) == 1
    tick = slot.ticks[0]
    assert tick["url"] == ("http://laya.test" if backend == "laya-served" else "http://clm.test")
    assert tick["request_id"] and tick["served_by"]
    assert "server_timing" in tick
    assert "decision_raw" not in tick


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_fails,fail_target", [(True, False), (False, True)])
async def test_capture_or_later_head_failure_returns_no_action(capture_fails, fail_target):
    page = Page(capture_fails=capture_fails)
    model, requests = served(page, fail_target=fail_target)
    slot = _slot_model([], runtime=page, decision_model=model, goal_value_cache=False, prefetch_values=False)
    before = deepcopy(page.completed)
    try:
        reply = await slot.invoke(_MESSAGES, tools=_TOOLS)
    finally:
        await model.close()
    assert reply.tool_calls is None
    assert json.loads(reply.content)["status"] == "BLOCKED"
    assert page.completed == before
    assert len(requests) == (0 if capture_fails else 2)
    assert slot.ticks == []
