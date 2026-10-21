"""JEV-VL served behavior at the Agents/Omni boundary; Pillow images and mocked HTTP only."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import time
from io import BytesIO
from typing import Any

import httpx
import pytest
import pytest_asyncio
from PIL import Image as PillowImage
from PIL.PngImagePlugin import PngInfo
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

from s1a.decision_models import ChoiceQuestion, Image, NoulQuestion, Observation
from s1a.decision_models.jev_vl_served import ServedJevVLModel

MODEL = "autotrust/JEV-27B-VL"
PROTOCOL = "jev27-bare-v1"
URL = "http://jev.test"
PICK = ChoiceQuestion({"first": "First option", "second": "Second option"})
CHECK = NoulQuestion("Is the task complete?")


def image_bytes(
    format: str = "PNG", *, color: str = "white", size: tuple[int, int] = (8, 8), metadata: str | None = None
) -> bytes:
    buffer = BytesIO()
    kwargs: dict[str, Any] = {}
    if metadata is not None:
        info = PngInfo()
        info.add_text("comment", metadata)
        kwargs["pnginfo"] = info
    PillowImage.new("RGB", size, color).save(buffer, format=format, **kwargs)
    return buffer.getvalue()


PNG = image_bytes()
OBSERVATION = Observation({"task": "Pick a visible control"}, (Image(PNG),))


def corrupt_idat_crc() -> bytes:
    data = bytearray(PNG)
    offset = 8
    while offset < len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        if data[offset + 4 : offset + 8] == b"IDAT":
            data[offset + 8 + size] ^= 1
            return bytes(data)
        offset += size + 12
    raise AssertionError("fixture has no IDAT chunk")


def animated_png() -> bytes:
    buffer = BytesIO()
    PillowImage.new("RGB", (8, 8), "white").save(
        buffer, format="PNG", save_all=True, append_images=[PillowImage.new("RGB", (8, 8), "black")], duration=100
    )
    return buffer.getvalue()


def flat_answer(body: dict[str, Any], **changes: Any) -> dict[str, Any]:
    options = body.get("options", ["false", "true"])
    payload = {
        "model": MODEL,
        "protocol": PROTOCOL,
        "kind": body["kind"],
        "effective_kind": body["kind"],
        "options": options,
        "choice_index": 1,
        "choice": options[1],
        "probabilities": [0.2, 0.8],
        "adaptation": "native",
        "usage": {"prompt_tokens": 17, "completion_tokens": 1, "total_tokens": 18},
        "elapsed_seconds": 0.001,
        "num_model_requests": 1,
    }
    return payload | changes


class Server:
    def __init__(self, script: list[Any] | None = None) -> None:
        self.script = list(script or [])
        self.requests: list[httpx.Request] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.script:
            return httpx.Response(200, json=flat_answer(json.loads(request.content)))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        result = step(request) if callable(step) else step
        return await result if inspect.isawaitable(result) else result


@pytest_asyncio.fixture
async def make_model():
    models = []

    def make(*, script=None, timeout_s=5.0, url=URL):
        server = Server(script)
        model = ServedJevVLModel(url=url, timeout_s=timeout_s, transport=httpx.MockTransport(server.handle))
        models.append(model)
        return model, server

    yield make
    for model in models:
        await model.close()


def body(server: Server, index: int = 0) -> dict[str, Any]:
    return json.loads(server.requests[index].content)


def assert_status(caught: pytest.ExceptionInfo[BaseError], expected=StatusCode.MODEL_CALL_FAILED) -> None:
    assert caught.value.status == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [PICK, ChoiceQuestion({"only": "Only option"})])
async def test_idat_crc_corruption_is_rejected_before_forced_or_http_decision(make_model, question):
    model, server = make_model()
    observation = Observation({}, (Image(corrupt_idat_crc()),))
    with pytest.raises(BaseError) as caught:
        await model.decide_many(observation, {"pick": question})
    assert_status(caught, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
    assert server.requests == []


@pytest.mark.asyncio
async def test_extreme_integer_probability_is_a_model_error(make_model):
    def invalid(request):
        return httpx.Response(200, json=flat_answer(json.loads(request.content), probabilities=[10**1000, 0.8]))

    model, _ = make_model(script=[invalid])
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"pick": PICK})
    assert_status(caught)


@pytest.mark.asyncio
async def test_integer_usage_counts_do_not_overflow_float_validation(make_model):
    def reported(request):
        return httpx.Response(
            200,
            json=flat_answer(json.loads(request.content), usage={"prompt_tokens": 10**1000, "completion_tokens": 1}),
        )

    model, _ = make_model(script=[reported])
    decision = await model.decide_many(OBSERVATION, {"pick": PICK})
    assert decision.usage.known is True
    assert decision.usage.input_tokens == 10**1000
    assert decision.usage.output_tokens == 1


@pytest.mark.asyncio
async def test_choice_maps_index_to_key_even_when_descriptions_repeat(make_model):
    model, server = make_model(url=URL + "/gateway/")
    decision = await model.decide_many(OBSERVATION, {"pick": ChoiceQuestion({"a": "Same", "b": "Same"})})
    choice = decision.choice("pick")
    assert choice.key == "b"
    assert choice.probabilities == {"a": 0.2, "b": 0.8}
    assert choice.confidence == 0.8
    assert server.requests[0].method == "POST"
    assert str(server.requests[0].url) == URL + "/gateway/v1/systemone"
    assert body(server)["options"] == ["Same", "Same"]
    assert body(server)["model"] == decision.model == model.model == MODEL
    assert model.name == "jev-vl-served" and model.supports_images and model.deterministic
    assert model.bills_input_tokens is False


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["The current screen", {"image": "ordinary state field", "type": "text", "z": 2}])
async def test_state_and_current_image_bytes_are_preserved(make_model, state):
    model, server = make_model()
    await model.decide_many(Observation(state, (Image(PNG),)), {"pick": PICK})
    parts = body(server)["state"]
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    assert parts == [text, {"image": "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")}]


@pytest.mark.asyncio
async def test_option_dicts_use_stable_json_and_instructions_keep_every_field(make_model):
    model, server = make_model()
    question = ChoiceQuestion(
        {"z": {"z": 1, "a": "按钮"}, "a": {"a": "按钮", "z": 1}},
        goal="Complete checkout",
        operation="Click a control",
        rules=("Do not buy yet", "Keep the existing address"),
    )
    await model.decide_many(OBSERVATION, {"pick": question})
    sent = body(server)
    assert sent["options"] == [json.dumps({"a": "按钮", "z": 1}, ensure_ascii=False, sort_keys=True)] * 2
    assert json.loads(sent["question"]) == {
        "goal": question.goal,
        "operation": question.operation,
        "rules": list(question.rules),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("probabilities,index", [([0.2, 0.8], 1), ([0.8, 0.2], 0)])
async def test_noul_uses_false_true_and_returns_probability_of_true(make_model, probabilities, index):
    def respond(request):
        sent = json.loads(request.content)
        return httpx.Response(
            200,
            json=flat_answer(sent, probabilities=probabilities, choice_index=index, choice=["false", "true"][index]),
        )

    model, server = make_model(script=[respond])
    question = NoulQuestion("Has it finished?", {"false": "Still busy", "true": "Success is visible"})
    decision = await model.decide_many(OBSERVATION, {"check": question})
    sent = body(server)
    assert sent["kind"] == "noul"
    assert sent.get("options", ["false", "true"]) == ["false", "true"]
    assert json.loads(sent["question"]) == {"question": question.question, "criteria": question.criteria}
    assert decision.noul("check").p == probabilities[1]
    assert decision.noul("check").confidence == 0.8


@pytest.mark.asyncio
async def test_bare_noul_question_is_not_rewritten(make_model):
    model, server = make_model()
    await model.decide_many(OBSERVATION, {"check": CHECK})
    assert body(server)["question"] == CHECK.question


@pytest.mark.asyncio
async def test_forced_choice_is_a_known_zero_with_no_model_request(make_model):
    model, server = make_model()
    decision = await model.decide_many(OBSERVATION, {"forced": ChoiceQuestion({"only": "The sole action"})})
    assert decision.choice("forced").key == "only"
    assert decision.choice("forced").probabilities == {"only": 1.0}
    assert decision.usage.known and decision.usage.input_tokens == decision.usage.output_tokens == 0
    assert decision.raw["forced"] is True and decision.raw["num_model_requests"] == 0
    assert decision.raw["request_id"] is None
    assert decision.raw["responses"][0]["request_id"] is None
    assert not server.requests


@pytest.mark.asyncio
async def test_many_questions_are_serial_and_preserve_raw_provenance(make_model):
    events = []
    payloads = []

    async def respond(request):
        sent = json.loads(request.content)
        events.append(("start", sent["kind"]))
        await asyncio.sleep(0.015)
        payload = flat_answer(sent, usage={"prompt_tokens": 11, "completion_tokens": 2})
        payloads.append(payload)
        events.append(("end", sent["kind"]))
        return httpx.Response(200, json=payload, headers={"X-Request-ID": "server-id", "Server-Timing": "infer;dur=1"})

    model, server = make_model(script=[respond, respond])
    decision = await model.decide_many(
        OBSERVATION, {"pick": PICK, "forced": ChoiceQuestion({"only": "Only"}), "check": CHECK}
    )
    assert events == [("start", "choice"), ("end", "choice"), ("start", "noul"), ("end", "noul")]
    assert list(decision.answers) == ["pick", "forced", "check"]
    assert (decision.usage.input_tokens, decision.usage.output_tokens, decision.usage.known) == (22, 4, True)
    assert decision.latency_ms >= 25
    assert decision.raw["num_model_requests"] == len(server.requests) == 2
    assert decision.raw["forced"] is False and decision.raw["url"] == URL
    records = decision.raw["responses"]
    assert [record["question_id"] for record in records] == ["pick", "forced", "check"]
    assert [record["payload"] for record in records if not record["forced"]] == payloads
    learned = [record for record in records if not record["forced"]]
    assert all(record["headers"]["x-request-id"] == "server-id" for record in learned)
    assert all(record["request_id"] for record in learned)
    assert len({record["request_id"] for record in learned}) == 2
    assert [record["request_id"] for record in learned] == [
        request.headers["x-request-id"] for request in server.requests
    ]
    assert records[1]["forced"] is True and records[1]["request_id"] is None
    assert decision.raw["request_id"]
    assert decision.raw["server_timing"]["pick"] == {"infer": 1.0}
    assert decision.raw["served_by"]["checkpoint"] == MODEL
    assert decision.raw["served_by"]["device"] is None and decision.raw["served_by"]["revision"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,expected_input,expected_output,known",
    [
        (None, 0, 0, False),
        ({"completion_tokens": 2}, 0, 2, False),
        ({"prompt_tokens": True, "completion_tokens": 1}, 1, 1, False),
        ({"prompt_tokens": 3.5, "completion_tokens": -1}, 3, 0, False),
        ({"prompt_tokens": 0}, 0, 0, True),
    ],
)
async def test_unknown_usage_in_any_response_keeps_the_batch_unknown(
    make_model, usage, expected_input, expected_output, known
):
    def respond(request):
        return httpx.Response(200, json=flat_answer(json.loads(request.content), usage=usage))

    model, _ = make_model(script=[respond])
    decision = await model.decide_many(OBSERVATION, {"first": PICK, "second": CHECK})
    assert decision.usage.known is known
    assert decision.usage.input_tokens == expected_input + 17
    assert decision.usage.output_tokens == expected_output + 1


@pytest.mark.asyncio
async def test_all_fanout_requests_share_one_deadline(make_model):
    finished = []

    async def delayed(request):
        await asyncio.sleep(0.06)
        finished.append(request)
        return httpx.Response(200, json=flat_answer(json.loads(request.content)))

    model, server = make_model(script=[delayed, delayed], timeout_s=0.1)
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"first": PICK, "second": CHECK})
    assert_status(caught)
    assert len(server.requests) == 2 and len(finished) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_s", [0.01, 5.0])
async def test_image_preparation_counts_toward_latency_and_deadline(make_model, monkeypatch, timeout_s):
    original_open = PillowImage.open

    def slow_open(*args, **kwargs):
        time.sleep(0.025)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(PillowImage, "open", slow_open)
    model, server = make_model(timeout_s=timeout_s)
    if timeout_s < 0.025:
        with pytest.raises(BaseError) as caught:
            await model.decide_many(OBSERVATION, {"pick": PICK})
        assert_status(caught)
        assert not server.requests
    else:
        decision = await model.decide_many(OBSERVATION, {"pick": PICK})
        assert decision.latency_ms >= 20 and len(server.requests) == 1


@pytest.mark.asyncio
async def test_trickle_body_is_bounded_by_the_total_deadline(make_model):
    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in (b"{", b'"model":', b'"slow"', b"}"):
                await asyncio.sleep(0.025)
                yield chunk

    model, server = make_model(script=[httpx.Response(200, stream=Trickle())], timeout_s=0.06)
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"pick": PICK})
    assert_status(caught)
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_an_error_after_one_answer_returns_no_partial_decision_or_retry(make_model):
    model, server = make_model(
        script=[
            lambda request: httpx.Response(200, json=flat_answer(json.loads(request.content))),
            httpx.Response(500, json={"error": {"message": "worker failed"}}),
        ]
    )
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"first": PICK, "second": CHECK, "third": PICK}, attempts=4)
    assert_status(caught)
    assert len(server.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("format,media_type", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
async def test_a_new_screenshot_changes_the_body_without_resizing_or_caching(make_model, format, media_type):
    model, server = make_model()
    frames = [image_bytes(format, color=color) for color in ("white", "black")]
    for data in frames:
        await model.decide_many(Observation("Same state", (Image(data, media_type),)), {"pick": PICK})
    assert server.requests[0].content != server.requests[1].content
    for index, data in enumerate(frames):
        uri = body(server, index)["state"][1]["image"]
        assert uri.startswith(f"data:{media_type};base64,")
        assert base64.b64decode(uri.split(",", 1)[1]) == data


@pytest.mark.asyncio
@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize(
    "images",
    [
        (),
        (Image(PNG), Image(PNG)),
        (Image(b""),),
        (Image(PNG, "image/gif"),),
        (Image(PNG, "image/jpeg"),),
        (Image(image_bytes("GIF")),),
        (Image(animated_png()),),
        (Image(b"not an image"),),
        (Image(PNG[:25]),),
        (Image(image_bytes(size=(2049, 1))),),
        (Image(image_bytes(size=(1100, 1000))),),
        (Image(image_bytes(size=(201, 1))),),
    ],
)
async def test_invalid_current_images_fail_before_http_even_for_forced_choices(make_model, forced, images):
    model, server = make_model()
    question = ChoiceQuestion({"only": "Only"}) if forced else PICK
    with pytest.raises(BaseError) as caught:
        await model.decide_many(Observation("Current state", images), {"pick": question})
    assert_status(caught, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
    assert not server.requests


@pytest.mark.asyncio
async def test_single_image_above_four_mib_is_rejected_before_http(make_model):
    data = image_bytes(metadata="x" * (4 * 1024 * 1024))
    assert len(data) > 4 * 1024 * 1024
    model, server = make_model()
    with pytest.raises(BaseError) as caught:
        await model.decide_many(Observation("Current state", (Image(data),)), {"pick": PICK})
    assert_status(caught, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
    assert not server.requests


@pytest.mark.asyncio
async def test_maximum_side_length_is_accepted_when_pixel_count_is_small(make_model):
    model, server = make_model()
    await model.decide_many(Observation("Screen", (Image(image_bytes(size=(2048, 16))),)), {"pick": PICK})
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_body_between_four_and_eight_mib_is_accepted(make_model):
    model, server = make_model()
    await model.decide_many(Observation("x" * (4 * 1024 * 1024 + 1024), (Image(PNG),)), {"pick": PICK})
    assert 4 * 1024 * 1024 < len(server.requests[0].content) < 8 * 1024 * 1024


@pytest.mark.asyncio
async def test_utf8_body_above_eight_mib_is_rejected_before_http(make_model):
    model, server = make_model()
    with pytest.raises(BaseError) as caught:
        await model.decide_many(Observation("图" * (3 * 1024 * 1024), (Image(PNG),)), {"pick": PICK})
    assert_status(caught, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
    assert not server.requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"model": "other/model"},
        {"protocol": "other-protocol"},
        {"kind": "noul"},
        {"effective_kind": "noul"},
        {"options": ["Second option", "First option"]},
        {"options": ["First option"]},
        {"choice_index": True},
        {"choice_index": -1},
        {"choice_index": 2},
        {"choice_index": 1.0},
        {"choice_index": "1"},
        {"choice": "First option"},
        {"probabilities": {"first": 0.2, "second": 0.8}},
        {"probabilities": [1.0]},
        {"probabilities": [False, True]},
        {"probabilities": [-0.1, 1.1]},
        {"probabilities": [0.2, 0.7998]},
        {"probabilities": [0.8, 0.2]},
    ],
)
async def test_bad_flat_responses_fail_without_retry_or_probability_repair(make_model, changes):
    def respond(request):
        return httpx.Response(200, json=flat_answer(json.loads(request.content), **changes))

    model, server = make_model(script=[respond])
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"pick": PICK}, attempts=4)
    assert_status(caught)
    assert len(server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["model", "protocol", "kind", "effective_kind", "options", "choice_index", "choice", "probabilities"]
)
async def test_every_required_flat_response_field_is_checked(make_model, field):
    def respond(request):
        payload = flat_answer(json.loads(request.content))
        del payload[field]
        return httpx.Response(200, json=payload)

    model, _ = make_model(script=[respond])
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"pick": PICK})
    assert_status(caught)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_nonfinite_probability_is_rejected(make_model, value):
    def respond(request):
        payload = flat_answer(json.loads(request.content), probabilities=[0.2, value])
        return httpx.Response(200, content=json.dumps(payload), headers={"content-type": "application/json"})

    model, _ = make_model(script=[respond])
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"pick": PICK})
    assert_status(caught)


@pytest.mark.asyncio
@pytest.mark.parametrize("probabilities", [[0.2, 0.80005], [0.5, 0.5]])
async def test_valid_small_sum_error_and_tied_peak_keep_original_probabilities(make_model, probabilities):
    def respond(request):
        return httpx.Response(200, json=flat_answer(json.loads(request.content), probabilities=probabilities))

    model, _ = make_model(script=[respond])
    decision = await model.decide_many(OBSERVATION, {"pick": PICK})
    assert decision.choice("pick").probabilities == dict(zip(PICK.ids, probabilities))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404, 413, 422, 429, 500, 502, 504])
async def test_http_failures_preserve_status_and_reason_and_do_not_retry(make_model, status):
    model, server = make_model(script=[httpx.Response(status, json={"error": {"message": "fixture rejection"}})])
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"pick": PICK}, attempts=4)
    assert_status(caught)
    assert str(status) in str(caught.value) and "fixture rejection" in str(caught.value)
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_unauthorized_is_a_configuration_error(make_model):
    model, server = make_model(script=[httpx.Response(401, json={"error": {"message": "unauthorized"}})])
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"pick": PICK})
    assert_status(caught, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
    assert len(server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not JSON", "[]", "null", "42"])
async def test_bad_json_or_nonobject_response_is_a_model_call_failure(make_model, content):
    model, server = make_model(script=[httpx.Response(200, text=content)])
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"pick": PICK})
    assert_status(caught)
    assert len(server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ConnectError("connection refused"), httpx.ReadTimeout("read deadline")])
async def test_transport_errors_are_model_call_failures_without_retry(make_model, error):
    model, server = make_model(script=[error])
    with pytest.raises(BaseError) as caught:
        await model.decide_many(OBSERVATION, {"pick": PICK}, attempts=4)
    assert_status(caught)
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_close_releases_the_pooled_transport_and_is_idempotent():
    class Transport(httpx.MockTransport):
        closed = False

        async def aclose(self):
            self.closed = True
            await super().aclose()

    transport = Transport(lambda _: httpx.Response(200))
    model = ServedJevVLModel(url=URL, transport=transport)
    await model.close()
    await model.close()
    assert transport.closed


@pytest.mark.asyncio
async def test_from_env_requires_its_explicit_url_and_needs_no_cloud_key(monkeypatch):
    for key in ("JEV_VL_URL", "S1A_DECISION_TIMEOUT_S", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(BaseError) as caught:
        ServedJevVLModel.from_env()
    assert_status(caught, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
    assert "JEV_VL_URL" in str(caught.value)
    monkeypatch.setenv("JEV_VL_URL", URL)
    model = ServedJevVLModel.from_env()
    assert model.model == MODEL
    await model.close()


@pytest.mark.parametrize("value", ["soon", "nan", "inf", "-inf", "0", "-1"])
def test_from_env_uses_the_existing_decision_timeout_validation(monkeypatch, value):
    monkeypatch.setenv("JEV_VL_URL", URL)
    monkeypatch.setenv("S1A_DECISION_TIMEOUT_S", value)
    with pytest.raises(BaseError) as caught:
        ServedJevVLModel.from_env()
    assert_status(caught, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
    assert "S1A_DECISION_TIMEOUT_S" in str(caught.value)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0.0, -1.0])
def test_constructor_rejects_unbounded_or_nonpositive_deadline(timeout):
    with pytest.raises(BaseError) as caught:
        ServedJevVLModel(url=URL, timeout_s=timeout)
    assert_status(caught, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
