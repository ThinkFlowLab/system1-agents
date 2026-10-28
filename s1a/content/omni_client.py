# coding: utf-8
"""HTTP client for draft System1-Omni workers that speak ``POST /v1/systemone``.

The default worker is StartLux Instant 4B from InferMatrixLab/system1-omni draft pull request 164.
Fact-check may switch to the PPLX draft worker from pull request 166, on its own base URL.
Clef, pull request 165, is refused on this path. None of those pull requests need to be merged:
the client only needs a base URL that already serves the contract.

Each question is its own short prefill, Public231-style: one ``state`` string and one question.
StartLux is sent ``think_mode: never``. PPLX is sent ``thinking: {enabled: false}``.
Those are different draft contracts; one body is not sent to both.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from s1a.content.rules import FACT_SCORE, SCREEN_SCORE, fact_questions, fact_state, screen_questions, screen_state

CLEF_PR = 165
DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class WorkerProfile:
    key: str
    model: str
    pr: int
    hot_path: bool
    thinking: str
    role: str

    @property
    def pr_url(self) -> str:
        return f"https://github.com/InferMatrixLab/system1-omni/pull/{self.pr}"


WORKERS = {
    "startlux": WorkerProfile(
        key="startlux",
        model="StartLux-Decision-v1.1-4B-Instant",
        pr=164,
        hot_path=True,
        thinking="think_mode",
        role="default screen and fact-check",
    ),
    "pplx": WorkerProfile(
        key="pplx",
        model="perplexity-ai/pplx-decider-v1.1-27b",
        pr=166,
        hot_path=False,
        thinking="thinking_object",
        role="optional heavier fact-check",
    ),
}


def require_worker(name: str, *, fact_check: bool) -> WorkerProfile:
    if name == "clef":
        raise ValueError(
            "Clef is not on the content hot path "
            f"(InferMatrixLab/system1-omni pull request {CLEF_PR}). "
            "Use startlux for screen and fact-check, or pplx for fact-check only."
        )
    profile = WORKERS.get(name)
    if profile is None:
        raise ValueError(f"unknown worker {name!r}; choose startlux or pplx")
    if not fact_check and name != "startlux":
        raise ValueError("the screen step uses the StartLux Instant 4B draft worker; pplx is fact-check only")
    return profile


def check_base_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("an Omni base URL must be http(s) without credentials, a query, or a path")
    return url.strip().rstrip("/")


def timeout_from_env(env: Mapping[str, str]) -> float:
    raw = (env.get("OMNI_TIMEOUT_S") or str(DEFAULT_TIMEOUT_S)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("OMNI_TIMEOUT_S must be a finite number above 0") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("OMNI_TIMEOUT_S must be a finite number above 0")
    return value


def worker_record(profile: WorkerProfile, base_url: str, model: str) -> dict[str, Any]:
    return {
        "worker": profile.key,
        "model": model,
        "pr": profile.pr,
        "pr_url": profile.pr_url,
        "draft": True,
        "merged_required": False,
        "base_url": base_url,
        "hot_path": profile.hot_path,
        "prefill": "short",
        "questions": ["choice", "score"],
        "thinking": profile.thinking,
    }


class OmniClient:
    """One draft worker. Synchronous, one POST per question, no retry.

    Retries would blur the latency a later C1 gate reads. A failed status raises and does not
    include the response body, which can echo the submitted state.
    """

    def __init__(
        self,
        base_url: str,
        *,
        profile: WorkerProfile,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite number above 0")
        self.base_url = check_base_url(base_url)
        self.profile = profile
        self.model = model or profile.model
        self._timeout_s = timeout_s
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            timeout=timeout_s,
            headers=headers,
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        try:
            response = self._client.get(f"{self.base_url}/health")
        except httpx.HTTPError as exc:
            raise RuntimeError(f"no Omni worker at {self.base_url}: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise RuntimeError(f"Omni health at {self.base_url} returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Omni health at {self.base_url} was not JSON") from exc
        if not isinstance(body, dict):
            raise RuntimeError(f"Omni health at {self.base_url} was not a JSON object")
        return body

    def ask(self, state: str, question_id: str, question: dict[str, Any]) -> tuple[dict[str, Any], int, str]:
        if not state.strip():
            raise ValueError("an Omni prefill state must not be empty")
        if len(question) == 0:
            raise ValueError("an Omni question must not be empty")
        body = self._body(state, question_id, question)
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.base_url}/v1/systemone",
                json=body,
                headers={"X-Request-Id": request_id},
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Omni request to {self.base_url} failed: {type(exc).__name__}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        if response.status_code != 200:
            raise RuntimeError(f"Omni {self.base_url} returned HTTP {response.status_code} for question {question_id}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Omni {self.base_url} returned a non-JSON body for {question_id}") from exc
        answers = payload.get("answers") if isinstance(payload, dict) else None
        answer = answers.get(question_id) if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise RuntimeError(f"Omni {self.base_url} returned no answer for {question_id}")
        if isinstance(payload, dict) and isinstance(payload.get("model"), str):
            answer = dict(answer)
            answer["_served_model"] = payload["model"]
        return answer, latency_ms, request_id

    def _body(self, state: str, question_id: str, question: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {question_id: question},
        }
        if self.profile.thinking == "think_mode":
            body["think_mode"] = "never"
        elif self.profile.thinking == "thinking_object":
            body["thinking"] = {"enabled": False}
        else:
            raise ValueError(f"worker {self.profile.key} has no thinking contract")
        return body


def _finite_number(value: Any, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"Omni {what} was not a finite number")
    return float(value)


def parse_choice(answer: dict[str, Any], criteria: dict[str, str]) -> tuple[str, dict[str, float]]:
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in criteria:
        raise RuntimeError("Omni choice was outside the offered keys")
    raw = answer.get("probabilities")
    if not isinstance(raw, dict):
        raise RuntimeError("Omni choice was missing probabilities")
    probabilities = {key: _finite_number(raw.get(key), what=f"probability {key}") for key in criteria}
    return choice, probabilities


def parse_score(answer: dict[str, Any], rubric: list[str]) -> tuple[float, int, dict[str, float], str]:
    score = _finite_number(answer.get("score"), what="score")
    if score < 0:
        raise RuntimeError("Omni score was negative")
    raw = answer.get("probabilities")
    probabilities: dict[str, float] = {}
    if raw is not None:
        if not isinstance(raw, dict):
            raise RuntimeError("Omni score probabilities were not an object")
        for index in range(len(rubric)):
            key = str(index)
            if key in raw:
                probabilities[key] = _finite_number(raw[key], what=f"score probability {key}")
    if probabilities:
        level = max(range(len(rubric)), key=lambda index: probabilities.get(str(index), float("-inf")))
    else:
        level = min(len(rubric) - 1, max(0, int(round(score))))
    return score, level, probabilities, rubric[level]


class OmniDecider:
    """Screen on one client, fact-check on the same client or a heavier draft worker."""

    name = "omni"

    def __init__(self, screen: OmniClient, fact: OmniClient) -> None:
        self.screen_client = screen
        self.fact_client = fact

    def close(self) -> None:
        self.screen_client.close()
        if self.fact_client is not self.screen_client:
            self.fact_client.close()

    def screen(self, item: dict[str, Any]) -> dict[str, Any]:
        return _pair(self.screen_client, screen_state(item), screen_questions(), "screen", "relevance", SCREEN_SCORE)

    def fact_check(self, item: dict[str, Any]) -> dict[str, Any]:
        return _pair(self.fact_client, fact_state(item), fact_questions(), "verdict", "support", FACT_SCORE)


def _pair(
    client: OmniClient,
    state: str,
    questions: dict[str, dict[str, Any]],
    choice_id: str,
    score_id: str,
    rubric: list[str],
) -> dict[str, Any]:
    """Ask the choice and the score as two short prefills. Each is judged against the label later."""
    choice_answer, choice_ms, choice_request = client.ask(state, choice_id, questions[choice_id])
    score_answer, score_ms, score_request = client.ask(state, score_id, questions[score_id])
    criteria = questions[choice_id]["criteria"]
    choice, probabilities = parse_choice(choice_answer, criteria)
    score, level, score_probabilities, score_name = parse_score(score_answer, rubric)
    served = choice_answer.get("_served_model")
    model = served if isinstance(served, str) else client.model
    return {
        "choice": choice,
        "probabilities": probabilities,
        "score": score,
        "score_level": level,
        "score_name": score_name,
        "score_probabilities": score_probabilities,
        "latency_ms": choice_ms + score_ms,
        "backend": "omni",
        "model": model,
        "requested_model": client.model,
        "worker": client.profile.key,
        "calls": [
            {"question": choice_id, "latency_ms": choice_ms, "request_id": choice_request},
            {"question": score_id, "latency_ms": score_ms, "request_id": score_request},
        ],
    }


def decider_from_env(
    env: Mapping[str, str],
    *,
    fact_check_worker: str = "startlux",
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> OmniDecider:
    """Build the screen client from ``OMNI_BASE_URL`` and, for PPLX, a second URL.

    ``handler`` wraps a fresh mock transport per client so tests never open a socket.
    """
    screen_profile = require_worker("startlux", fact_check=False)
    fact_profile = require_worker(fact_check_worker, fact_check=True)
    base_url = check_base_url(env.get("OMNI_BASE_URL") or "")
    timeout_s = timeout_from_env(env)
    api_key = (env.get("OMNI_API_KEY") or "").strip() or None
    screen_model = (env.get("OMNI_MODEL") or "").strip() or screen_profile.model
    fact_url = (env.get("OMNI_FACTCHECK_BASE_URL") or "").strip()
    fact_model = (env.get("OMNI_FACTCHECK_MODEL") or "").strip() or fact_profile.model
    if fact_profile.key == "pplx" and not fact_url:
        raise ValueError(
            "PPLX fact-check needs OMNI_FACTCHECK_BASE_URL. "
            "Draft pull request 166 is a different worker from StartLux and is not assumed to share its port."
        )
    if fact_profile.key == "startlux" and not fact_url:
        fact_url = base_url
        fact_model = (env.get("OMNI_FACTCHECK_MODEL") or "").strip() or screen_model
    else:
        fact_url = check_base_url(fact_url)

    def transport() -> httpx.BaseTransport | None:
        if handler is None:
            return None
        return httpx.MockTransport(handler)

    screen = OmniClient(
        base_url,
        profile=screen_profile,
        model=screen_model,
        api_key=api_key,
        timeout_s=timeout_s,
        transport=transport(),
    )
    if fact_url == screen.base_url and fact_profile.key == screen_profile.key and fact_model == screen.model:
        return OmniDecider(screen, screen)
    fact_key = (env.get("OMNI_FACTCHECK_API_KEY") or env.get("OMNI_API_KEY") or "").strip() or None
    fact = OmniClient(
        fact_url,
        profile=fact_profile,
        model=fact_model,
        api_key=fact_key,
        timeout_s=timeout_s,
        transport=transport(),
    )
    return OmniDecider(screen, fact)
