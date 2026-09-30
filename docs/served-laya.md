# Served Laya: a decision model over HTTP

## 1. Requirements

Functional (from #20):
- An agent can use a Laya model served by system1-omni instead of loading it in process, from the CLI
  and from MCP, with no cloud key; the in-process `--model laya` stays.
- `choice` and `noul`, one or several questions per request, through the existing answer validation.
- Runs record which model, checkpoint and serving backend answered.

Non-functional:
- Latency: a warm decision on MPS takes 25–160 ms server-side; the client adds little on localhost.
- Failure: a slow or absent server fails one decision within a bounded deadline, with a clear error.
- Contract: one written interface both repositories test against (OpenAPI 3.1, [api/laya-systemone.openapi.yaml](api/laya-systemone.openapi.yaml)).

Constraints: today's server is laya-serve 0.3.20 behind the system1-omni worker and optionally the Rust
frontend. The design sets the target interface; section 4 says where each part that is not there yet
gets built, and how the client works with both in the meantime.

## 2. High level

```
agent step ──► ServedLayaModel (s1a) ──HTTP──► [omni-jev frontend :8080] ──► Laya worker :8000 ──► laya (MPS/CPU)
                 │  laya_question()              forwards unchanged           warmup before listen
                 │  answer validation            502/504 if worker down       /health: device, revision, compile
                 └─ run record: /health snapshot + client round trip
```

## 3. Interface (target; full spec in [api/laya-systemone.openapi.yaml](api/laya-systemone.openapi.yaml))

| | |
|---|---|
| `POST /v1/systemone` | `{model?, state, questions}` → `{model, served_by, answers, usage, routing}` |
| `GET /livez`, `/readyz` | process up; ready to serve (503 + `Retry-After` until warm). `/health` stays as an alias |
| Errors | RFC 9457 `application/problem+json` with a stable `code`; 422 lists every invalid question |
| Tracing | `X-Request-Id` (or W3C `traceparent`) in, echoed out, used as problem `instance` |
| Timing | `Server-Timing: queue, infer` from the worker, `proxy` added by the frontend |
| Overload | bounded queue; 503 + `Retry-After` when full |
| Auth | optional bearer token; 401 with `WWW-Authenticate: Bearer` |
| Limits | 1–64 questions, state ≤ 50,000 chars, body ≤ 2 MiB; advertised in `/readyz` |
| Evolution | additive within `/v1`; unknown request fields ignored; breaking change → `/v2`, `Deprecation`/`Sunset` |
| Semantics | deterministic and side-effect free: any request may be retried, no idempotency key |

## 4. From today to the target

Today's behaviour is specified in [api/laya-systemone.current.openapi.yaml](api/laya-systemone.current.openapi.yaml), checked against
traffic captured from a running worker. Validating that traffic against the target spec: every error response, the missing
`served_by`, and the absent `X-Request-Id` / `Server-Timing` headers are the gap; `/health` already
matches. Most of it sits in the worker, which wraps laya-serve's app, so laya itself needs no change.

| item | today | built in |
|---|---|---|
| problem+json errors with `code` | `{"detail"}` (worker), text/plain (frontend 502/504) | worker: exception handler over laya-serve's app; frontend: its two error bodies |
| 422 lists every invalid question | first invalid question only | worker: validate all questions before calling laya |
| `X-Request-Id`, `traceparent` | not handled | worker middleware; frontend forwards the headers (it already forwards the rest) |
| `Server-Timing` | none | worker middleware around the inference call; frontend appends `proxy` |
| `served_by`, `model` = checkpoint | `model` is always `laya-rl-agent` | worker: add from the loaded agent (it already reports these in `/health`) |
| `/livez`, `/readyz` | worker binds after warmup, `/health` only | worker: bind first, gate `/readyz` on warmup |
| 503 + `Retry-After` on overload | requests queue without bound | worker: bounded queue in front of laya-serve's single inference thread |
| 415 on non-JSON bodies | parsed regardless of Content-Type | worker middleware |
| limits in `/readyz` | not advertised | worker |

Client compatibility during the change: branch on the status code, read `code` when the body is
problem+json and fall back to `detail`; read identity from `served_by` when present, else from the
`/health` snapshot taken at warm-up.

## 5. Client design (system1-agents)

- **Selection.** `--model laya-served`, a new name so run records say served Laya, not Jev or in-process Laya.
- **Configuration.** `LAYA_SERVED_URL` (required), `LAYA_SERVED_MODEL` (default `english`),
  `LAYA_SERVED_API_KEY` (optional), `LAYA_SERVED_TIMEOUT_S` (default 5, one deadline per decision,
  retries included).
- **Request.** Questions serialised with the existing `laya_question()`, which keeps Laya's own `noul`
  shape (a plain-string instruction). `score` is not sent until an agent needs it.
- **Identity.** Each response's `served_by` goes into the run record; until servers send it, `warm()`
  reads `/health` once and records checkpoint, revision, device, dtypes and compile mode.
- **Errors → agent errors.**

  | outcome | handling |
  |---|---|
  | connection refused / reset | retry once within the deadline (worker may be starting or restarting) |
  | 502, 504 | retry once within the deadline |
  | 503 (overloaded or not ready) | wait `Retry-After` if it fits the deadline, then retry once |
  | 400, 413, 422 | fail at once: the request is wrong, a retry returns the same |
  | 401 | fail at once as a configuration error |
  | 500 | fail at once: the same request fails the same way |
  | deadline passed | fail with a timeout error naming the URL |

- **Timing.** The record keeps the client round trip per decision and, when present, `Server-Timing`'s
  `queue` and `infer`, so network, queueing and model time separate.
- **Tracing.** The client sends an `X-Request-Id` per decision and stores it with the step.

## 6. Trade-offs

| decision | chosen | alternative | why |
|---|---|---|---|
| spec | hand-written OpenAPI 3.1 target, plus an as-implemented spec checked against captured traffic | generate from FastAPI | laya-serve reads the raw body, so FastAPI's generated schema has no request or response shape |
| errors | RFC 9457 with a stable `code` | keep `{"detail"}` | clients need a machine-readable reason; `detail` wording changes between laya versions |
| retries | client owns them; servers never retry | retries in the frontend | the client knows its step deadline; a retrying proxy multiplies load when the worker is saturated |
| identity | per-response `served_by` | only `/health` | a record then stays correct across worker restarts and multi-model routing |
| overload | 503 + `Retry-After` from a bounded queue | 429 | the limit is server capacity, not a per-client quota |
| readiness | `/livez` + `/readyz` | bind only after warmup (today) | a supervisor can tell a slow start from a dead process |
| timing | `Server-Timing` header | a field in the body | standard, visible in tooling, keeps the Jev-compatible body unchanged |
| model name | `laya-served` | reuse `laya` with a URL switch | in-process and served runs stay distinguishable in records and evaluations |

## 7. Revisit when

- Several agents share one worker: per-client quotas (429) on top of the capacity limit.
- Throughput matters more than single-request latency: batching concurrent requests in the worker.
- A second model family (ThinkFlowLab/system1-omni#9) reuses `/v1/systemone`: move `served_by` and the problem codes into a
  shared contract instead of the Laya spec.
- `score` becomes useful to an agent: extend the client; the server already answers it.
