"""Record /v1/systemone and /health responses from running servers into JSON fixtures."""

import http.client
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
server = sys.argv[2]
port = int(sys.argv[3])
token = sys.argv[4] if len(sys.argv) > 4 else ""
out.mkdir(parents=True, exist_ok=True)
Q = {
    "choice": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges and refunds", "technical": "Software problems"},
    },
    "score": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["Not urgent", "Needs attention soon", "Needs attention immediately"],
    },
    "noul": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
}
ST = "I was charged twice for my order. Please refund the duplicate today."


def req(method, path, body=None, auth=True, raw=False):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = {"Content-Type": "application/json"}
    if auth and token:
        h["Authorization"] = f"Bearer {token}"
    data = body if raw else (json.dumps(body).encode() if body is not None else None)
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    b = r.read()
    ctype = r.getheader("content-type", "")
    try:
        parsed = json.loads(b)
    except ValueError:
        parsed = b.decode(errors="replace")
    return {"status": r.status, "content_type": ctype.split(";")[0], "body": parsed}


cases = {"health": ("GET", "/health", None, True, False)}
if server != "frontend-down":
    cases.update(
        {
            "choice": (
                "POST",
                "/v1/systemone",
                {"model": "english", "state": ST, "questions": {"department": Q["choice"]}},
                True,
                False,
            ),
            "noul": (
                "POST",
                "/v1/systemone",
                {"model": "english", "state": ST, "questions": {"refund": Q["noul"]}},
                True,
                False,
            ),
            "combined": (
                "POST",
                "/v1/systemone",
                {
                    "model": "english",
                    "state": ST,
                    "questions": {"department": Q["choice"], "urgency": Q["score"], "refund": Q["noul"]},
                },
                True,
                False,
            ),
            "malformed_json": ("POST", "/v1/systemone", b"{not json", True, True),
            "no_questions": ("POST", "/v1/systemone", {"model": "english", "state": ST}, True, False),
            "invalid_question": (
                "POST",
                "/v1/systemone",
                {"model": "english", "state": ST, "questions": {"q": {"type": "bogus", "instructions": "?"}}},
                True,
                False,
            ),
            "too_many_questions": (
                "POST",
                "/v1/systemone",
                {"model": "english", "state": "x", "questions": {f"q{i}": Q["noul"] for i in range(65)}},
                True,
                False,
            ),
        }
    )
    if token:
        cases["unauthorized"] = (
            "POST",
            "/v1/systemone",
            {"model": "english", "state": ST, "questions": {"refund": Q["noul"]}},
            False,
            False,
        )
else:
    cases = {
        "unreachable": (
            "POST",
            "/v1/systemone",
            {"model": "english", "state": ST, "questions": {"refund": Q["noul"]}},
            True,
            False,
        )
    }
for name, (m, p, b, a, raw) in cases.items():
    rec = req(m, p, b, a, raw)
    if not raw and b is not None and name != "too_many_questions":
        rec["request"] = b
    (out / f"{server}.{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(f"{server:14} {name:18} {rec['status']} {rec['content_type']}")
