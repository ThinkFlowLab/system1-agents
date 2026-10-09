"""Record real clm-serve responses as fixtures, the way tests/data/served_laya/ was recorded.

    python3 record.py http://127.0.0.1:8091                  # every fixture
    python3 record.py http://127.0.0.1:8091 browser_target   # only this one

Writes tests/data/clm/<name>.json as {"status", "content_type", "body"} (or "text" for a body that is not JSON),
so the tests replay what a server actually sent rather than what we imagine it sends.

Name the fixtures you mean when the encoder behind the URL is not the trained one. ``choice`` and ``noul`` carry
the trained encoder's numbers, which the tests assert; recording everything against ``recipe/clm/native``'s CPU
stub replaces them with the stub's, and nothing about the file says which it holds.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent

CHOICE = {
    "model": "clm-latest",
    "state": {"ticket": "I was charged twice for order 4411."},
    "questions": {
        "pick": {
            "type": "choice",
            "instructions": "Which queue?",
            "criteria": {"billing": "Charges and refunds", "technical": "Software problems"},
        }
    },
}
NOUL = {
    "model": "clm-latest",
    "state": {"ticket": "Please refund the duplicate charge."},
    "questions": {"check": {"type": "noul", "instructions": "Does the customer ask for a refund?"}},
}
BAD = {"model": "clm-latest", "state": {}, "questions": {"pick": {"type": "choice", "criteria": {}}}}
# What the browser front asks. s1a/browser/action_space.py builds `<op>_target` criteria as objects --
# {element, current_value, option?, role/checked/selected/expanded/region?} -- so strings, empty strings and bools
# all reach clm-serve. Recorded to pin that it takes them (a later maintainer asked; it does, 200 not 422).
#
# NOTE: recorded against a local clm-serve whose encoder is recipe/clm/native's CPU stub, so the numbers below are
# the stub's. Only the status and the answer's shape are evidence here; tests must not assert these probabilities.
BROWSER = {
    "model": "clm-latest",
    "state": {"url": "https://example.test/checkout", "title": "Checkout"},
    "questions": {
        "click_target": {
            "type": "choice",
            "instructions": "Pick the element to click.",
            "criteria": {
                "e1": {"element": "[e1] Sign in", "current_value": "", "role": "button"},
                "e2": {"element": "[e2] Apply coupon", "current_value": "", "role": "button"},
                "e3": {"element": "[e3] Gift wrap", "current_value": "", "role": "checkbox", "checked": True},
            },
        }
    },
}


def record(url: str, name: str, body: dict | None, method: str = "POST", path: str = "/v1/systemone") -> None:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status, content_type, raw = response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        status, content_type, raw = error.code, error.headers.get("Content-Type", ""), error.read()
    entry: dict = {"status": status, "content_type": content_type.split(";")[0]}
    try:
        entry["body"] = json.loads(raw)
    except ValueError:
        entry["text"] = raw.decode("utf-8", "replace")
    (OUT / f"{name}.json").write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    print(f"{name}: {status} {entry.get('body', entry.get('text'))!r}"[:160])


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091"
    wanted = set(sys.argv[2:])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body, kwargs in (
        ("models", None, {"method": "GET", "path": "/v1/models"}),
        ("choice", CHOICE, {}),
        ("noul", NOUL, {}),
        ("bad_question", BAD, {}),
        ("browser_target", BROWSER, {}),
    ):
        if wanted and name not in wanted:
            continue
        record(url, name, body, **kwargs)
    unknown = wanted - {"models", "choice", "noul", "bad_question", "browser_target"}
    if unknown:
        raise SystemExit(f"no such fixture: {', '.join(sorted(unknown))}")


if __name__ == "__main__":
    main()
