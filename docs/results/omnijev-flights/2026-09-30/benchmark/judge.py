"""Vision judge: does the final screenshot show the task done? One call to a vision chat model through OpenRouter.

Usage: python judge.py final.png "<the task>"
Prints one JSON line: {"success": true|false, "reason": "...", "judge": "<model>"}.
Reads OPENROUTER_API_KEY from the environment or from system1-agents/.env; JUDGE_MODEL picks the model.
"""

import base64
import json
import os
import pathlib
import re
import sys
import urllib.request

JUDGE_MODEL = os.getenv("JUDGE_MODEL") or "google/gemini-2.5-flash"
PROMPT = (
    "You are a strict judge of a web task. Task given to a browser agent:\n{task}\n\n"
    "Look only at the screenshot of the page where the agent stopped. The task counts as done ONLY if the page "
    "shows a list of flight results that match the task: the right origin, the right destination, the right date, "
    "one way. A filled form without results, an error page, 'No results', or results for another route or date "
    "is NOT done.\n"
    'Answer with JSON only: {{"success": true or false, "reason": "<one short sentence>"}}'
)


def api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    env = pathlib.Path(__file__).with_name("system1-agents") / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("no OPENROUTER_API_KEY")


def main() -> None:
    image = base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode()
    body = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT.format(task=sys.argv[2])},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    reply = json.load(urllib.request.urlopen(request, timeout=60))["choices"][0]["message"]["content"]
    found = re.search(r"\{.*\}", reply, re.S)
    try:
        verdict = json.loads(found.group(0)) if found else {}
    except json.JSONDecodeError:
        verdict = {}
    print(
        json.dumps(
            {
                "success": bool(verdict.get("success")),
                "reason": str(verdict.get("reason") or reply.strip())[:200],
                "judge": JUDGE_MODEL,
            }
        )
    )


if __name__ == "__main__":
    main()
