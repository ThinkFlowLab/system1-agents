"""Every run in this folder: its final page and its actions, read from the run's own records.

    python docs/results/omnijev-flights/summarize.py > docs/results/omnijev-flights/actions.txt

The final page of a run that ends on openJiuwen's form budget is the ``current_page`` of its ``browser_result``; the
search date comes from the results URL's ``tfs`` parameter.
"""

import base64
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def final_page(answer):
    terminal = answer.get("terminal") or {}
    if terminal.get("url"):
        return answer.get("status"), terminal["url"]
    result = json.loads(answer.get("final") or "{}").get("browser_result", {})
    return "partial: " + ",".join(result.get("blockers", [])), (result.get("current_page") or {}).get("url")


def search_date(url):
    match = re.search(r"tfs=([A-Za-z0-9_\-]+)", url or "")
    if not match:
        return "-"
    raw = base64.urlsafe_b64decode(match.group(1) + "=" * (-len(match.group(1)) % 4))
    date = re.search(rb"20\d\d-\d\d-\d\d", raw)
    return date.group().decode() if date else "-"


for run in sorted(glob.glob(os.path.join(HERE, "2026-09-*", "2026-*"))):
    answer = json.load(open(os.path.join(run, "answer.json"), encoding="utf-8"))
    ticks = json.load(open(os.path.join(run, "decision_ticks.json"), encoding="utf-8")).get("ticks", [])
    status, url = final_page(answer)
    page = "results page" if url and "/flights/search" in url else "home page / form"
    print(f"== {os.path.basename(run)}  model={answer.get('model')}  decisions={len(ticks)}  end={status}")
    print(f"   final page: {page}, search date {search_date(url)}")
    for tick in ticks:
        target = str(tick.get("target"))[:70]
        print(
            f"   {tick['tick']:>2} {tick['operation']:<11} {target:<70} conf {tick.get('confidence')}"
            f"  {tick.get('decision_ms')} ms, screenshot {tick.get('screenshot_ms')} ms"
        )
    print()
