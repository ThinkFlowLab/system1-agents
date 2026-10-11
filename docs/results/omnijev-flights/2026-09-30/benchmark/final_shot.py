"""Screenshot of the Google Flights tab the run ended on (Chrome on port 9222), for the judge.

Usage: python final_shot.py out.png      (prints the page URL; exits 1 when there is no flights tab)
"""

import base64
import json
import sys
import urllib.request

import websocket

tabs = [t for t in json.load(urllib.request.urlopen("http://127.0.0.1:9222/json/list")) if t["type"] == "page"]
tabs = [t for t in tabs if "google.com/travel" in t.get("url", "")] or tabs
if not tabs:
    print("no page tab")
    sys.exit(1)
tab = tabs[0]  # Chrome lists the most recently used tab first
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=30, suppress_origin=True)
ws.send(json.dumps({"id": 1, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
while True:
    msg = json.loads(ws.recv())
    if msg.get("id") == 1:
        break
ws.close()
open(sys.argv[1], "wb").write(base64.b64decode(msg["result"]["data"]))
print(tab.get("url", ""))
