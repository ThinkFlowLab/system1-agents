"""Bridge: laya-mlx snake demo -> system1-omni /v1/systemone backend."""

import json
import urllib.request


class SystemOneBackend:
    """Same predict(state, questions) shape as laya_mlx.Agent."""

    def __init__(self, base_url, model="english", timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health(self):
        with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout) as response:
            return json.loads(response.read())

    def predict(self, state, questions):
        request = urllib.request.Request(
            self.base_url + "/v1/systemone",
            data=json.dumps({"model": self.model, "state": state, "questions": questions}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())
