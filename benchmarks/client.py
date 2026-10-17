"""One synchronous POST per decision request; no retries or model inference."""

import json
import math
import time
from urllib.parse import urlsplit

import httpx


def decode(text):
    def invalid_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite JSON number: {value}")
        return result

    return json.loads(text, parse_constant=invalid_constant, parse_float=finite_float)


class DecisionClient:
    def __init__(self, endpoint, *, path, model=None, timeout_s=120, transport=None):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        url = urlsplit(endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Use an HTTP endpoint without credentials or query parameters")
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("path must be an absolute API path without query parameters")
        self.url, self.model = endpoint.rstrip("/") + path, model
        self.http = httpx.Client(timeout=timeout_s, transport=transport, follow_redirects=False, trust_env=False)

    def request(self, body):
        request = dict(body)
        if self.model is not None:
            request.setdefault("model", self.model)
        request = decode(json.dumps(request, ensure_ascii=False, allow_nan=False))
        raw = dict(request=request, response_body=None, response=None, http_status=None, client_error=None)
        raw["ts"] = time.time()
        started = time.perf_counter()
        try:
            response = self.http.post(self.url, json=request)
            raw.update(http_status=response.status_code, response_body=response.text)
        except httpx.HTTPError as exc:
            raw["client_error"] = f"{type(exc).__name__}: {exc}"
        except KeyboardInterrupt:
            raw["client_error"] = "KeyboardInterrupt"
        finally:
            raw["latency_s"] = time.perf_counter() - started
        if raw["response_body"] is not None:
            try:
                raw["response"] = decode(raw["response_body"])
            except ValueError:
                pass  # The original body remains available for offline diagnosis.
        return raw

    def close(self):
        self.http.close()
