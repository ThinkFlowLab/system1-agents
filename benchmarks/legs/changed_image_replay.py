"""Replay a frozen inline-image JSONL sequence and retain every HTTP attempt."""

import argparse
import base64
from contextlib import closing
import json
import os
from pathlib import Path
import sys
import time

from benchmarks.client import DecisionClient, decode
from benchmarks.evidence import digest, dump, outside_repo, write_json


def image_hashes(request):
    state = request.get("state")
    if not isinstance(state, list):
        raise ValueError("state must be a list containing inline PNG/JPEG images")
    hashes = []
    for part in state:
        if not isinstance(part, dict):
            continue
        if "image" in part:
            uri = part["image"]
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            uri = image.get("url") if isinstance(image, dict) else None
        else:
            continue
        if not isinstance(uri, str):
            raise ValueError("image URI must be a string")
        header, separator, payload = uri.partition(",")
        signatures = {"data:image/png;base64": b"\x89PNG\r\n\x1a\n", "data:image/jpeg;base64": b"\xff\xd8\xff"}
        if not separator or header not in signatures:
            raise ValueError("Only inline base64 PNG/JPEG images are allowed")
        data = base64.b64decode(payload, validate=True)
        if not data.startswith(signatures[header]):
            raise ValueError("Image bytes do not match their PNG/JPEG MIME signature")
        hashes.append(digest(uri.encode("utf-8")))
    if not hashes:
        raise ValueError("Each request must contain at least one inline image")
    return hashes


def load_manifest(source):
    rows, seen = [], set()
    for number, line in enumerate(source.splitlines(), 1):
        try:
            row = decode(line)
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key] for key in ("id", "group")
            ):
                raise ValueError("Each row requires nonempty id and group strings")
            if row["id"] in seen:
                raise ValueError("Duplicate input id")
            request = row.get("request")
            if not isinstance(request, dict) or request.get("kind") not in ("choice", "noul", "score"):
                raise ValueError("request.kind must be choice, noul or score")
            if not isinstance(request.get("question"), str):
                raise ValueError("request.question must be a string")
            if request["kind"] == "choice":
                options = request.get("options")
                if (
                    not isinstance(options, list)
                    or not 2 <= len(options) <= 256
                    or any(not isinstance(option, str) for option in options)
                ):
                    raise ValueError("choice requires 2–256 string options")
            elif "options" in request:
                raise ValueError("options is only allowed for choice")
            hashes = image_hashes(request)
            dump(row).encode("utf-8")
        except ValueError as exc:
            raise ValueError(f"Manifest line {number}: {exc}") from exc
        seen.add(row["id"])
        rows.append((row, hashes))
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def response_error(request, response):
    if not isinstance(response, dict):
        return "Expected a JSON response object"
    kind = request["kind"]
    options = request.get("options", ["false", "true"] if kind == "noul" else list("012345"))
    if response.get("kind") != kind or response.get("effective_kind") != kind:
        return "Missing or mismatched kind/effective_kind"
    if response.get("options") != options:
        return "Response options differ from the requested kind or option order"
    probabilities = response.get("probabilities")
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != len(options)
        or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in probabilities)
    ):
        return "Expected one finite probability in [0, 1] per option"
    index = response.get("choice_index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(options):
        return "choice_index is outside the options"
    if response.get("choice") != options[index]:
        return "choice differs from options[choice_index]"
    return None


def record_for(row, hashes, sequence, raw, raw_path, output):
    code = raw["http_status"]
    if raw["client_error"] is not None:
        status, error = "client_error", raw["client_error"]
    elif code is None or not 200 <= code < 300:
        status, error = ("refused" if code in (400, 422) else "http_error"), raw["response_body"]
    else:
        error = response_error(row["request"], raw["response"])
        status = "parse_error" if error else "ok"
    text = raw["response_body"]
    return dict(
        sequence=sequence,
        id=row["id"],
        group=row["group"],
        kind=row["request"]["kind"],
        request_sha256=digest(dump(raw["request"]).encode("utf-8")),
        image_uri_sha256=hashes,
        response_body_sha256=digest(text.encode("utf-8")) if text is not None else None,
        raw_file=str(raw_path.relative_to(output)),
        raw_sha256=digest(raw_path.read_bytes()),
        ts=raw["ts"],
        latency_s=raw["latency_s"],
        status_code=code,
        status=status,
        ok=status == "ok",
        error=error,
    )


def predict(args):
    source = args.manifest.read_bytes()
    rows = load_manifest(source)
    environment = args.run_meta.read_bytes()
    meta = decode(environment)
    for group, fields in (("checkpoint", ("repo", "revision")), ("backend", ("name", "version"))):
        values = meta.get(group) if isinstance(meta, dict) else None
        if not isinstance(values, dict) or any(not isinstance(values.get(k), str) or not values[k] for k in fields):
            raise ValueError(f"run-meta requires actual {group}: {fields}")
    if args.path not in ("/v1/decide", "/v1/systemone"):
        raise ValueError("Choose /v1/decide or /v1/systemone explicitly")
    if args.cache_state not in ("unknown", "cold", "warm"):
        raise ValueError("cache-state must be unknown, cold or warm")
    output = outside_repo(args.out)
    with closing(DecisionClient(args.endpoint, path=args.path, model=None, timeout_s=args.timeout)) as client:
        output.mkdir(parents=True, exist_ok=False)
        (output / "raw").mkdir()
        (output / "manifest.jsonl").write_bytes(source)
        (output / "environment.json").write_bytes(environment)
        write_json(
            output / "run.meta.json",
            dict(
                argv=sys.argv,
                endpoint=args.endpoint,
                path=args.path,
                environment=meta,
                environment_file=str(args.run_meta.resolve()),
                environment_sha256=digest(environment),
                manifest_file=str(args.manifest.resolve()),
                manifest_sha256=digest(source),
                planned=len(rows),
                timeout_s=args.timeout,
                cache_state_declared=args.cache_state,
                order="manifest sequential; no warmup, retries or deduplication",
                latency_basis="client HTTP POST through response.text; excludes JSON decode and service startup",
                ts=time.time(),
            ),
        )
        attempted, succeeded, stop = 0, 0, None
        try:
            with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
                for sequence, (row, hashes) in enumerate(rows, 1):
                    attempted += 1
                    raw = client.request(row["request"])
                    raw_path = output / "raw" / f"{sequence:06d}.json"
                    # Escaping preserves decoded lone surrogates from malformed service responses.
                    with raw_path.open("x", encoding="utf-8") as raw_stream:
                        raw_stream.write(json.dumps(raw, ensure_ascii=True, allow_nan=False) + "\n")
                    record = record_for(row, hashes, sequence, raw, raw_path, output)
                    stream.write(dump(record) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    succeeded += record["ok"]
                    if record["status"] in ("client_error", "http_error"):
                        stop = raw["client_error"] or f"HTTP {raw['http_status']}"
                        break
        except KeyboardInterrupt:
            stop = "interrupted"
        except Exception as exc:
            stop = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_json(
                output / "completion.json",
                dict(
                    planned=len(rows),
                    attempted=attempted,
                    remaining=len(rows) - attempted,
                    succeeded=succeeded,
                    failed=attempted - succeeded,
                    stop_reason=stop,
                    ts=time.time(),
                ),
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--path", required=True, choices=("/v1/decide", "/v1/systemone"))
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--cache-state", choices=("unknown", "cold", "warm"), default="unknown")
    print(predict(parser.parse_args()))


if __name__ == "__main__":
    main()
