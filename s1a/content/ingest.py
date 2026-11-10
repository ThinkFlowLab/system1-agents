# coding: utf-8
"""Pluggable ingest. RSS is implemented. Twitter/X is a fixture plus an adapter hook.

A live X call is not shipped: CI has no X credentials, and this module does not read
``X_BEARER_TOKEN`` or open ``api.x.com``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree

import httpx

SOURCES = ("rss", "x", "fixture")
SCREEN_LABELS = ("keep", "drop")
FACT_LABELS = ("supported", "unsupported", "needs_human")
MAX_RSS_BYTES = 2_000_000

LIVE_X = (
    "Twitter/X ingest is not configured. Pass --x-fixture, or pass x_fetch to load_inputs. "
    "A live adapter reads X_BEARER_TOKEN itself and sends GET https://api.x.com/2/tweets/search/recent "
    "with query and max_results, then maps posts through posts_to_items. "
    "s1a.content does not send that request."
)


class UnconfiguredXAdapter:
    """The default X adapter. ``fetch`` fails closed and does not open a socket."""

    def fetch(self) -> list[dict[str, Any]]:
        raise ValueError(LIVE_X)


def normalize_item(row: Any) -> dict[str, Any]:
    """Copy the fields the desk may read. Unknown keys, including labels' neighbours, are dropped."""
    if not isinstance(row, dict):
        raise ValueError("a content item must be a JSON object")
    item: dict[str, Any] = {}
    for key in ("id", "title", "summary", "source"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"content items need a non-empty string {key}")
        item[key] = value.strip()
    if item["source"] not in SOURCES:
        raise ValueError(f"content source must be one of {', '.join(SOURCES)}")
    if "url" in row and row["url"] is not None:
        if not isinstance(row["url"], str) or not row["url"].strip():
            raise ValueError("url, when present, must be a non-empty string")
        item["url"] = row["url"].strip()
    if "label" in row and row["label"] is not None:
        if row["label"] not in SCREEN_LABELS:
            raise ValueError("screen label must be keep or drop")
        item["label"] = row["label"]
    if "draft" in row and row["draft"] is not None:
        item["draft"] = normalize_draft(row["draft"])
    return item


def normalize_draft(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("a draft must be a JSON object")
    claims = row.get("claims")
    sources = row.get("sources")
    claims_ok = (
        isinstance(claims, list) and claims and all(isinstance(claim, str) and claim.strip() for claim in claims)
    )
    if not claims_ok:
        raise ValueError("a draft needs a non-empty list of claim strings")
    if not isinstance(sources, list) or not sources:
        raise ValueError("a draft needs at least one source")
    cleaned_sources = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("a draft source must be an object with id and text")
        sid, text = source.get("id"), source.get("text")
        if not isinstance(sid, str) or not sid.strip() or not isinstance(text, str) or not text.strip():
            raise ValueError("a draft source needs non-empty string id and text")
        cleaned_sources.append({"id": sid.strip(), "text": text.strip()})
    draft: dict[str, Any] = {"claims": [claim.strip() for claim in claims], "sources": cleaned_sources}
    if "label" in row and row["label"] is not None:
        if row["label"] not in FACT_LABELS:
            raise ValueError("fact-check label must be supported, unsupported or needs_human")
        draft["label"] = row["label"]
    return draft


def public_view(item: dict[str, Any]) -> dict[str, Any]:
    """What a decider may see. Screen and fact labels are not copied."""
    view: dict[str, Any] = {key: item[key] for key in ("id", "title", "summary", "source")}
    if "url" in item:
        view["url"] = item["url"]
    draft = item.get("draft")
    if draft:
        view["draft"] = {
            "claims": list(draft["claims"]),
            "sources": [{"id": source["id"], "text": source["text"]} for source in draft["sources"]],
        }
    return view


def screen_view(item: dict[str, Any]) -> dict[str, Any]:
    view = public_view(item)
    view.pop("draft", None)
    return view


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """News items in JSONL. A bad line names the file and the column, not the line's text."""
    items = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"content items {path}: line {line_number}, column {exc.colno}: {exc.msg}") from exc
        try:
            items.append(normalize_item(row))
        except ValueError as exc:
            raise ValueError(f"content items {path}: line {line_number}: {exc}") from exc
    return items


def posts_to_items(posts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map X-shaped posts (``id``, ``text``) or ready news items into desk items."""
    items = []
    for post in posts:
        if not isinstance(post, dict):
            raise ValueError("an X post must be a JSON object")
        if "text" in post and "summary" not in post:
            text = post.get("text")
            ident = post.get("id")
            if not isinstance(ident, str) or not ident.strip() or not isinstance(text, str) or not text.strip():
                raise ValueError("an X post needs non-empty string id and text")
            row: dict[str, Any] = {
                "id": ident.strip(),
                "title": (post.get("title") or text).strip()[:80],
                "summary": text.strip(),
                "source": "x",
            }
            if post.get("url"):
                row["url"] = post["url"]
            if post.get("label"):
                row["label"] = post["label"]
            items.append(normalize_item(row))
        else:
            items.append(normalize_item(post))
    return items


def load_x_fixture(path: Path) -> list[dict[str, Any]]:
    posts = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            posts.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"x fixture {path}: line {line_number}, column {exc.colno}: {exc.msg}") from exc
    try:
        return posts_to_items(posts)
    except ValueError as exc:
        raise ValueError(f"x fixture {path}: {exc}") from exc


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    raw = "".join(element.itertext())
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _child(parent: ElementTree.Element, name: str) -> ElementTree.Element | None:
    for child in list(parent):
        if _local(child.tag) == name:
            return child
    return None


def parse_feed(xml_text: str, *, origin: str) -> list[dict[str, Any]]:
    """RSS 2.0 items or Atom entries. Ids prefer guid, then id, then link."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise ValueError(f"feed {origin}: {exc}") from exc
    parents = [element for element in root.iter() if _local(element.tag) in {"channel", "feed"}]
    if not parents:
        parents = [root]
    rows = []
    for parent in parents:
        for element in list(parent):
            kind = _local(element.tag)
            if kind not in {"item", "entry"}:
                continue
            title = _text(_child(element, "title"))
            summary = (
                _text(_child(element, "description"))
                or _text(_child(element, "summary"))
                or _text(_child(element, "content"))
            )
            link = _link(element)
            guid = _text(_child(element, "guid")) or _text(_child(element, "id")) or link
            if not title or not summary or not guid:
                raise ValueError(f"feed {origin}: an entry needs a title, a summary and an id or link")
            row: dict[str, Any] = {"id": guid, "title": title, "summary": summary, "source": "rss"}
            if link:
                row["url"] = link
            rows.append(normalize_item(row))
    if not rows:
        raise ValueError(f"feed {origin}: no entries")
    return rows


def _link(element: ElementTree.Element) -> str:
    for child in list(element):
        if _local(child.tag) != "link":
            continue
        href = (child.get("href") or "").strip()
        if href:
            return href
        if child.text and child.text.strip():
            return child.text.strip()
    return ""


def load_rss_file(path: Path) -> list[dict[str, Any]]:
    return parse_feed(path.read_text(encoding="utf-8"), origin=str(path))


def check_http_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("a feed URL must be http(s) without credentials")
    return url.strip()


def load_rss_url(url: str, *, client: httpx.Client) -> list[dict[str, Any]]:
    """GET one feed. The caller owns the client; tests pass a mock transport."""
    target = check_http_url(url)
    try:
        response = client.get(target)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"feed {target} was not fetched: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise RuntimeError(f"feed {target} returned HTTP {response.status_code}")
    body = response.content
    if len(body) > MAX_RSS_BYTES:
        raise ValueError(f"feed {target} is larger than {MAX_RSS_BYTES} bytes")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"feed {target} is not utf-8") from exc
    return parse_feed(text, origin=target)


def ensure_unique(items: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for item in items:
        if item["id"] in seen:
            raise ValueError(f"duplicate content id {item['id']!r}")
        seen.add(item["id"])


def load_inputs(
    *,
    items: Path | None = None,
    rss_files: list[Path] | None = None,
    rss_urls: list[str] | None = None,
    x_files: list[Path] | None = None,
    x_fetch: Callable[[], list[dict[str, Any]]] | None = None,
    http: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Fixture JSONL, then RSS files, then RSS URLs, then X fixtures, then an optional adapter."""
    loaded: list[dict[str, Any]] = []
    if items is not None:
        loaded.extend(load_jsonl(items))
    for path in rss_files or []:
        loaded.extend(load_rss_file(path))
    if rss_urls:
        if http is None:
            raise ValueError("rss urls need an HTTP client")
        for url in rss_urls:
            loaded.extend(load_rss_url(url, client=http))
    for path in x_files or []:
        loaded.extend(load_x_fixture(path))
    if x_fetch is not None:
        loaded.extend(posts_to_items(x_fetch()))
    if not loaded:
        raise ValueError("content workflow needs at least one item")
    ensure_unique(loaded)
    return loaded
