# coding: utf-8
"""Environment-driven configuration shared by the solvers, the evals and the examples."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openjiuwen.core.foundation.llm import Model, init_model
from openjiuwen.harness.tools.browser_move.utils.env import DEFAULT_PLAYWRIGHT_MCP_ARGS, load_repo_dotenv

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ["S1A_HOME"]).resolve() if os.environ.get("S1A_HOME") else ROOT  # runtime output root
JEV_ENV = ("TYPESAFE_API_KEY", "TYPESAFE_API_URL", "TYPESAFE_MODEL", "OPENROUTER_API_KEY")  # read raw by wire.py
ENV_ALIASES = (  # OPENROUTER_* stands in for the chat model's LLM_*
    ("LLM_API_KEY", "OPENROUTER_API_KEY"),
    ("LLM_BASE_URL", "OPENROUTER_BASE_URL"),
)


def first_env(*names: str) -> str:
    """The first non-empty value among the named variables, stripped; empty when none is set."""
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def load_env() -> None:
    """The repo's own .env, then the openjiuwen checkout's (a dev install); exported variables win."""
    load_dotenv(ROOT / ".env", override=False)
    load_repo_dotenv()
    for name in JEV_ENV:  # a whitespace-only key is an empty key, reported at startup rather than as a 401 later
        if name in os.environ:
            os.environ[name] = os.environ[name].strip()
    for alias, source in ENV_ALIASES:
        if not os.getenv(alias) and os.getenv(source):
            os.environ[alias] = os.environ[source]


def chat_model_from_env() -> Model:
    """The chat model as agent-eval's runner configures it: OPENAI_API_KEY or LLM_API_KEY, OPENAI_BASE_URL, MODEL_NAME."""
    api_key = first_env("OPENAI_API_KEY", "LLM_API_KEY")
    model_name = first_env("MODEL_NAME")
    if not api_key or not model_name:
        raise RuntimeError("chat model: set OPENAI_API_KEY (or LLM_API_KEY) and MODEL_NAME")
    provider = first_env("MODEL_PROVIDER") or "openai"
    headers: dict[str, str] = {}
    extra_body: dict[str, object] = {}
    if provider.lower() == "anthropic":
        # the DeepAgent re-inserts its working-context message every call; Claude then rejects replayed thinking blocks
        extra_body["thinking"] = {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
        headers["anthropic-beta"] = "thinking-binding-controls-2026-08-01"
        workspace = first_env(
            "ANTHROPIC_WORKSPACE_ID"
        )  # an org-level Anthropic key must name its workspace per request
        if workspace:
            headers["anthropic-workspace-id"] = workspace
    return init_model(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        api_base=first_env("OPENAI_BASE_URL", "LLM_BASE_URL") or "https://api.openai.com/v1",
        temperature=0.0,
        custom_headers=headers or None,
        extra_body=extra_body or None,
    )


def optional_chat_model() -> Model | None:
    """The chat model when the environment names one; None otherwise (the models that need it raise)."""
    try:
        return chat_model_from_env()
    except RuntimeError:
        return None


def browser_launch_args(headless: bool) -> str:
    """MCP launch flags: the configured server (PLAYWRIGHT_MCP_ARGS, else the npx package) with an in-memory profile
    per run, headless when asked. Args that attach to a running browser (``--cdp-endpoint``) are left alone."""
    configured = first_env("PLAYWRIGHT_MCP_ARGS")
    if "--cdp-endpoint" in configured:
        return configured
    base = (configured or DEFAULT_PLAYWRIGHT_MCP_ARGS).replace(" --isolated", "").replace(" --headless", "")
    return f"{base} --isolated" + (" --headless" if headless else "")
