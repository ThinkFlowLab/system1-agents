# coding: utf-8
"""A decision model by name, from the environment."""

from __future__ import annotations

import os

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.baselines import RandomModel, Rule, RuleModel
from s1a.decision_models.clm import ClmModel
from s1a.decision_models.cua import CuaS1Model
from s1a.decision_models.cua_four_b import CuaFourBModel
from s1a.decision_models.jev import JevModel
from s1a.decision_models.laya import LayaModel
from s1a.decision_models.omnijev import OmniJevModel
from s1a.decision_models.served import ServedLayaModel

DECISION_MODEL_NAMES = (
    "jev",
    "clm",
    "laya",
    "laya-served",
    "cua",
    "omnijev",
    "random",
    "rule",
)  # the names that build a decision model; ``llm`` is not one


def build_model(model_name: str, *, seed: int = 0, rule: tuple[str, Rule] | None = None) -> DecisionModel:
    """``jev``, ``clm``, ``laya``, ``laya-served``, ``cua`` and ``omnijev`` from the environment, ``random`` from the seed, ``rule`` from the agent's baseline."""
    match model_name:
        case "jev":
            return JevModel.from_env()
        case "clm":
            return ClmModel.from_env()  # CLM over HTTP; no torch, no cloud key
        case "laya":
            return LayaModel.from_env()  # the laya import happens inside
        case "laya-served":
            return ServedLayaModel.from_env()  # Laya over HTTP; no torch, no cloud key
        case "cua":
            variant = os.getenv("CUA_S1_VARIANT", "nano")
            if variant == "4b":
                return CuaFourBModel.from_env()
            if variant != "nano":
                raise ValueError("CUA_S1_VARIANT must be nano or 4b")
            return CuaS1Model.from_env()  # the cua_s1 import happens inside
        case "omnijev":
            return OmniJevModel.from_env()  # the OmniJev import happens inside; it decides over a screenshot
        case "random":
            return RandomModel(seed)
        case "rule":
            if rule is None:
                raise RuntimeError("this agent has no rule baseline")
            return RuleModel(*rule)
        case _:
            raise ValueError(f"unknown decision model {model_name!r}; one of {DECISION_MODEL_NAMES}")
