# coding: utf-8
"""The decision-model layer: typed questions over an observation, validated answers, one adapter per backend.

Nothing in here knows about DeepAgent, browsers or environments. ``laya`` and ``cua_s1`` are imported only inside
their models' ``from_env``, so the package imports without those extras.
"""

from __future__ import annotations

from s1a.decision_models.base import DecisionModel, JevTransport
from s1a.decision_models.baselines import RandomModel, Rule, RuleModel
from s1a.decision_models.cua import CuaS1Model
from s1a.decision_models.factory import DECISION_MODEL_NAMES, build_model
from s1a.decision_models.fakes import ScriptedModel, ScriptedTransport
from s1a.decision_models.jev import JevModel, jev_question
from s1a.decision_models.laya import LayaModel
from s1a.decision_models.types import (
    Answer,
    Choice,
    ChoiceQuestion,
    Decision,
    Image,
    Json,
    Noul,
    NoulQuestion,
    Observation,
    Question,
    Reply,
    Usage,
)
from s1a.decision_models.validation import choice_faults, validate_answers, validate_choice, validate_noul

__all__ = [
    "DECISION_MODEL_NAMES",
    "Answer",
    "DecisionModel",
    "JevTransport",
    "ScriptedModel",
    "ScriptedTransport",
    "Choice",
    "ChoiceQuestion",
    "CuaS1Model",
    "Decision",
    "Image",
    "JevModel",
    "Json",
    "LayaModel",
    "Noul",
    "NoulQuestion",
    "Observation",
    "Question",
    "RandomModel",
    "Reply",
    "Rule",
    "RuleModel",
    "Usage",
    "build_model",
    "choice_faults",
    "jev_question",
    "validate_answers",
    "validate_choice",
    "validate_noul",
]
