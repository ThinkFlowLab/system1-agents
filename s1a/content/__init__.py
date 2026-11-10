# coding: utf-8
"""Desk workflow: ingest, screen, human pick, fact-check, human publish decision.

System1-Agents owns this orchestration. A draft System1-Omni worker is used only when
``OMNI_BASE_URL`` is set, and only for the screen and fact-check questions. The package
never posts a story.
"""

from s1a.content.workflow import GateRequired, run_workflow

__all__ = ["GateRequired", "run_workflow"]
