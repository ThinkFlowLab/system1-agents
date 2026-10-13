# coding: utf-8
"""The repeatable bounded-recovery on/off subset: three local form tasks through the real browser front.

This package is an evaluation, not a runner for production code. It serves three fixture pages from a localhost
server, drives each one through the repository's ``browse`` front with a clearly-marked scripted decision model and
scripted planner, and judges completion only by the fixture server's own record of the real submit POST. Each trial
opens its own fresh fixture, so no earlier trial's submit can verify a later one. It never reads the model's
DONE/answer to decide success. See ``evals/recovery/__main__.py`` and ``evals/README.md``.
"""
