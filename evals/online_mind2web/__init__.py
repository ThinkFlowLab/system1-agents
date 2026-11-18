# coding: utf-8
"""A seeded Online-Mind2Web live slice: clean-profile browser tasks, WebJudge-graded (issue 116).

``tasks`` loads the committed slice (ids, levels, websites and the official revision; never the task text) and the
official file downloaded on demand; ``runner`` plays every (task, arm) attempt on the live web and records each one,
failures included; ``score`` turns the attempts into the result table. Success is WebJudge (threshold 3), not an
Allrecipes-style site check and not the #109 offline step collector.
"""
