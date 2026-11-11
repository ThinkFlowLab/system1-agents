# coding: utf-8
"""A seeded WebHarbor slice: locally reset, verifier-graded browser tasks (issue 118).

``tasks`` loads the committed slice (task ids and revisions only, never tasks or images) and the task metadata
downloaded on demand; ``control`` talks to WebHarbor's token-protected control plane; ``environment`` resets a site
and runs the site's own verifier inside its container; ``verdict`` reads that verifier's output; ``score`` turns the
attempts into the result table; ``runner`` plays every (task, arm) attempt and records each one, failures included.
"""
