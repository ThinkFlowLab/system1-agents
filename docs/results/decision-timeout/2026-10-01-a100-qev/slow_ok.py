# Test-only launcher: s1a with a 30 s limit per decision instead of 5 s, for local models slower than Jev.
import sys

import s1a.decision_models.jev as jev

_orig = jev.JevModel.from_env.__func__
jev.JevModel.from_env = classmethod(lambda cls, timeout_s=30.0: _orig(cls, timeout_s=timeout_s))

from s1a.entry import s1a  # noqa: E402

sys.argv = ["s1a", *sys.argv[1:]]
sys.exit(s1a())
