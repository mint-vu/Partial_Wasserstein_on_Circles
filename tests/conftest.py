import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
