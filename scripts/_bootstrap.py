"""Shared bootstrap so scripts run from a checkout without `pip install -e .`.

Import this first in every script: `import _bootstrap  # noqa`.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
