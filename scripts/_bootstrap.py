"""Put the repository root on ``sys.path`` so ``import qecsim`` works when a
script is run directly (``python scripts/foo.py``) without installing."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
