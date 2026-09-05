"""Ensures the repo root (and thus the ``qecsim`` package) is importable
during test collection without an editable install."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
