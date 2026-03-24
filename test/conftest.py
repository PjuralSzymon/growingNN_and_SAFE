"""
Pytest configuration: repo root on sys.path; lightweight stubs when optional deps are missing.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pipeline imports safe2 at module load time; stub if not installed
if "safe2" not in sys.modules:
    _safe2 = types.ModuleType("safe2")

    class SafeTransformer:  # noqa: D401
        """Placeholder; unit/integration tests patch pipeline.SafeTransformer."""

        pass

    _safe2.SafeTransformer = SafeTransformer
    sys.modules["safe2"] = _safe2

# kerne imports growingnn at module load; provide a MagicMock module if import fails
if "growingnn" not in sys.modules:
    try:
        import growingnn  # noqa: F401
    except ImportError:
        import unittest.mock as _mock

        _gnn = _mock.MagicMock()
        _gnn.montecarlo_alg = object()
        _gnn.greedy_alg = object()
        _gnn.random_alg = object()
        _gnn.SimulationScheduler.PROGRESS_CHECK = 0
        _gnn.SimulationScheduler.CONSTANT = 1
        _gnn.LearningRateScheduler.PROGRESIVE = 0
        _gnn.create_simulation_set_SAMLE = lambda *a, **k: None
        sys.modules["growingnn"] = _gnn
