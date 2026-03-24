"""
Regression tests: lock stable metrics (combo cardinality, stride math, split seed) to catch drift.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from impl.config import TRAIN_VAL_SPLIT_RANDOM_STATE, _word_extraction_stride, iter_embedding_param_combos
from impl.pipeline import _split_train_val, _train_val_random_state


def test_embedding_combo_count_unchanged():
    combos = list(iter_embedding_param_combos())
    assert len(combos) == 308


def test_ppmi_svd_fixed_epochs_and_stride_regression():
    ppmi = [c for c in iter_embedding_param_combos() if c[1] == "ppmi_svd"]
    assert all(c[0] == 200 for c in ppmi)
    assert len(ppmi) == 28


def test_word_extraction_stride_regression_table():
    assert _word_extraction_stride(6, 0.25) == 4
    assert _word_extraction_stride(6, 0.1) == 5


def test_train_val_random_state_digest_regression():
    digest = hashlib.md5(f"BeetleFly:{TRAIN_VAL_SPLIT_RANDOM_STATE}".encode()).hexdigest()
    assert _train_val_random_state("BeetleFly") == int(digest[:8], 16)


def test_stratified_split_sizes_regression():
    x = np.arange(20, dtype=float).reshape(20, 1)
    y = np.array([0] * 10 + [1] * 10)
    x_tr, x_val, y_tr, y_val = _split_train_val(x, y, 0.2, 12345)
    assert len(x_tr) == 16 and len(x_val) == 4
    assert sorted(np.unique(y_val).tolist()) == [0, 1]
