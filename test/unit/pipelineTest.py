"""Unit tests: config helpers, pipeline split/label helpers (no SAFE / GrowingNN training)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from impl.config import (
    TRAIN_VAL_SPLIT_RANDOM_STATE,
    _word_extraction_stride,
    iter_embedding_param_combos,
    resolve_embedding_workers,
)
from impl.pipeline import (
    _allocate_val_per_class_hamilton,
    _split_train_val,
    _train_val_random_state,
    encode_labels,
)


def test_word_extraction_stride_examples():
    assert _word_extraction_stride(6, 0.0) == 6
    assert _word_extraction_stride(6, 0.25) == 4
    assert _word_extraction_stride(4, 0.5) == 2
    assert _word_extraction_stride(1, 0.99) == 1


def test_train_val_random_state_stable():
    expected = int(
        hashlib.md5(f"BeetleFly:{TRAIN_VAL_SPLIT_RANDOM_STATE}".encode()).hexdigest()[:8],
        16,
    )
    assert _train_val_random_state("BeetleFly") == expected


def test_resolve_embedding_workers_clamps(monkeypatch):
    monkeypatch.setattr("impl.config.get_available_cpu_count", lambda: 4)
    w, avail = resolve_embedding_workers(100)
    assert avail == 4 and w == 4
    w2, _ = resolve_embedding_workers(None)
    assert w2 == 1
    w3, _ = resolve_embedding_workers(0)
    assert w3 == 1


def test_allocate_val_per_class_hamilton_balanced_two_classes():
    # 10 samples each class, 4 val total from 20 -> 2 per class
    floors = _allocate_val_per_class_hamilton([10, 10], 4, 20)
    assert floors.sum() == 4
    assert list(floors) == [2, 2]


def test_split_train_val_stratified_sizes_and_reproducible():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3))
    y = np.array([0] * 10 + [1] * 10)
    x_tr1, x_val1, y_tr1, y_val1 = _split_train_val(x, y, 0.2, 12345)
    x_tr2, x_val2, y_tr2, y_val2 = _split_train_val(x, y, 0.2, 12345)
    assert len(x_val1) == len(x_val2) == 4
    assert np.array_equal(x_tr1, x_tr2) and np.array_equal(x_val1, x_val2)
    assert set(np.unique(y_val1)) <= {0, 1}


def test_split_train_val_rejects_bad_fraction():
    x = np.zeros((5, 1))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="val_fraction"):
        _split_train_val(x, y, 0.0, 0)
    with pytest.raises(ValueError, match="at least 2"):
        _split_train_val(np.zeros((1, 1)), np.zeros(1), 0.2, 0)


def test_encode_labels_roundtrip_strings():
    y_train = np.array(["a", "b", "a"])
    y_test = np.array(["b", "a"])
    yt, ys = encode_labels(y_train, y_test)
    assert yt.shape == (3,) and ys.shape == (2,)
    assert set(yt.tolist()) <= {0, 1} and set(ys.tolist()) <= {0, 1}


def test_iter_embedding_param_combos_word2vec_shape():
    first = next(
        c
        for c in iter_embedding_param_combos()
        if c[1] == "word2vec"
    )
    emb_epochs, method, ws, ngram, dm, ov = first
    assert method == "word2vec"
    assert ngram is None and dm == 0
    assert emb_epochs in (10, 20, 30, 40, 50)
    assert isinstance(ov, (int, float))
