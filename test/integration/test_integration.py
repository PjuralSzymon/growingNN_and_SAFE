"""
Integration tests: full run_single_experiment wiring with mocks (no real SAFE / GPU work).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from impl.pipeline import run_single_experiment


def _minimal_base_params():
    return {
        "epochs": 1,
        "generations": 1,
        "hidden_size": 8,
        "simulation_set_size": 4,
        "simulation_time": 1,
        "simulation_epochs": 1,
        "simulation_scheduler_type": "progress_check",
        "ACTIVATION_FUN": "Sigmoid",
        "convolution": False,
        "augmentation_mode": None,
        "DATA_augmentation_factor_jitter": 0,
        "DATA_augmentation_factor_warp": 0,
        "DATA_augmentation_factor_rgw": 0,
        "DATA_MIN_SAMPLES_PER_CLASS": 1,
        "optimization_factor": 0.5,
        "STOPPER_TARGET_ACCURACY": 0.85,
    }


def _tiny_arrays(n_per_class=4, n_features=16):
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2 * n_per_class, n_features)).astype(np.float32)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return x, y


@patch("impl.pipeline.kerne.train_growingnn")
@patch("impl.pipeline.SafeTransformer")
@patch(
    "impl.pipeline.iter_embedding_param_combos",
    return_value=[
        (10, "word2vec", 1, None, 0, 0.0),
        (20, "word2vec", 2, None, 0, 0.1),
    ],
)
def test_run_single_experiment_selects_best_val_acc(_mock_iter, mock_safe_cls, mock_train):
    rng = np.random.default_rng(0)

    def _transform(x):
        n = len(x)
        return rng.standard_normal((n, 4)).astype(np.float32)

    mock_inst = MagicMock()
    mock_inst.fit_on_train_and_test = MagicMock(return_value=None)
    mock_inst.transform.side_effect = _transform
    mock_inst.transform_with_word_augmentation = lambda xr, yr: (_transform(xr), yr)
    mock_safe_cls.return_value = mock_inst

    mock_train.side_effect = [
        {
            "accuracy_train": 0.5,
            "accuracy_val": 0.6,
            "accuracy_test": 0.55,
            "params": 10.0,
        },
        {
            "accuracy_train": 0.7,
            "accuracy_val": 0.85,
            "accuracy_test": 0.8,
            "params": 20.0,
        },
    ]

    x_train, y_train = _tiny_arrays()
    x_test, y_test = _tiny_arrays(n_per_class=2)

    best, rows, combo = run_single_experiment(
        "IntegrationToy",
        x_train,
        y_train,
        x_test,
        y_test,
        _minimal_base_params(),
        word_length=4,
        alphabet_size=7,
        embedding_dim=4,
        batch_size=4,
        learning_rate=0.01,
        verbose_safe=False,
        n_workers=1,
        train_val_split_fraction=0.25,
    )

    assert combo is not None and best is not None
    assert best["accuracy_val"] == pytest.approx(0.85)
    assert best["SAFE_embedding_epochs"] == 20
    assert len(rows) == 2
    chosen = [r for r in rows if r.get("chosen")]
    assert len(chosen) == 1
    assert chosen[0]["SAFE_embedding_epochs"] == 20
