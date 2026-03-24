"""Unit tests: train_growingnn with mocked growingnn backend (no real training)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from impl import kerne


@pytest.fixture
def mock_gnn(monkeypatch):
    """Replace kerne.gnn so trainer.train returns a toy model."""
    mock_model = MagicMock()
    mock_model.evaluate.side_effect = [0.91, 0.88, 0.77]
    mock_model.get_parametr_count.return_value = 100

    mock_gnn = MagicMock()
    mock_gnn.montecarlo_alg = object()
    mock_gnn.greedy_alg = object()
    mock_gnn.random_alg = object()
    mock_gnn.SimulationScheduler.PROGRESS_CHECK = 0
    mock_gnn.SimulationScheduler.CONSTANT = 1
    mock_gnn.LearningRateScheduler.PROGRESIVE = 0
    mock_gnn.Loss.multiclass_cross_entropy = object()
    mock_gnn.create_simulation_set_SAMLE = lambda *a, **k: None
    mock_gnn.trainer.train.return_value = mock_model

    monkeypatch.setattr(kerne, "gnn", mock_gnn)
    return mock_gnn, mock_model


def test_train_growingnn_returns_expected_keys(mock_gnn):
    _mock_gnn, mock_model = mock_gnn
    x_train = np.random.randn(8, 4).astype(np.float32)
    x_val = np.random.randn(3, 4).astype(np.float32)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    y_val = np.array([0, 1, 0], dtype=int)

    out = kerne.train_growingnn(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=[0, 1],
        input_size=4,
        hidden_size=8,
        output_size=2,
        input_shape=None,
        kernel_size=3,
        epochs=1,
        generations=1,
        simulation_set_size=4,
        simulation_time=1,
        simulation_epochs=1,
        simulation_scheduler_type="progress_check",
        optimization_factor=0.5,
        stopper_factor=0.9,
        learning_rate=0.01,
        activation_fun_name="Sigmoid",
        x_test_eval=np.random.randn(2, 4).astype(np.float32),
        y_test_eval=np.array([1, 0], dtype=int),
    )

    assert set(out.keys()) == {"accuracy_train", "accuracy_val", "accuracy_test", "params"}
    assert out["accuracy_val"] == pytest.approx(0.91)
    assert out["accuracy_train"] == pytest.approx(0.88)
    assert out["accuracy_test"] == pytest.approx(0.77)
    assert out["params"] == 100.0
    _mock_gnn.trainer.train.assert_called_once()
    assert mock_model.evaluate.call_count == 3


def test_label_encoder_reexport():
    assert kerne.LabelEncoder.__name__ == "LabelEncoder"
