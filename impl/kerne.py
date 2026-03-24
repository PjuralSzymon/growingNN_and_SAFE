"""
SAFEgrowingNN-internal kerne module: LabelEncoder and train_growingnn for GrowingNN training.

Duplicated from the repo's kerne.py so that SAFEgrowingNN can work as a self-contained package.
Depends on the external 'growingnn' package (path set below or via GROWINGNN_PATH).
"""

import os
import sys
from typing import List, Dict, Optional

import numpy as np

# Re-export for pipeline.encode_labels and external use
from sklearn.preprocessing import LabelEncoder

# Ensure growingnn is on path (sibling repo or env)
def _ensure_growingnn_path():
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _candidates = [
        os.environ.get("GROWINGNN_PATH"),
        r"D:/repos/growingnn",
        os.path.join(_base, "..", "growingnn"),
        os.path.join(_base, "growingnn"),
    ]
    for p in _candidates:
        if p and os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
            return
    # last resort: append so import growingnn may still work if already on path
    if r"D:/repos/growingnn" not in sys.path:
        sys.path.append(r"D:/repos/growingnn")


_ensure_growingnn_path()
import growingnn as gnn  # type: ignore[import-untyped]  # external repo, path set above


# ===================== Config (minimal for train_growingnn) =====================

class Config:
    class GrowingNN:
        EPOCHS = 20
        GENERATIONS = 5
        BATCH_SIZE = 12
        SIMULATION_TIME = 60
        SIMULATION_EPOCHS = 20
        SIMULATION_SET_SIZE = 20
        LEARNING_RATE = 0.03
        LR_DECAY = 0.8
        SIMULATION_ALG = "montecarlo"
        SIMULATION_SCHEDULER_TYPE = "progress_check"
        OPTIMIZER = "adam"
        OPTIMIZATION_FACTOR = 0.5
        # Default for direct `train_growingnn` calls; pipeline passes `stopper_factor` explicitly.
        STOPPER_TARGET_ACCURACY = 0.85


# ===================== GrowingNN Wrapper =====================

def train_growingnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    labels: List[int],
    input_size: int,
    hidden_size: int,
    output_size: int,
    input_shape: tuple,
    kernel_size: int,
    epochs: int = Config.GrowingNN.EPOCHS,
    generations: int = Config.GrowingNN.GENERATIONS,
    model_name: str = "growingnn_model",
    batch_size: int = Config.GrowingNN.BATCH_SIZE,
    simulation_set_size: int = Config.GrowingNN.SIMULATION_SET_SIZE,
    simulation_alg: str = Config.GrowingNN.SIMULATION_ALG,
    optimizer: str = Config.GrowingNN.OPTIMIZER,
    simulation_time: int = Config.GrowingNN.SIMULATION_TIME,
    simulation_epochs: int = Config.GrowingNN.SIMULATION_EPOCHS,
    simulation_scheduler_type: str = Config.GrowingNN.SIMULATION_SCHEDULER_TYPE,
    optimization_factor: float = Config.GrowingNN.OPTIMIZATION_FACTOR,
    stopper_factor: float = Config.GrowingNN.STOPPER_TARGET_ACCURACY,
    learning_rate: float = Config.GrowingNN.LEARNING_RATE,
    activation_fun_name: str = 'Sigmoid',
    x_test_eval: Optional[np.ndarray] = None,
    y_test_eval: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Train a GrowingNN model. Expects (samples, features); transposes to (features, samples) when input_shape is None.
    """
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    if input_shape is None:
        x_train_gnn = x_train.T
        x_val_gnn = x_val.T
    else:
        x_train_gnn = x_train
        x_val_gnn = x_val

    simulation_alg_map = {
        "montecarlo": gnn.montecarlo_alg,
        "greedy": gnn.greedy_alg,
        "random": gnn.random_alg,
    }
    simulation_alg_obj = simulation_alg_map.get(simulation_alg.lower(), gnn.montecarlo_alg)

    scheduler_type_map = {
        "progress_check": gnn.SimulationScheduler.PROGRESS_CHECK,
        "constant": gnn.SimulationScheduler.CONSTANT,
    }
    scheduler_type = scheduler_type_map.get(
        simulation_scheduler_type.lower(),
        gnn.SimulationScheduler.PROGRESS_CHECK,
    )

    if optimizer.lower() == "adam":
        optimizer_obj = gnn.AdamOptimizer()
    else:
        optimizer_obj = gnn.SGDOptimizer()

    os.makedirs("./model_output/", exist_ok=True)

    trained_model = gnn.trainer.train(
        path="./model_output/",
        x_train=x_train_gnn,
        y_train=y_train,
        x_test=x_val_gnn,
        y_test=y_val,
        labels=list(range(output_size)),
        input_size=x_train_gnn.shape[0],
        hidden_size=hidden_size,
        output_size=output_size,
        model_name=model_name,
        epochs=epochs,
        generations=generations,
        input_shape=input_shape,
        kernel_size=kernel_size,
        deepth=2,
        batch_size=batch_size,
        simulation_set_size=simulation_set_size,
        simulation_alg=simulation_alg_obj,
        sim_set_generator=gnn.create_simulation_set_SAMLE,
        simulation_scheduler=gnn.SimulationScheduler(
            scheduler_type,
            simulation_time=simulation_time,
            simulation_epochs=simulation_epochs,
        ),
        lr_scheduler=gnn.LearningRateScheduler(
            gnn.LearningRateScheduler.PROGRESIVE,
            learning_rate,
            Config.GrowingNN.LR_DECAY,
        ),
        loss_function=gnn.Loss.multiclass_cross_entropy,
        optimizer=optimizer_obj,
        input_paths=1,
        simulation_score=gnn.Simulation_score(weight_countW=optimization_factor),
        stopper=gnn.AccuracyStopper(target_accuracy=stopper_factor),
        activation_fun=gnn.Activations.getByName(activation_fun_name),
    )

    accuracy_val = trained_model.evaluate(x_val_gnn, y_val)
    accuracy_train = trained_model.evaluate(x_train_gnn, y_train)
    params = trained_model.get_parametr_count()
    out: Dict[str, float] = {
        "accuracy_train": float(accuracy_train),
        "accuracy_val": float(accuracy_val),
        "params": float(params),
    }
    if x_test_eval is not None and y_test_eval is not None:
        x_test_eval = x_test_eval.astype(np.float32)
        y_test_eval = y_test_eval.astype(int)
        if input_shape is None:
            x_test_gnn = x_test_eval.T
        else:
            x_test_gnn = x_test_eval
        out["accuracy_test"] = float(trained_model.evaluate(x_test_gnn, y_test_eval))
    else:
        out["accuracy_test"] = float("nan")
    return out
