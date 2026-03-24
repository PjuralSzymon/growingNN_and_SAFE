"""
Train SAFE + GrowingNN on UCR datasets under example/data/<DatasetName>/.

Shipped examples:
  - Earthquakes
  - DistalPhalanxOutlineAgeGroup

Default uses a small embedding grid unless --full (very slow).

Run from repository root:

    python example/run_example_on_data.py
    python example/run_example_on_data.py --dataset DistalPhalanxOutlineAgeGroup
    python example/run_example_on_data.py --dataset Earthquakes --full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from impl.pipeline import run_single_experiment

# Datasets with example/data/<name>/ pre-populated in this repo (extend as you add folders).
DATASETS_SHIPPED = (
    "Earthquakes",
    "DistalPhalanxOutlineAgeGroup",
)


def _find_ucr_data_dir(dataset_name: str) -> Path:
    base = Path(__file__).resolve().parent
    candidates = [
        base / "data" / dataset_name,
        base / dataset_name,
    ]
    train_name = f"{dataset_name}_TRAIN.txt"
    test_name = f"{dataset_name}_TEST.txt"
    for p in candidates:
        train, test = p / train_name, p / test_name
        if train.is_file() and test.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find {train_name} and {test_name}. "
        f"Tried: {[str(c / train_name) for c in candidates]}"
    )


def load_ucr_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """UCR rows: first column = class label, remaining columns = time series."""
    data = np.loadtxt(path, dtype=np.float64)
    y = data[:, 0].astype(np.int64)
    x = data[:, 1:].astype(np.float32)
    return x, y


def zscore_per_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.maximum(std, 1e-8)
    return ((x - mean) / std).astype(np.float32)


def _demo_embedding_combos():
    """Small grid for quick runs (epochs, method, window, ngram, doc2vec_dm, overlap)."""
    yield (20, "word2vec", 4, None, 0, 0.0)
    yield (30, "word2vec", 5, None, 0, 0.25)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UCR dataset: SAFE + GrowingNN (embedding search + GrowingNN training).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Earthquakes",
        metavar="NAME",
        help=f"UCR folder / file prefix under example/data/<NAME>/ (default: Earthquakes). "
        f"Shipped in repo: {', '.join(DATASETS_SHIPPED)}.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full embedding grid from config (many hours on CPU).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel embedding search workers (default: 1).",
    )
    args = parser.parse_args()

    dataset_name = args.dataset.strip()
    if not dataset_name:
        parser.error("--dataset must be non-empty")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    data_dir = _find_ucr_data_dir(dataset_name)
    x_train, y_train = load_ucr_txt(data_dir / f"{dataset_name}_TRAIN.txt")
    x_test, y_test = load_ucr_txt(data_dir / f"{dataset_name}_TEST.txt")

    x_train = zscore_per_series(x_train)
    x_test = zscore_per_series(x_test)

    logging.info(
        "Loaded %s: train %s, test %s, series length %d, classes %s",
        dataset_name,
        x_train.shape,
        x_test.shape,
        x_train.shape[1],
        np.unique(np.concatenate([y_train, y_test])),
    )

    base_params = {
        "epochs": 80,
        "generations": 5,
        "hidden_size": 256,
        "simulation_set_size": 60,
        "simulation_time": 45,
        "simulation_epochs": 12,
        "simulation_scheduler_type": "progress_check",
        "ACTIVATION_FUN": "Sigmoid",
        "convolution": False,
        "augmentation_mode": None,
        "DATA_augmentation_factor_jitter": 0,
        "DATA_augmentation_factor_warp": 0,
        "DATA_augmentation_factor_rgw": 0,
        "DATA_MIN_SAMPLES_PER_CLASS": 1,
        "optimization_factor": 0.5,
        "STOPPER_TARGET_ACCURACY": 0.95,
    }

    import impl.pipeline as pipeline_mod

    if not args.full:
        pipeline_mod.iter_embedding_param_combos = _demo_embedding_combos
        logging.info("Quick mode: 2 embedding combinations. Use --full for the full grid.")

    best, log_rows, combo = run_single_experiment(
        dataset_name,
        x_train,
        y_train,
        x_test,
        y_test,
        base_params,
        word_length=6,
        alphabet_size=7,
        embedding_dim=8,
        batch_size=8,
        learning_rate=1e-4,
        verbose_safe=False,
        n_workers=args.n_workers,
        train_val_split_fraction=0.25,
        stopper_target_accuracy=0.95,
    )

    if best is None:
        logging.error("All embedding runs failed.")
        sys.exit(1)

    logging.info("Best validation accuracy: %.4f", best["accuracy_val"])
    logging.info("Test accuracy: %.4f", best["accuracy_test"])
    logging.info("Best embedding combo: %s", combo)
    logging.info("Train accuracy: %.4f", best["accuracy_train"])


if __name__ == "__main__":
    main()
