"""
Reproduce the SAFE + GrowingNN paper experiments and compare against published numbers.

There was no remaining batch driver in this repo (README still mentioned
``growingnn_parameter_analysis_10/11.py``). This script is that driver:

1. Load the 14 UCR datasets used in the paper (or a subset).
2. Apply the archived GrowingNN / SAFE hyperparameters from ``results/``.
3. Call ``impl.pipeline.run_single_experiment`` (SAFE embedding search + GrowingNN).
4. Write new rows to ``results_rerun/`` (gitignored). Archived paper CSVs in ``results/`` are never overwritten.
5. Print a comparison vs paper Table 1 / Table 2 and the archived run.

Run from the repository root::

    python experiments/run_paper_experiments.py --compare-only
    python experiments/run_paper_experiments.py --list
    python experiments/run_paper_experiments.py --list --download
    python experiments/run_paper_experiments.py --suite table1 --quick
    python experiments/run_paper_experiments.py --suite paper --n-workers 4

A **full** run trains GrowingNN once per embedding combo (308 combos in ``impl/config.py``)
for every dataset. That is the original paper procedure and is very slow on CPU.
Use ``--quick`` first to check that data + dependencies work.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from impl.config import iter_embedding_param_combos

RESULTS_DIR = _REPO_ROOT / "results"
DEFAULT_RERUN_DIR = _REPO_ROOT / "results_rerun"
DEFAULT_DATA_ROOT = _REPO_ROOT / "data"
LEGACY_DATA_ROOT = _REPO_ROOT / "example" / "data"
ARCHIVED_RESULTS_CSV = RESULTS_DIR / "parameter_analysis_results.csv"
META_CSV = RESULTS_DIR / "per_dataset_meta_parameters.csv"
BASE_GRID_CSV = RESULTS_DIR / "base_parameter_grid.csv"
DEFAULT_OUTPUT_CSV_NAME = "paper_results.csv"

# Paper Table 1 lists 12 datasets (called "14" in the text). Table 2 lists 14.
TABLE1_DATASETS = (
    "Coffee",
    "Computers",
    "DiatomSizeReduction",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG5000",
    "GunPoint",
    "MiddlePhalanxOutlineCorrect",
    "OliveOil",
    "Plane",
    "Strawberry",
    "ToeSegmentation2",
)

TABLE2_DATASETS = (
    "Coffee",
    "Computers",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG5000",
    "GunPoint",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "OliveOil",
    "Plane",
    "Strawberry",
    "ToeSegmentation2",
)

# Extra datasets present in the archived CSV but not in paper tables.
ARCHIVED_ONLY_DATASETS = ("BeetleFly", "BirdChicken")

SUITES = {
    "table1": TABLE1_DATASETS,
    "paper": TABLE2_DATASETS,
    "table2": TABLE2_DATASETS,
    "archived": TABLE2_DATASETS + ARCHIVED_ONLY_DATASETS,
}

# Paper Table 1, embedding+GrowingNN test accuracy (%). Missing for the two
# extra Table 2 datasets.
PAPER_TABLE1_TEST_PCT = {
    "Coffee": 92.9,
    "Computers": 67.0,
    "DiatomSizeReduction": 77.4,
    "DistalPhalanxTW": 64.1,
    "Earthquakes": 74.8,
    "ECG5000": 91.6,
    "GunPoint": 77.3,
    "MiddlePhalanxOutlineCorrect": 57.2,
    "OliveOil": 83.0,
    "Plane": 92.3,
    "Strawberry": 92.4,
    "ToeSegmentation2": 75.3,
}

PAPER_TABLE2_PARAMS = {
    "Coffee": 195584,
    "Computers": 293888,
    "DiatomSizeReduction": 804864,
    "DistalPhalanxOutlineAgeGroup": 1295872,
    "DistalPhalanxTW": 1660,
    "Earthquakes": 1545,
    "ECG5000": 379392,
    "GunPoint": 152576,
    "MiddlePhalanxOutlineAgeGroup": 8313,
    "MiddlePhalanxOutlineCorrect": 1926,
    "OliveOil": 250880,
    "Plane": 593408,
    "Strawberry": 476160,
    "ToeSegmentation2": 234496,
}

# Working per-dataset zips. Replace {name} with Coffee, GunPoint, ...
# Literal <Name> 404s; ClassificationDownloads/ and Downloads/ currently return HTML.
DOWNLOAD_URL_TEMPLATES = (
    "https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip",
)

RESULT_COLUMNS = [
    "dataset",
    "accuracy_train",
    "accuracy_val",
    "accuracy_test",
    "params",
    "error",
    "epochs",
    "generations",
    "hidden_size",
    "batch_size",
    "simulation_set_size",
    "simulation_time",
    "simulation_epochs",
    "simulation_scheduler_type",
    "ACTIVATION_FUN",
    "convolution",
    "learning_rate",
    "optimization_factor",
    "augmentation_mode",
    "SAFE_word_length",
    "SAFE_ALPHABET_SIZE",
    "SAFE_embedding_dim",
    "TRAIN_VAL_SPLIT_FRACTION",
    "STOPPER_TARGET_ACCURACY",
    "SAFE_embedding_epochs",
    "SAFE_embedding_method",
    "SAFE_window_size",
    "SAFE_window_overlap_fraction",
    "SAFE_stride",
    "SAFE_fasttext_max_ngram",
    "SAFE_doc2vec_dm",
]


# ---------------------------------------------------------------------------
# Paper / archived hyper-parameters
# ---------------------------------------------------------------------------

def _parse_base_grid(path: Path) -> dict[str, str]:
    grid: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grid[row["parameter"]] = row["values"]
    return grid


def load_paper_hparams() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return (shared GrowingNN/SAFE knobs, per-dataset SAFE/train knobs)."""
    grid = _parse_base_grid(BASE_GRID_CSV)

    def _as_bool(value: str) -> bool:
        return str(value).strip().lower() in {"true", "1", "yes"}

    shared = {
        "epochs": int(grid["epochs"]),
        "generations": int(grid["generations"]),
        "hidden_size": int(grid["hidden_size"]),
        "simulation_set_size": int(grid["simulation_set_size"]),
        "simulation_time": int(grid["simulation_time"]),
        "simulation_epochs": int(grid["simulation_epochs"]),
        "simulation_scheduler_type": grid["simulation_scheduler_type"],
        "ACTIVATION_FUN": grid["ACTIVATION_FUN"],
        "convolution": _as_bool(grid["convolution"]),
        "optimization_factor": float(grid["optimization_factor"]),
        "augmentation_mode": None if grid["augmentation_mode"] in {"none", ""} else grid["augmentation_mode"],
        "DATA_augmentation_factor_jitter": 0,
        "DATA_augmentation_factor_warp": 0,
        "DATA_augmentation_factor_rgw": 0,
        "DATA_MIN_SAMPLES_PER_CLASS": 1,
        "train_val_split_fraction": float(grid["TRAIN_VAL_SPLIT_FRACTION"]),
        "stopper_target_accuracy": float(grid["STOPPER_TARGET_ACCURACY"]),
    }

    per_dataset: dict[str, dict[str, Any]] = {}
    with META_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_dataset[row["dataset"]] = {
                "embedding_dim": int(row["SAFE_embedding_dim"]),
                "alphabet_size": int(row["SAX_alphabet_size"]),
                "word_length": int(row["SAFE_word_length"]),
                "batch_size": int(row["batch_size"]),
                "learning_rate": float(row["learning_rate"]),
            }
    return shared, per_dataset


def load_archived_results(path: Path = ARCHIVED_RESULTS_CSV) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[str(row["dataset"])] = row.to_dict()
    return out


# ---------------------------------------------------------------------------
# Data loading / download
# ---------------------------------------------------------------------------

def zscore_per_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = np.maximum(x.std(axis=1, keepdims=True), 1e-8)
    return ((x - mean) / std).astype(np.float32)


def _load_ucr_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 1:].astype(np.float32), data[:, 0].astype(np.int64)


def _load_ts_univariate(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    ys: list[str] = []
    in_data = False
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            if not in_data:
                continue
            if ":" not in line:
                raise ValueError(f"Unexpected .ts data line in {path}: {line[:80]}")
            values_part, label = line.rsplit(":", 1)
            values = [
                float("nan") if v in {"?", ""} else float(v)
                for v in values_part.split(",")
            ]
            xs.append(values)
            ys.append(label.strip())
    y_as_float = True
    labels: list[Any] = []
    for y in ys:
        try:
            labels.append(int(float(y)))
        except ValueError:
            y_as_float = False
            break
    if not y_as_float:
        uniq = {y: i for i, y in enumerate(sorted(set(ys)))}
        labels = [uniq[y] for y in ys]
    return np.asarray(xs, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _find_split_file(dataset_dir: Path, dataset_name: str, split: str) -> Path | None:
    split_u = split.upper()
    candidates = [
        dataset_dir / f"{dataset_name}_{split_u}.txt",
        dataset_dir / f"{dataset_name}_{split_u}.tsv",
        dataset_dir / f"{dataset_name}_{split_u}.ts",
        dataset_dir / dataset_name / f"{dataset_name}_{split_u}.txt",
        dataset_dir / dataset_name / f"{dataset_name}_{split_u}.tsv",
        dataset_dir / dataset_name / f"{dataset_name}_{split_u}.ts",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = list(dataset_dir.rglob(f"{dataset_name}_{split_u}.*"))
    for path in matches:
        if path.suffix.lower() in {".txt", ".tsv", ".ts"}:
            return path
    return None


def find_dataset_dir(dataset_name: str, data_root: Path) -> Path | None:
    direct = data_root / dataset_name
    if _find_split_file(direct, dataset_name, "train") and _find_split_file(direct, dataset_name, "test"):
        return direct
    if _find_split_file(data_root, dataset_name, "train") and _find_split_file(data_root, dataset_name, "test"):
        return data_root
    return None


def iter_data_roots(primary: Path) -> list[Path]:
    roots = [primary]
    if LEGACY_DATA_ROOT.resolve() != primary.resolve():
        roots.append(LEGACY_DATA_ROOT)
    return roots


def locate_dataset(dataset_name: str, data_root: Path) -> Path | None:
    """Find a dataset under data/ (downloads) or example/data/ (older local copies)."""
    for root in iter_data_roots(data_root):
        found = find_dataset_dir(dataset_name, root)
        if found is not None:
            return found
    return None


def load_dataset(dataset_name: str, data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_dir = locate_dataset(dataset_name, data_root)
    if dataset_dir is None:
        raise FileNotFoundError(
            f"Missing {dataset_name} under {data_root}. "
            "Re-run with --list --download, or copy UCR TRAIN/TEST files into "
            f"{data_root / dataset_name}/"
        )
    train_path = _find_split_file(dataset_dir, dataset_name, "train")
    test_path = _find_split_file(dataset_dir, dataset_name, "test")
    assert train_path is not None and test_path is not None

    def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
        if path.suffix.lower() == ".ts":
            return _load_ts_univariate(path)
        return _load_ucr_table(path)

    x_train, y_train = _load(train_path)
    x_test, y_test = _load(test_path)
    return x_train, y_train, x_test, y_test


def _http_get(url: str, timeout: int = 120) -> bytes:
    request = Request(url, headers={"User-Agent": "growingNN_and_SAFE/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if not payload.startswith(b"PK"):
        snippet = payload[:80].decode("ascii", errors="replace").replace("\n", " ")
        raise zipfile.BadZipFile(f"not a zip (got {len(payload)} bytes starting {snippet!r})")
    return payload


def download_dataset(dataset_name: str, data_root: Path) -> Path:
    dest = data_root / dataset_name
    dest.mkdir(parents=True, exist_ok=True)
    logging.info("Saving %s under %s", dataset_name, dest)
    last_error: Exception | None = None
    for template in DOWNLOAD_URL_TEMPLATES:
        url = template.format(name=dataset_name)
        logging.info("Downloading %s", url)
        try:
            payload = _http_get(url)
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                zf.extractall(dest)
            if _find_split_file(dest, dataset_name, "train") is None:
                raise FileNotFoundError(f"Zip from {url} had no TRAIN split for {dataset_name}")
            logging.info("Saved %s to %s", dataset_name, dest)
            return dest
        except (URLError, TimeoutError, zipfile.BadZipFile, FileNotFoundError, OSError) as exc:
            last_error = exc
            logging.warning("Download failed for %s: %s", url, exc)
    raise RuntimeError(
        f"Could not download {dataset_name}. Last error: {last_error}. "
        "Download the UCR 2018 archive manually from "
        "https://www.timeseriesclassification.com/ and copy the dataset folder "
        f"to {dest}"
    )


def ensure_dataset(dataset_name: str, data_root: Path, download: bool) -> None:
    if locate_dataset(dataset_name, data_root) is not None:
        return
    if not download:
        raise FileNotFoundError(
            f"{dataset_name} not found in {data_root}. Re-run without --no-download, "
            f"or place {dataset_name}_TRAIN.txt and {dataset_name}_TEST.txt under "
            f"{data_root / dataset_name}/"
        )
    download_dataset(dataset_name, data_root)


def prepare_datasets(dataset_names: tuple[str, ...], data_root: Path, download: bool) -> list[str]:
    """Make sure every dataset is on disk. Return names that are still missing."""
    missing: list[str] = []
    for name in dataset_names:
        found = locate_dataset(name, data_root)
        if found is not None:
            logging.info("Data ready: %s (%s)", name, found)
            continue
        if not download:
            logging.error("Missing %s (download disabled)", name)
            missing.append(name)
            continue
        try:
            download_dataset(name, data_root)
        except Exception:
            logging.exception("Failed to download %s", name)
            missing.append(name)
    return missing


def drop_error_rows(path: Path) -> int:
    """Remove failed rows from a previous run so those datasets can be retried cleanly."""
    if not path.is_file():
        return 0
    df = pd.read_csv(path)
    if df.empty or "error" not in df.columns:
        return 0
    err = df["error"].fillna("").astype(str).str.strip()
    bad = err.ne("") & err.ne("nan")
    n_bad = int(bad.sum())
    if n_bad == 0:
        return 0
    df.loc[~bad].to_csv(path, index=False)
    return n_bad


# ---------------------------------------------------------------------------
# Running one dataset
# ---------------------------------------------------------------------------

def _quick_embedding_combos() -> Iterable[tuple]:
    yield (20, "word2vec", 4, None, 0, 0.0)
    yield (30, "word2vec", 5, None, 0, 0.25)


def _limit_combos(max_combos: int | None):
    def _gen():
        for i, combo in enumerate(iter_embedding_param_combos()):
            if max_combos is not None and i >= max_combos:
                break
            yield combo

    return _gen


def apply_quick_hparams(shared: dict[str, Any]) -> dict[str, Any]:
    """Smaller GrowingNN settings so a smoke run finishes in minutes, not days."""
    out = dict(shared)
    out.update(
        {
            "epochs": 80,
            "generations": 5,
            "hidden_size": 256,
            "simulation_set_size": 60,
            "simulation_time": 45,
            "simulation_epochs": 12,
        }
    )
    return out


def run_one_dataset(
    dataset_name: str,
    data_root: Path,
    shared: dict[str, Any],
    per_dataset: dict[str, dict[str, Any]],
    n_workers: int,
    download: bool,
) -> dict[str, Any]:
    if dataset_name not in per_dataset:
        raise KeyError(
            f"No per-dataset hyperparameters for {dataset_name} in {META_CSV}"
        )
    ensure_dataset(dataset_name, data_root, download)
    x_train, y_train, x_test, y_test = load_dataset(dataset_name, data_root)
    x_train = zscore_per_series(x_train)
    x_test = zscore_per_series(x_test)

    meta = per_dataset[dataset_name]
    base_params = {
        "epochs": shared["epochs"],
        "generations": shared["generations"],
        "hidden_size": shared["hidden_size"],
        "simulation_set_size": shared["simulation_set_size"],
        "simulation_time": shared["simulation_time"],
        "simulation_epochs": shared["simulation_epochs"],
        "simulation_scheduler_type": shared["simulation_scheduler_type"],
        "ACTIVATION_FUN": shared["ACTIVATION_FUN"],
        "convolution": shared["convolution"],
        "augmentation_mode": shared["augmentation_mode"],
        "DATA_augmentation_factor_jitter": shared["DATA_augmentation_factor_jitter"],
        "DATA_augmentation_factor_warp": shared["DATA_augmentation_factor_warp"],
        "DATA_augmentation_factor_rgw": shared["DATA_augmentation_factor_rgw"],
        "DATA_MIN_SAMPLES_PER_CLASS": shared["DATA_MIN_SAMPLES_PER_CLASS"],
        "optimization_factor": shared["optimization_factor"],
        "STOPPER_TARGET_ACCURACY": shared["stopper_target_accuracy"],
    }

    logging.info(
        "Running %s: train %s, test %s, length %d, classes %s | "
        "word_length=%s alphabet=%s emb_dim=%s batch=%s lr=%s",
        dataset_name,
        x_train.shape,
        x_test.shape,
        x_train.shape[1],
        np.unique(np.concatenate([y_train, y_test])),
        meta["word_length"],
        meta["alphabet_size"],
        meta["embedding_dim"],
        meta["batch_size"],
        meta["learning_rate"],
    )

    from impl.pipeline import run_single_experiment

    best, _log_rows, _combo = run_single_experiment(
        dataset_name,
        x_train,
        y_train,
        x_test,
        y_test,
        base_params,
        word_length=meta["word_length"],
        alphabet_size=meta["alphabet_size"],
        embedding_dim=meta["embedding_dim"],
        batch_size=meta["batch_size"],
        learning_rate=meta["learning_rate"],
        verbose_safe=False,
        n_workers=n_workers,
        train_val_split_fraction=shared["train_val_split_fraction"],
        stopper_target_accuracy=shared["stopper_target_accuracy"],
    )
    if best is None:
        raise RuntimeError(f"All embedding runs failed for {dataset_name}")

    row = {key: "" for key in RESULT_COLUMNS}
    row.update(
        {
            "dataset": dataset_name,
            "accuracy_train": best["accuracy_train"],
            "accuracy_val": best["accuracy_val"],
            "accuracy_test": best["accuracy_test"],
            "params": best["params"],
            "error": "",
            "epochs": shared["epochs"],
            "generations": shared["generations"],
            "hidden_size": shared["hidden_size"],
            "batch_size": meta["batch_size"],
            "simulation_set_size": shared["simulation_set_size"],
            "simulation_time": shared["simulation_time"],
            "simulation_epochs": shared["simulation_epochs"],
            "simulation_scheduler_type": shared["simulation_scheduler_type"],
            "ACTIVATION_FUN": shared["ACTIVATION_FUN"],
            "convolution": shared["convolution"],
            "learning_rate": meta["learning_rate"],
            "optimization_factor": shared["optimization_factor"],
            "augmentation_mode": shared["augmentation_mode"] or "none",
            "SAFE_word_length": meta["word_length"],
            "SAFE_ALPHABET_SIZE": meta["alphabet_size"],
            "SAFE_embedding_dim": meta["embedding_dim"],
            "TRAIN_VAL_SPLIT_FRACTION": shared["train_val_split_fraction"],
            "STOPPER_TARGET_ACCURACY": shared["stopper_target_accuracy"],
            "SAFE_embedding_epochs": best["SAFE_embedding_epochs"],
            "SAFE_embedding_method": best["SAFE_embedding_method"],
            "SAFE_window_size": best["SAFE_window_size"],
            "SAFE_window_overlap_fraction": best["SAFE_window_overlap_fraction"],
            "SAFE_stride": best["SAFE_stride"],
            "SAFE_fasttext_max_ngram": best["SAFE_fasttext_max_ngram"] or "",
            "SAFE_doc2vec_dm": best["SAFE_doc2vec_dm"],
        }
    )
    return row


def append_result_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


def already_finished(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        return set()
    ok = df
    if "error" in df.columns:
        err = df["error"].fillna("").astype(str).str.strip()
        ok = df[err.eq("") | err.eq("nan")]
    return set(ok["dataset"].astype(str))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "nan"}:
        return True
    return False


def _fmt_pct(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    return f"{100.0 * float(value):5.1f}"


def _fmt_num(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    return f"{int(round(float(value)))}"


def print_comparison(
    datasets: tuple[str, ...],
    rerun_rows: dict[str, dict[str, Any]],
    archived: dict[str, dict[str, Any]],
) -> None:
    header = (
        f"{'dataset':32} {'paper%':>7} {'archived%':>10} {'rerun%':>7} "
        f"{'d_paper':>8} {'paper_n':>10} {'rerun_n':>10}"
    )
    logging.info("Comparison (test accuracy % and parameter count)")
    logging.info(header)
    logging.info("-" * len(header))

    paper_vals: list[float] = []
    archived_vals: list[float] = []
    rerun_vals: list[float] = []

    for name in datasets:
        paper_pct = PAPER_TABLE1_TEST_PCT.get(name)
        archived_acc = archived.get(name, {}).get("accuracy_test")
        rerun_acc = rerun_rows.get(name, {}).get("accuracy_test")
        paper_n = PAPER_TABLE2_PARAMS.get(name)
        rerun_n = rerun_rows.get(name, {}).get("params")

        delta = "n/a"
        if paper_pct is not None and not _is_missing(rerun_acc):
            delta = f"{100.0 * float(rerun_acc) - paper_pct:+6.1f}"

        logging.info(
            "%-32s %7s %10s %7s %8s %10s %10s",
            name,
            f"{paper_pct:5.1f}" if paper_pct is not None else "n/a",
            _fmt_pct(archived_acc),
            _fmt_pct(rerun_acc),
            delta,
            _fmt_num(paper_n),
            _fmt_num(rerun_n),
        )
        if paper_pct is not None:
            paper_vals.append(paper_pct / 100.0)
        if not _is_missing(archived_acc):
            archived_vals.append(float(archived_acc))
        if not _is_missing(rerun_acc):
            rerun_vals.append(float(rerun_acc))

    logging.info("-" * len(header))
    logging.info(
        "%-32s %7s %10s %7s",
        "mean (available rows)",
        f"{100.0 * float(np.mean(paper_vals)):5.1f}" if paper_vals else "n/a",
        _fmt_pct(float(np.mean(archived_vals))) if archived_vals else "n/a",
        _fmt_pct(float(np.mean(rerun_vals))) if rerun_vals else "n/a",
    )


def print_suite_status(datasets: tuple[str, ...], data_root: Path, output_csv: Path) -> None:
    done = already_finished(output_csv)
    logging.info("Suite status  data_root=%s  output=%s", data_root, output_csv)
    for name in datasets:
        data_ok = locate_dataset(name, data_root) is not None
        logging.info(
            "  %-32s data=%-3s rerun=%s",
            name,
            "yes" if data_ok else "NO",
            "done" if name in done else "pending",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun SAFE + GrowingNN paper experiments and compare results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Suites:\n"
            "  table1    12 datasets in paper Table 1 (accuracy)\n"
            "  paper     14 datasets in paper Table 2 (default; paper claims 14)\n"
            "  archived  16 datasets from results/parameter_analysis_results.csv\n"
        ),
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="paper",
        help="Which dataset list to run (default: paper = 14 Table 2 datasets).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME",
        help="Override suite with explicit dataset names.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Folder of UCR datasets (default: data/, gitignored).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RERUN_DIR,
        help="Folder for this rerun (default: results_rerun/). "
        "Does not modify the archived paper files in results/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV to append rerun rows to (default: <results-dir>/paper_results.csv).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel embedding-search workers (default: 4).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing UCR datasets from "
        "https://www.timeseriesclassification.com/aeon-toolkit/<Dataset>.zip "
        "(e.g. Coffee.zip). Use with --list to fetch without training.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing datasets.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: 2 embedding combos and smaller GrowingNN settings.",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        metavar="N",
        help="Cap the embedding grid (paper GrowingNN settings still used).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip datasets already in --output (default: true).",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Do not train; only print paper vs archived vs --output.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List suite datasets. Combine with --download to fetch missing files and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    datasets = tuple(args.datasets) if args.datasets else SUITES[args.suite]
    shared, per_dataset = load_paper_hparams()
    archived = load_archived_results()
    results_dir = args.results_dir if args.results_dir.is_absolute() else _REPO_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        output_csv = results_dir / DEFAULT_OUTPUT_CSV_NAME
    else:
        output_csv = args.output if args.output.is_absolute() else _REPO_ROOT / args.output
    data_root = args.data_root if args.data_root.is_absolute() else _REPO_ROOT / args.data_root
    logging.info("Archived paper CSVs (read-only): %s", RESULTS_DIR)
    logging.info("This rerun writes to: %s", output_csv)

    n_full_combos = len(list(iter_embedding_param_combos()))
    logging.info("Embedding grid size in impl/config.py: %d combinations", n_full_combos)
    logging.info(
        "Paper GrowingNN settings: epochs=%s generations=%s hidden=%s "
        "sim_set=%s val_frac=%s stopper=%s activation=%s opt_factor=%s",
        shared["epochs"],
        shared["generations"],
        shared["hidden_size"],
        shared["simulation_set_size"],
        shared["train_val_split_fraction"],
        shared["stopper_target_accuracy"],
        shared["ACTIVATION_FUN"],
        shared["optimization_factor"],
    )

    if args.list:
        want_download = bool(args.download) and not args.no_download
        if want_download:
            logging.info(
                "Downloading missing datasets into %s from "
                "https://www.timeseriesclassification.com/aeon-toolkit/<Dataset>.zip",
                data_root,
            )
            still_missing = prepare_datasets(datasets, data_root, download=True)
            if still_missing:
                logging.error("Still missing: %s", ", ".join(still_missing))
                print_suite_status(datasets, data_root, output_csv)
                return 1
        print_suite_status(datasets, data_root, output_csv)
        if not want_download:
            missing_now = [n for n in datasets if locate_dataset(n, data_root) is None]
            if missing_now:
                logging.info(
                    "%d dataset(s) missing. Fetch with: python experiments/run_paper_experiments.py --list --download",
                    len(missing_now),
                )
        return 0

    rerun_now = load_archived_results(output_csv) if output_csv.is_file() else {}
    if args.compare_only:
        print_comparison(datasets, rerun_now, archived)
        return 0

    if args.quick:
        shared = apply_quick_hparams(shared)
        import impl.pipeline as pipeline_mod

        pipeline_mod.iter_embedding_param_combos = _quick_embedding_combos
        logging.warning(
            "QUICK mode: 2 embedding combos and reduced GrowingNN. "
            "Do not compare these numbers to the paper. Re-run without --quick for that."
        )
    elif args.max_combos is not None:
        import impl.pipeline as pipeline_mod

        pipeline_mod.iter_embedding_param_combos = _limit_combos(args.max_combos)
        logging.warning(
            "Embedding grid capped at %d combos (full grid is %d).",
            args.max_combos,
            n_full_combos,
        )
    else:
        logging.warning(
            "FULL paper run: %d embedding combos x %d datasets. "
            "Each combo trains GrowingNN (epochs=%s, generations=%s). "
            "Expect a long CPU job. Start with --quick if you have not run this yet.",
            n_full_combos,
            len(datasets),
            shared["epochs"],
            shared["generations"],
        )

    if args.dry_run:
        print_suite_status(datasets, data_root, output_csv)
        logging.info("Dry run: not training.")
        return 0

    n_dropped = drop_error_rows(output_csv)
    if n_dropped:
        logging.info("Removed %d failed row(s) from %s so they can be retried", n_dropped, output_csv)

    pending = [name for name in datasets if name not in (already_finished(output_csv) if args.resume else set())]
    still_missing = prepare_datasets(tuple(pending), data_root, download=not args.no_download)
    if still_missing:
        logging.error(
            "Missing datasets: %s. Copy UCR TRAIN/TEST files into %s/<Name>/ "
            "or re-run without --no-download.",
            ", ".join(still_missing),
            data_root,
        )
        return 1

    done = already_finished(output_csv) if args.resume else set()
    for name in datasets:
        if name in done:
            logging.info("Skipping %s (already in %s)", name, output_csv)
            continue
        try:
            row = run_one_dataset(
                name,
                data_root,
                shared,
                per_dataset,
                n_workers=args.n_workers,
                download=not args.no_download,
            )
        except Exception as exc:
            logging.exception("Dataset %s failed", name)
            row = {key: "" for key in RESULT_COLUMNS}
            row.update({"dataset": name, "error": str(exc)})
        append_result_row(output_csv, row)
        logging.info(
            "Wrote %s  test_acc=%s  params=%s  error=%s",
            name,
            row.get("accuracy_test"),
            row.get("params"),
            row.get("error"),
        )

    rerun_now = load_archived_results(output_csv) if output_csv.is_file() else {}
    print_comparison(datasets, rerun_now, archived)
    logging.info("Results CSV: %s", output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
