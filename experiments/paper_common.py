"""
Shared helpers for paper / adaptive experiment runners.

Used by run_adaptive_experiments.py. The exhaustive paper runner may keep its own
copies; this module is the canonical shared surface for new experiments.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "results"
DEFAULT_DATA_ROOT = _REPO_ROOT / "data"
LEGACY_DATA_ROOT = _REPO_ROOT / "example" / "data"
ARCHIVED_RESULTS_CSV = RESULTS_DIR / "parameter_analysis_results.csv"
META_CSV = RESULTS_DIR / "per_dataset_meta_parameters.csv"
BASE_GRID_CSV = RESULTS_DIR / "base_parameter_grid.csv"
DEFAULT_OUTPUT_CSV_NAME = "paper_results.csv"

# Paper Table 2 / main suite (14 datasets).
PAPER_DATASETS = (
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


def paper_target_accuracy(dataset_name: str, archived: dict[str, dict[str, Any]] | None = None) -> float | None:
    """Table 1 test accuracy as fraction in [0,1], else archived test, else None."""
    if dataset_name in PAPER_TABLE1_TEST_PCT:
        return PAPER_TABLE1_TEST_PCT[dataset_name] / 100.0
    if archived and dataset_name in archived:
        acc = archived[dataset_name].get("accuracy_test")
        if acc is not None and not (isinstance(acc, float) and np.isnan(acc)):
            return float(acc)
    return None


def _parse_base_grid(path: Path) -> dict[str, str]:
    grid: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grid[row["parameter"]] = row["values"]
    return grid


def load_paper_hparams() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
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
    labels: list[Any] = []
    y_as_float = True
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


def locate_dataset(dataset_name: str, data_root: Path) -> Path | None:
    roots = [data_root]
    if LEGACY_DATA_ROOT.resolve() != data_root.resolve():
        roots.append(LEGACY_DATA_ROOT)
    for root in roots:
        found = find_dataset_dir(dataset_name, root)
        if found is not None:
            return found
    return None


def load_dataset(dataset_name: str, data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_dir = locate_dataset(dataset_name, data_root)
    if dataset_dir is None:
        raise FileNotFoundError(
            f"Missing {dataset_name} under {data_root}. "
            f"Pass --download or copy UCR files into {data_root / dataset_name}/"
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
    raise RuntimeError(f"Could not download {dataset_name}. Last error: {last_error}")


def prepare_datasets(dataset_names: tuple[str, ...], data_root: Path, download: bool) -> list[str]:
    missing: list[str] = []
    for name in dataset_names:
        if locate_dataset(name, data_root) is not None:
            logging.info("Data ready: %s", name)
            continue
        if not download:
            missing.append(name)
            continue
        try:
            download_dataset(name, data_root)
        except Exception:
            logging.exception("Failed to download %s", name)
            missing.append(name)
    return missing


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
    # also require non-empty accuracy_test when column exists
    if "accuracy_test" in ok.columns:
        acc = ok["accuracy_test"]
        mask = acc.notna() & (acc.astype(str).str.strip() != "")
        ok = ok.loc[mask]
    return set(ok["dataset"].astype(str))


def drop_error_rows(path: Path) -> int:
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


def build_base_params(shared: dict[str, Any]) -> dict[str, Any]:
    return {
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


def result_row_from_best(
    dataset_name: str,
    shared: dict[str, Any],
    meta: dict[str, Any],
    best: dict[str, Any],
) -> dict[str, Any]:
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
            "SAFE_fasttext_max_ngram": best.get("SAFE_fasttext_max_ngram") or "",
            "SAFE_doc2vec_dm": best["SAFE_doc2vec_dm"],
        }
    )
    return row
