from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.adaptive_search import run_tpe_like_search
from experiments.adaptive_viz import save_dataset_artifacts
from experiments.paper_common import (
    ARCHIVED_RESULTS_CSV,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_CSV_NAME,
    PAPER_DATASETS,
    PAPER_TABLE1_TEST_PCT,
    PAPER_TABLE2_PARAMS,
    RESULTS_DIR,
    append_result_row,
    already_finished,
    build_base_params,
    drop_error_rows,
    load_archived_results,
    load_dataset,
    load_paper_hparams,
    locate_dataset,
    paper_target_accuracy,
    prepare_datasets,
    result_row_from_best,
    zscore_per_series,
)
from impl.config import _word_extraction_stride, iter_embedding_param_combos
from impl.pipeline import _run_one_embedding, _split_train_val, _train_val_random_state

DEFAULT_ADAPTIVE_DIR = _REPO_ROOT / "results_adaptive"


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


def print_comparison(
    datasets: tuple[str, ...],
    rerun_rows: dict[str, dict[str, Any]],
    archived: dict[str, dict[str, Any]],
) -> None:
    header = (
        f"{'dataset':32} {'paper%':>7} {'archived%':>10} {'adapt%':>7} "
        f"{'d_paper':>8} {'paper_n':>10} {'adapt_n':>10}"
    )
    logging.info("Comparison (test accuracy % and parameter count)")
    logging.info(header)
    logging.info("-" * len(header))
    paper_vals: list[float] = []
    archived_vals: list[float] = []
    adapt_vals: list[float] = []
    for name in datasets:
        paper_pct = PAPER_TABLE1_TEST_PCT.get(name)
        archived_acc = archived.get(name, {}).get("accuracy_test")
        adapt_acc = rerun_rows.get(name, {}).get("accuracy_test")
        paper_n = PAPER_TABLE2_PARAMS.get(name)
        adapt_n = rerun_rows.get(name, {}).get("params")
        delta = "n/a"
        if paper_pct is not None and not _is_missing(adapt_acc):
            delta = f"{100.0 * float(adapt_acc) - paper_pct:+6.1f}"
        logging.info(
            "%-32s %7s %10s %7s %8s %10s %10s",
            name,
            f"{paper_pct:5.1f}" if paper_pct is not None else "n/a",
            _fmt_pct(archived_acc),
            _fmt_pct(adapt_acc),
            delta,
            f"{paper_n}" if paper_n is not None else "n/a",
            f"{int(round(float(adapt_n)))}" if not _is_missing(adapt_n) else "n/a",
        )
        if paper_pct is not None:
            paper_vals.append(paper_pct / 100.0)
        if not _is_missing(archived_acc):
            archived_vals.append(float(archived_acc))
        if not _is_missing(adapt_acc):
            adapt_vals.append(float(adapt_acc))
    logging.info("-" * len(header))
    logging.info(
        "%-32s %7s %10s %7s",
        "mean (available rows)",
        f"{100.0 * float(np.mean(paper_vals)):5.1f}" if paper_vals else "n/a",
        _fmt_pct(float(np.mean(archived_vals))) if archived_vals else "n/a",
        _fmt_pct(float(np.mean(adapt_vals))) if adapt_vals else "n/a",
    )


def print_suite_status(datasets: tuple[str, ...], data_root: Path, output_csv: Path) -> None:
    done = already_finished(output_csv)
    logging.info("Suite status  data_root=%s  output=%s", data_root, output_csv)
    for name in datasets:
        data_ok = locate_dataset(name, data_root) is not None
        logging.info(
            "  %-32s data=%-3s adaptive=%s",
            name,
            "yes" if data_ok else "NO",
            "done" if name in done else "pending",
        )


def run_one_dataset_adaptive(
    dataset_name: str,
    data_root: Path,
    shared: dict[str, Any],
    per_dataset: dict[str, dict[str, Any]],
    archived: dict[str, dict[str, Any]],
    dataset_out_dir: Path,
    *,
    max_iters: int,
    n_init: int,
    target_tol: float,
    target_stop: bool,
    sampler_seed: int | None,
    download: bool,
) -> dict[str, Any]:
    if dataset_name not in per_dataset:
        raise KeyError(f"No per-dataset hyperparameters for {dataset_name}")
    if locate_dataset(dataset_name, data_root) is None and not download:
        raise FileNotFoundError(f"{dataset_name} not found under {data_root}")
    if locate_dataset(dataset_name, data_root) is None:
        from experiments.paper_common import download_dataset

        download_dataset(dataset_name, data_root)

    x_train, y_train, x_test, y_test = load_dataset(dataset_name, data_root)
    x_train = zscore_per_series(x_train)
    x_test = zscore_per_series(x_test)

    meta = per_dataset[dataset_name]
    base_params = build_base_params(shared)
    tv_frac = float(shared["train_val_split_fraction"])
    stopper = float(shared["stopper_target_accuracy"])
    split_rs = _train_val_random_state(dataset_name)
    x_tr_raw, x_val_raw, y_tr, y_val = _split_train_val(x_train, y_train, tv_frac, split_rs)

    paper_target = paper_target_accuracy(dataset_name, archived)
    logging.info(
        "Adaptive %s: train=%s val=%s test=%s | paper_target=%s tol=%s max_iters=%s | "
        "GrowingNN stopper=%s (epochs=%s generations=%s are budgets)",
        dataset_name,
        x_tr_raw.shape,
        x_val_raw.shape,
        x_test.shape,
        paper_target,
        target_tol,
        max_iters,
        stopper,
        shared["epochs"],
        shared["generations"],
    )

    word_length = meta["word_length"]
    alphabet_size = meta["alphabet_size"]
    embedding_dim = meta["embedding_dim"]
    batch_size = meta["batch_size"]
    learning_rate = meta["learning_rate"]
    experiment_id = f"adaptive_{dataset_name}"

    def evaluate_fn(combo, iteration: int):
        # ``stopper`` = GrowingNN AccuracyStopper (train-loop early exit).
        # Meta-search early exit uses paper_target / target_tol after this returns.
        emb_idx = iteration - 1
        _idx, val_acc, _train_acc, result, combo_ret, log_row = _run_one_embedding(
            emb_idx,
            combo,
            dataset_name,
            experiment_id,
            word_length,
            alphabet_size,
            embedding_dim,
            base_params,
            x_tr_raw,
            y_tr,
            x_val_raw,
            y_val,
            x_test,
            y_test,
            tv_frac,
            stopper,
            batch_size,
            learning_rate,
            verbose_safe=False,
            mute_output=False,
        )
        if val_acc is None or result is None:
            return None
        emb_epochs, emb_method, emb_ws, emb_ngram, emb_dm, emb_overlap = combo_ret
        return {
            "accuracy_train": result.get("accuracy_train", log_row.get("accuracy_train", 0.0)),
            "accuracy_val": result.get("accuracy_val", val_acc),
            "accuracy_test": result.get("accuracy_test", log_row.get("accuracy_test", float("nan"))),
            "params": result.get("params", 0.0),
            "SAFE_embedding_epochs": emb_epochs,
            "SAFE_embedding_method": emb_method,
            "SAFE_window_size": emb_ws,
            "SAFE_window_overlap_fraction": emb_overlap,
            "SAFE_stride": _word_extraction_stride(word_length, emb_overlap),
            "SAFE_fasttext_max_ngram": emb_ngram,
            "SAFE_doc2vec_dm": emb_dm,
        }

    search = run_tpe_like_search(
        evaluate_fn,
        max_iters=max_iters,
        n_init=n_init,
        paper_target=paper_target,
        target_tol=target_tol,
        target_stop=target_stop,
        sampler_seed=sampler_seed,
    )
    save_dataset_artifacts(dataset_out_dir, dataset_name, search)

    if search.best_row is None:
        raise RuntimeError(f"All adaptive trials failed for {dataset_name}")

    row = result_row_from_best(dataset_name, shared, meta, search.best_row)
    logging.info(
        "Adaptive done %s: stop=%s iters=%d seed=%s best_val=%s best_test=%s",
        dataset_name,
        search.stop_reason,
        search.n_iters,
        search.sampler_seed,
        search.best_row.get("accuracy_val"),
        search.best_row.get("accuracy_test"),
    )
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Budgeted adaptive SAFE + GrowingNN search (separate from exhaustive paper runner).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="UCR datasets folder (default: data/).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_ADAPTIVE_DIR,
        help="Output folder (default: results_adaptive/).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary CSV (default: <results-dir>/paper_results.csv).",
    )
    p.add_argument("--max-iters", type=int, default=50, help="Max embedding trials per dataset (default: 50).")
    p.add_argument("--n-init", type=int, default=5, help="Uniform warm-up trials before weighted sampling.")
    p.add_argument(
        "--target-tol",
        type=float,
        default=0.04,
        help=(
            "Meta-search only: after a finished trial, stop sampling more combos if "
            "test >= paper_target - tol (default: 0.04). Does not stop GrowingNN mid-train; "
            "that uses STOPPER_TARGET_ACCURACY from the paper grid. See README_adaptive.md."
        ),
    )
    p.add_argument(
        "--no-target-stop",
        action="store_true",
        help="Never stop early on paper target; always run max-iters (or pool exhaust).",
    )
    p.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Optional fixed sampler seed (default: random each run; always logged).",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Kept for CLI symmetry; adaptive search is sequential (default: 1).",
    )
    p.add_argument("--download", action="store_true", help="Download missing UCR datasets.")
    p.add_argument("--no-download", action="store_true", help="Do not download missing datasets.")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip datasets already finished in --output (default: true).",
    )
    p.add_argument("--compare-only", action="store_true", help="Only print paper vs archived vs adaptive CSV.")
    p.add_argument("--list", action="store_true", help="List datasets / readiness; optional --download.")
    p.add_argument("--dry-run", action="store_true", help="Print plan and exit.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    datasets = PAPER_DATASETS
    shared, per_dataset = load_paper_hparams()
    archived = load_archived_results(ARCHIVED_RESULTS_CSV)

    results_dir = args.results_dir if args.results_dir.is_absolute() else _REPO_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        output_csv = results_dir / DEFAULT_OUTPUT_CSV_NAME
    else:
        output_csv = args.output if args.output.is_absolute() else _REPO_ROOT / args.output
    data_root = args.data_root if args.data_root.is_absolute() else _REPO_ROOT / args.data_root

    n_pool = len(list(iter_embedding_param_combos()))
    logging.info("Archived paper CSVs (read-only): %s", RESULTS_DIR)
    logging.info("Adaptive run writes to: %s", results_dir)
    logging.info("Embedding pool size: %d | max_iters=%d n_init=%d target_tol=%s", n_pool, args.max_iters, args.n_init, args.target_tol)
    if args.n_workers != 1:
        logging.warning("Adaptive search updates weights after each trial; using sequential evaluation (n_workers ignored).")

    if args.list:
        want_download = bool(args.download) and not args.no_download
        if want_download:
            still_missing = prepare_datasets(datasets, data_root, download=True)
            if still_missing:
                logging.error("Still missing: %s", ", ".join(still_missing))
                print_suite_status(datasets, data_root, output_csv)
                return 1
        print_suite_status(datasets, data_root, output_csv)
        return 0

    adapt_now = load_archived_results(output_csv) if output_csv.is_file() else {}
    if args.compare_only:
        print_comparison(datasets, adapt_now, archived)
        return 0

    if args.dry_run:
        print_suite_status(datasets, data_root, output_csv)
        logging.info("Dry run: not training.")
        return 0

    n_dropped = drop_error_rows(output_csv)
    if n_dropped:
        logging.info("Removed %d failed row(s) from %s", n_dropped, output_csv)

    done = already_finished(output_csv) if args.resume else set()
    pending = [n for n in datasets if n not in done]
    still_missing = prepare_datasets(tuple(pending), data_root, download=not args.no_download)
    if still_missing:
        logging.error("Missing datasets: %s", ", ".join(still_missing))
        return 1

    for name in datasets:
        if name in done:
            logging.info("Skipping %s (already in %s)", name, output_csv)
            continue
        ds_dir = results_dir / name
        try:
            row = run_one_dataset_adaptive(
                name,
                data_root,
                shared,
                per_dataset,
                archived,
                ds_dir,
                max_iters=args.max_iters,
                n_init=args.n_init,
                target_tol=args.target_tol,
                target_stop=not args.no_target_stop,
                sampler_seed=args.sampler_seed,
                download=not args.no_download,
            )
        except Exception as exc:
            logging.exception("Dataset %s failed", name)
            from experiments.paper_common import RESULT_COLUMNS

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

    adapt_now = load_archived_results(output_csv) if output_csv.is_file() else {}
    print_comparison(datasets, adapt_now, archived)
    logging.info("Results CSV: %s", output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
