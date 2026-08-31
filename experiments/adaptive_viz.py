"""Visualization for experiment search outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiments.adaptive_search import SearchResult, TrialRecord


def _combo_fields(combo: tuple) -> dict[str, Any]:
    epochs, method, window, ngram, dm, overlap = combo
    return {
        "SAFE_embedding_epochs": epochs,
        "SAFE_embedding_method": method,
        "SAFE_window_size": window,
        "SAFE_fasttext_max_ngram": ngram if ngram is not None else "",
        "SAFE_doc2vec_dm": dm,
        "SAFE_window_overlap_fraction": overlap,
    }


def save_trials_csv(path: Path, trials: list[TrialRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "iteration",
        "accuracy_train",
        "accuracy_val",
        "accuracy_test",
        "params",
        "best_val_so_far",
        "best_test_at_best_val",
        "SAFE_embedding_epochs",
        "SAFE_embedding_method",
        "SAFE_window_size",
        "SAFE_window_overlap_fraction",
        "SAFE_fasttext_max_ngram",
        "SAFE_doc2vec_dm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trials:
            row = {
                "iteration": t.iteration,
                "accuracy_train": t.accuracy_train,
                "accuracy_val": t.accuracy_val,
                "accuracy_test": t.accuracy_test,
                "params": t.params,
                "best_val_so_far": t.best_val_so_far,
                "best_test_at_best_val": t.best_test_at_best_val,
            }
            row.update(_combo_fields(t.combo))
            writer.writerow(row)


def save_search_state(path: Path, result: SearchResult, dataset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "sampler_seed": result.sampler_seed,
        "stop_reason": result.stop_reason,
        "n_iters": result.n_iters,
        "paper_target": result.paper_target,
        "final_probs": result.final_probs,
        "final_grades": result.final_grades,
        "best_combo": list(result.best_combo) if result.best_combo is not None else None,
        "best_val": None if result.best_row is None else result.best_row.get("accuracy_val"),
        "best_test": None if result.best_row is None else result.best_row.get("accuracy_test"),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def plot_best_vs_iteration(
    path: Path,
    trials: list[TrialRecord],
    *,
    dataset: str,
    paper_target: float | None,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [t.iteration for t in trials]
    best_val = [t.best_val_so_far for t in trials]
    val_at = [t.accuracy_val for t in trials]
    test_at = [t.accuracy_test for t in trials]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, best_val, marker="o", linewidth=2, label="best val so far")
    ax.plot(xs, val_at, marker="s", linewidth=1, alpha=0.75, label="this iter val")
    ax.plot(
        xs,
        test_at,
        marker="x",
        linewidth=1,
        alpha=0.45,
        label="this iter test (report only)",
    )
    if paper_target is not None:
        ax.axhline(
            paper_target,
            color="C3",
            linestyle="--",
            linewidth=1.5,
            label=f"paper target (val stop) {paper_target:.3f}",
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{dataset}: best validation vs iteration (stop on val vs paper)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_axis_probs(
    path: Path,
    probs: dict[str, float],
    *,
    dataset: str,
    axis: str,
    stop_reason: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    # sort keys naturally
    items = list(probs.items())

    def _sort_key(kv):
        k = kv[0]
        try:
            return (0, float(k))
        except ValueError:
            return (1, k)

    items.sort(key=_sort_key)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(labels)), values, color="C0", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("P(value)")
    ax.set_ylim(0.0, max(values + [0.01]) * 1.15)
    ax.set_title(f"{dataset}: final P({axis})  [{stop_reason}]")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_dataset_artifacts(
    out_dir: Path,
    dataset: str,
    result: SearchResult,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_trials_csv(out_dir / "trials.csv", result.trials)
    save_search_state(out_dir / "search_state.json", result, dataset)
    plot_best_vs_iteration(
        out_dir / "best_accuracy_vs_iteration.png",
        result.trials,
        dataset=dataset,
        paper_target=result.paper_target,
    )
    for axis, pmap in result.final_probs.items():
        plot_axis_probs(
            out_dir / f"prob_{axis}.png",
            pmap,
            dataset=dataset,
            axis=axis,
            stop_reason=result.stop_reason,
        )
