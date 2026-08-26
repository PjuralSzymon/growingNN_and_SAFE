
from __future__ import annotations

import logging
import math
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from impl.config import (
    EMBEDDING_DOC2VEC_DM,
    EMBEDDING_EPOCHS,
    EMBEDDING_FASTTEXT_MAX_NGRAM,
    EMBEDDING_METHODS,
    EMBEDDING_WINDOW_OVERLAP_FRACTION,
    EMBEDDING_WINDOW_SIZES,
    iter_embedding_param_combos,
)

# Combo: (epochs, method, window, ngram|None, dm, overlap)
Combo = tuple[Any, ...]

AXIS_NAMES = ("epochs", "method", "window", "overlap")


def _combo_key(combo: Combo) -> tuple:
    epochs, method, window, ngram, dm, overlap = combo
    return (int(epochs), str(method), int(window), ngram if ngram is not None else None, int(dm), float(overlap))


def all_paper_combos() -> list[Combo]:
    return list(iter_embedding_param_combos())


def axis_value_spaces() -> dict[str, list[Any]]:
    """Values that appear on each axis in the paper grid (for grading / plots)."""
    epochs = list(dict.fromkeys([*EMBEDDING_EPOCHS, 200]))  # ppmi_svd uses 200
    return {
        "epochs": epochs,
        "method": list(EMBEDDING_METHODS),
        "window": list(EMBEDDING_WINDOW_SIZES),
        "overlap": list(EMBEDDING_WINDOW_OVERLAP_FRACTION),
    }


def _combo_axes(combo: Combo) -> dict[str, Any]:
    epochs, method, window, _ngram, _dm, overlap = combo
    return {
        "epochs": int(epochs),
        "method": str(method),
        "window": int(window),
        "overlap": float(overlap),
    }


@dataclass
class TrialRecord:
    iteration: int
    combo: Combo
    accuracy_train: float
    accuracy_val: float
    accuracy_test: float
    params: float
    best_val_so_far: float
    best_test_at_best_val: float
    probs: dict[str, dict[str, float]]
    grades: dict[str, dict[str, float]]


@dataclass
class SearchResult:
    best_row: dict[str, Any] | None
    best_combo: Combo | None
    trials: list[TrialRecord]
    stop_reason: str
    sampler_seed: int
    final_probs: dict[str, dict[str, float]]
    final_grades: dict[str, dict[str, float]]
    paper_target: float | None
    n_iters: int = 0


@dataclass
class AxisGrader:
    """Maintain per-axis grades in [0,1] and softmax sampling probabilities."""

    spaces: dict[str, list[Any]] = field(default_factory=axis_value_spaces)
    grades: dict[str, dict[str, float]] = field(default_factory=dict)
    tau: float = 0.15
    beta: float = 0.3  # EMA toward new raw mean

    def __post_init__(self) -> None:
        if not self.grades:
            self.grades = {
                axis: {self._key(v): 0.5 for v in values}
                for axis, values in self.spaces.items()
            }

    @staticmethod
    def _key(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    def probs(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for axis, gmap in self.grades.items():
            keys = list(gmap.keys())
            logits = [gmap[k] / max(self.tau, 1e-6) for k in keys]
            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            s = sum(exps) or 1.0
            out[axis] = {k: e / s for k, e in zip(keys, exps)}
        return out

    def update_from_trials(self, trials: list[TrialRecord]) -> None:
        """Recompute raw mean val per axis value (in [0,1]), EMA into grades."""
        buckets: dict[str, dict[str, list[float]]] = {
            axis: {self._key(v): [] for v in values} for axis, values in self.spaces.items()
        }
        for t in trials:
            if t.accuracy_val is None or (isinstance(t.accuracy_val, float) and np.isnan(t.accuracy_val)):
                continue
            axes = _combo_axes(t.combo)
            for axis, val in axes.items():
                k = self._key(val)
                if k not in buckets[axis]:
                    buckets[axis][k] = []
                    self.grades.setdefault(axis, {})[k] = 0.5
                # clamp val accuracy to [0,1]
                s = float(np.clip(t.accuracy_val, 0.0, 1.0))
                buckets[axis][k].append(s)

        for axis, by_val in buckets.items():
            for k, scores in by_val.items():
                if not scores:
                    continue
                raw = float(np.mean(scores))  # already in [0,1]
                prev = self.grades[axis].get(k, 0.5)
                self.grades[axis][k] = (1.0 - self.beta) * prev + self.beta * raw


def _sample_from_categorical(rng: np.random.Generator, probs: dict[str, float]) -> str:
    keys = list(probs.keys())
    p = np.array([probs[k] for k in keys], dtype=np.float64)
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= 0:
        p = np.ones(len(keys)) / len(keys)
    else:
        p = p / s
    idx = int(rng.choice(len(keys), p=p))
    return keys[idx]


def _parse_axis_value(axis: str, key: str) -> Any:
    if axis == "method":
        return key
    if axis == "overlap":
        return float(key)
    return int(float(key))


def sample_combo(
    rng: np.random.Generator,
    probs: dict[str, dict[str, float]],
    unevaluated: set[tuple],
    unevaluated_list: list[Combo],
    max_rejects: int = 200,
) -> Combo | None:
    """Sample a combo from axis probs; fall back to weighted pick among unevaluated."""
    if not unevaluated_list:
        return None

    key_to_combo = {_combo_key(c): c for c in unevaluated_list}

    for _ in range(max_rejects):
        # Independent axis draws — may produce invalid / evaluated combos
        epochs_k = _sample_from_categorical(rng, probs["epochs"])
        method_k = _sample_from_categorical(rng, probs["method"])
        window_k = _sample_from_categorical(rng, probs["window"])
        overlap_k = _sample_from_categorical(rng, probs["overlap"])
        epochs = _parse_axis_value("epochs", epochs_k)
        method = _parse_axis_value("method", method_k)
        window = _parse_axis_value("window", window_k)
        overlap = _parse_axis_value("overlap", overlap_k)

        # Match paper grid construction for ngram/dm
        if method == "ppmi_svd":
            if epochs != 200:
                continue
            ngram, dm = None, 0
        elif method == "fasttext":
            if epochs == 200:
                continue
            ngram, dm = EMBEDDING_FASTTEXT_MAX_NGRAM[0], 0
        else:
            if epochs == 200:
                continue
            ngram, dm = None, int(EMBEDDING_DOC2VEC_DM[0]) if method == "doc2vec" else 0

        cand = (epochs, method, window, ngram, dm, overlap)
        ck = _combo_key(cand)
        if ck in unevaluated:
            return key_to_combo[ck]

    # Weighted among remaining: weight ∝ product of axis probs
    weights = []
    for c in unevaluated_list:
        axes = _combo_axes(c)
        w = 1.0
        for axis, val in axes.items():
            k = AxisGrader._key(val)
            w *= max(probs.get(axis, {}).get(k, 1e-6), 1e-12)
        weights.append(w)
    warr = np.asarray(weights, dtype=np.float64)
    s = warr.sum()
    if s <= 0:
        idx = int(rng.integers(0, len(unevaluated_list)))
    else:
        warr /= s
        idx = int(rng.choice(len(unevaluated_list), p=warr))
    return unevaluated_list[idx]


EvaluateFn = Callable[[Combo, int], dict[str, Any] | None]
# evaluate_fn(combo, iteration) -> dict with accuracy_train/val/test, params, SAFE_* fields
# or None on failure


def run_tpe_like_search(
    evaluate_fn: EvaluateFn,
    *,
    max_iters: int = 50,
    n_init: int = 5,
    paper_target: float | None = None,
    target_tol: float = 0.04,
    target_stop: bool = True,
    tau: float = 0.15,
    beta: float = 0.3,
    sampler_seed: int | None = None,
) -> SearchResult:
    pool = all_paper_combos()
    unevaluated_list = list(pool)
    unevaluated = {_combo_key(c) for c in unevaluated_list}

    if sampler_seed is None:
        sampler_seed = secrets.randbits(32)
    rng = np.random.default_rng(sampler_seed)
    logging.info("Adaptive sampler_seed=%s  pool=%d  max_iters=%d", sampler_seed, len(pool), max_iters)

    grader = AxisGrader(tau=tau, beta=beta)
    trials: list[TrialRecord] = []
    best_val = -1.0
    best_row: dict[str, Any] | None = None
    best_combo: Combo | None = None
    best_test_at_best_val = float("nan")
    stop_reason = "max_iters"

    for it in range(1, max_iters + 1):
        if not unevaluated_list:
            stop_reason = "pool_exhausted"
            break

        probs = grader.probs()
        if it <= n_init:
            idx = int(rng.integers(0, len(unevaluated_list)))
            combo = unevaluated_list[idx]
        else:
            combo = sample_combo(rng, probs, unevaluated, unevaluated_list)
            if combo is None:
                stop_reason = "pool_exhausted"
                break

        ck = _combo_key(combo)
        unevaluated.discard(ck)
        unevaluated_list = [c for c in unevaluated_list if _combo_key(c) != ck]

        logging.info(
            "  [adaptive %d/%d] epochs=%s method=%s window=%s overlap=%s",
            it,
            max_iters,
            combo[0],
            combo[1],
            combo[2],
            combo[5],
        )
        row = evaluate_fn(combo, it)
        if row is None:
            logging.warning("  [adaptive %d/%d] evaluation failed; continuing", it, max_iters)
            # still record a failed-ish trial with nan for grading skip
            trial = TrialRecord(
                iteration=it,
                combo=combo,
                accuracy_train=float("nan"),
                accuracy_val=float("nan"),
                accuracy_test=float("nan"),
                params=float("nan"),
                best_val_so_far=best_val if best_val >= 0 else float("nan"),
                best_test_at_best_val=best_test_at_best_val,
                probs={a: dict(p) for a, p in probs.items()},
                grades={a: dict(g) for a, g in grader.grades.items()},
            )
            trials.append(trial)
            continue

        val = float(row["accuracy_val"])
        test = float(row["accuracy_test"])
        if val > best_val:
            best_val = val
            best_row = row
            best_combo = combo
            best_test_at_best_val = test

        trial = TrialRecord(
            iteration=it,
            combo=combo,
            accuracy_train=float(row["accuracy_train"]),
            accuracy_val=val,
            accuracy_test=test,
            params=float(row.get("params", float("nan"))),
            best_val_so_far=best_val,
            best_test_at_best_val=best_test_at_best_val,
            probs={a: dict(p) for a, p in grader.probs().items()},
            grades={a: dict(g) for a, g in grader.grades.items()},
        )
        # append before update so grades use this trial
        trials.append(trial)
        grader.update_from_trials(trials)
        # refresh stored probs/grades on last trial after update
        trial.probs = {a: dict(p) for a, p in grader.probs().items()}
        trial.grades = {a: dict(g) for a, g in grader.grades.items()}

        logging.info(
            "  [adaptive %d/%d] val=%.4f test=%.4f best_val=%.4f",
            it,
            max_iters,
            val,
            test,
            best_val,
        )

        if target_stop and paper_target is not None and not np.isnan(test):
            if test >= paper_target - target_tol:
                stop_reason = "target_hit"
                logging.info(
                    "  Target hit: test=%.4f >= paper_target=%.4f - tol=%.4f",
                    test,
                    paper_target,
                    target_tol,
                )
                break

    return SearchResult(
        best_row=best_row,
        best_combo=best_combo,
        trials=trials,
        stop_reason=stop_reason,
        sampler_seed=sampler_seed,
        final_probs=grader.probs(),
        final_grades={a: dict(g) for a, g in grader.grades.items()},
        paper_target=paper_target,
        n_iters=len(trials),
    )
