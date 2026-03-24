"""
SAFEgrowingNN pipeline: try all embedding combos -> SAFE transform -> GrowingNN train -> select best by validation accuracy.
Embedding search can be parallelized via n_workers / EMBEDDING_SEARCH_MAX_WORKERS.
"""

import hashlib
import logging
import os
import random
import sys
import threading
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from safe2 import SafeTransformer


class _ThreadMutingWriter:
    """Wraps a stream; writes from registered threads go to devnull, others to the real stream."""

    __slots__ = ("_real", "_muted_threads", "_devnull")

    def __init__(self, real_stream, muted_threads_set):
        self._real = real_stream
        self._muted_threads = muted_threads_set
        self._devnull = open(os.devnull, "w", encoding="utf-8")

    def write(self, data):
        if threading.current_thread() in self._muted_threads:
            self._devnull.write(data)
        else:
            self._real.write(data)

    def flush(self):
        if threading.current_thread() in self._muted_threads:
            self._devnull.flush()
        else:
            self._real.flush()

    def close(self):
        self._devnull.close()

from .config import (
    iter_embedding_param_combos,
    EMBEDDING_SEARCH_MAX_WORKERS,
    EMBEDDING_SEARCH_USE_PROCESSES,
    MUTE_VERBOSE_INSIDE_THREAD,
    DEFAULT_TRAIN_VAL_SPLIT_FRACTION,
    TRAIN_VAL_SPLIT_RANDOM_STATE,
    DEFAULT_STOPPER_TARGET_ACCURACY,
    resolve_embedding_workers,
    _word_extraction_stride,
)
from . import kerne


def encode_labels(y_train, y_test):
    """Encode labels to integers; fit on y_train, transform y_test."""
    le = kerne.LabelEncoder()
    return le.fit_transform(y_train), le.transform(y_test)


def _encode_labels_train_val_test(y_train, y_val, y_test):
    le = kerne.LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    return y_train_enc, le.transform(y_val), le.transform(y_test)


def _train_val_random_state(dataset_name: str) -> int:
    """Stable RNG seed for train/val split (reproducible across Python runs for the same dataset)."""
    digest = hashlib.md5(f"{dataset_name}:{TRAIN_VAL_SPLIT_RANDOM_STATE}".encode()).hexdigest()
    return int(digest[:8], 16)


def _allocate_val_per_class_hamilton(
    class_sizes: list[int],
    n_val_target: int,
    n_total: int,
) -> np.ndarray:
    """
    Hamilton / largest remainder: integer val counts per class summing to n_val_target, each
    proportional to class size, with val_c <= max(0, n_c - 1) (singletons get 0).

    Caller must ensure n_val_target <= sum(max(0, n_c - 1) for n_c in class_sizes).
    """
    k = len(class_sizes)
    ideals = np.zeros(k, dtype=float)
    cap = np.zeros(k, dtype=int)
    for i, n_c in enumerate(class_sizes):
        if n_c <= 1:
            ideals[i] = 0.0
            cap[i] = 0
        else:
            ideals[i] = n_c * n_val_target / n_total
            cap[i] = n_c - 1

    floors = np.minimum(np.floor(ideals).astype(int), cap)
    remainder = int(n_val_target - floors.sum())

    if remainder > 0:
        frac = ideals - np.floor(ideals)
        order = np.argsort(-frac)
        guard = 0
        while remainder > 0 and guard < n_val_target * k + k + 2:
            guard += 1
            moved = False
            for idx in order:
                if remainder <= 0:
                    break
                if floors[idx] < cap[idx]:
                    floors[idx] += 1
                    remainder -= 1
                    moved = True
            if not moved:
                for idx in range(k):
                    if remainder <= 0:
                        break
                    if floors[idx] < cap[idx]:
                        floors[idx] += 1
                        remainder -= 1
                        moved = True
                if not moved:
                    break

    if remainder < 0:
        order_big = np.argsort(floors)
        guard = 0
        while remainder < 0 and guard < n_val_target * k + k + 2:
            guard += 1
            moved = False
            for j in range(k - 1, -1, -1):
                if remainder >= 0:
                    break
                idx = int(order_big[j])
                if floors[idx] > 0:
                    floors[idx] -= 1
                    remainder += 1
                    moved = True
            if not moved:
                break

    return floors


def _split_train_val(x_train, y_train, val_fraction, random_state):
    """
    Stratified train/val split with a **globally balanced** validation size.

    Target validation size is ``round(n * val_fraction)`` (clamped to ``[1, n - 1]``). Validation
    slots are allocated across classes with the Hamilton (largest remainder) method so each
    class shares that total in proportion to its size—more equal overall split than independent
    per-class rounding, which can drift far from ``val_fraction``.

    For each class with at least two samples, val count is at most ``n_c - 1`` (at least one
    train sample). Singleton classes contribute only to train.

    If the requested val size exceeds what stratification allows (sum of per-class caps), the
    target is reduced to that cap so proportions stay valid. If every class is a singleton,
    falls back to a single random split of the requested proportion.
    """
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    n = len(y_train)
    if n < 2:
        raise ValueError(
            "Training set must have at least 2 samples for train/validation split."
        )
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be in (0, 1).")

    labels, inverse = np.unique(y_train, return_inverse=True)
    counts = np.bincount(inverse)
    class_sizes = [int(counts[i]) for i in range(len(labels))]
    max_strat_val = sum(max(0, nc - 1) for nc in class_sizes)

    if max_strat_val == 0:
        perm = rng.permutation(n)
        n_val = max(1, min(n - 1, int(round(n * val_fraction))))
        val_idx_arr = perm[:n_val].astype(np.intp)
        train_idx_arr = perm[n_val:].astype(np.intp)
        return (
            x_train[train_idx_arr],
            x_train[val_idx_arr],
            y_train[train_idx_arr],
            y_train[val_idx_arr],
        )

    n_val_target = max(1, min(n - 1, int(round(n * val_fraction))))
    n_val_target = min(n_val_target, max_strat_val)

    val_counts = _allocate_val_per_class_hamilton(class_sizes, n_val_target, n)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for li, lab in enumerate(labels):
        idx = np.where(y_train == lab)[0]
        rng.shuffle(idx)
        n_c = len(idx)
        nv = int(val_counts[li])
        nv = max(0, min(nv, n_c))
        if n_c >= 2:
            nv = min(nv, n_c - 1)
        val_idx.extend(idx[:nv].astype(int).tolist())
        train_idx.extend(idx[nv:].astype(int).tolist())

    train_idx_arr = np.asarray(train_idx, dtype=np.intp)
    val_idx_arr = np.asarray(val_idx, dtype=np.intp)

    return (
        x_train[train_idx_arr],
        x_train[val_idx_arr],
        y_train[train_idx_arr],
        y_train[val_idx_arr],
    )


def _run_one_embedding(
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
    x_test_raw,
    y_test,
    train_val_split_fraction,
    stopper_target,
    batch_size,
    learning_rate,
    verbose_safe,
    mute_output=False,
):
    """Run a single embedding combo; return (emb_idx, val_acc, train_acc, result, combo, log_row) or failure tuple with val_acc None.
    When mute_output is True (e.g. in process workers), stdout/stderr are redirected to devnull for the duration.
    """
    devnull = None
    saved_out = saved_err = None
    if mute_output:
        saved_out, saved_err = sys.stdout, sys.stderr
        devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = sys.stderr = devnull
    try:
        emb_epochs, emb_method, emb_ws, emb_ngram, emb_dm, emb_overlap = combo
        # Per-worker random seed so each thread/process gets different embedding randomness
        worker_seed = (os.getpid() * 31 + (threading.get_ident() % (2**32)) + (int(time.time_ns()) % (2**32))) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        stride = _word_extraction_stride(word_length, emb_overlap)
        log_row = {
            'dataset': dataset_name,
            'experiment_id': experiment_id,
            'TRAIN_VAL_SPLIT_FRACTION': train_val_split_fraction,
            'STOPPER_TARGET_ACCURACY': stopper_target,
            'SAFE_embedding_epochs': emb_epochs,
            'SAFE_embedding_method': emb_method,
            'SAFE_window_size': emb_ws,
            'SAFE_window_overlap_fraction': emb_overlap,
            'SAFE_stride': stride,
            'SAFE_fasttext_max_ngram': emb_ngram if emb_ngram is not None else '',
            'SAFE_doc2vec_dm': emb_dm,
            'accuracy_train': np.nan,
            'accuracy_val': np.nan,
            'accuracy_test': np.nan,
            'chosen': False,
        }
        try:
            safe = SafeTransformer(
                word_length=word_length,
                alphabet_size=alphabet_size,
                embedding_dim=embedding_dim,
                embedding_epochs=emb_epochs,
                embedding_method=emb_method,
                window_size=emb_ws,
                stride=stride,
                seed=worker_seed,
                fasttext_max_ngram=emb_ngram,
                doc2vec_dm=emb_dm,
                force_square_dimension=False,
                verbose=verbose_safe,
            )
            fit_aux = np.concatenate([x_val_raw, x_test_raw], axis=0)
            safe.fit_on_train_and_test(x_tr_raw, fit_aux)

            if base_params.get('augmentation_mode') in ('word_augmenter', 'classic_word_augmenter'):
                x_train_emb, y_train_emb = safe.transform_with_word_augmentation(x_tr_raw, y_tr)
            else:
                x_train_emb = safe.transform(x_tr_raw)
                y_train_emb = y_tr
            x_val_emb = safe.transform(x_val_raw)
            x_test_emb = safe.transform(x_test_raw)

            y_train_enc, y_val_enc, y_test_enc = _encode_labels_train_val_test(y_train_emb, y_val, y_test)
            input_size = x_train_emb.shape[1]
            output_size = len(np.unique(y_train_enc))

            if base_params.get('convolution'):
                x_train_emb = np.expand_dims(x_train_emb, axis=-1)
                x_val_emb = np.expand_dims(x_val_emb, axis=-1)
                x_test_emb = np.expand_dims(x_test_emb, axis=-1)
                input_shape = x_train_emb.shape[1:]
            else:
                x_train_emb = x_train_emb.reshape(x_train_emb.shape[0], -1)
                x_val_emb = x_val_emb.reshape(x_val_emb.shape[0], -1)
                x_test_emb = x_test_emb.reshape(x_test_emb.shape[0], -1)
                input_shape = None

            result = kerne.train_growingnn(
                x_train=x_train_emb,
                y_train=y_train_enc,
                x_val=x_val_emb,
                y_val=y_val_enc,
                labels=list(range(output_size)),
                input_size=input_size,
                hidden_size=int(base_params['hidden_size']),
                output_size=output_size,
                input_shape=input_shape,
                kernel_size=3,
                epochs=int(base_params['epochs']),
                generations=int(base_params['generations']),
                batch_size=batch_size,
                simulation_set_size=int(base_params['simulation_set_size']),
                simulation_time=int(base_params['simulation_time']),
                simulation_epochs=int(base_params['simulation_epochs']),
                simulation_scheduler_type=str(base_params['simulation_scheduler_type']),
                model_name=f"growingnn_{dataset_name}_emb{emb_idx}",
                optimization_factor=float(base_params['optimization_factor']),
                stopper_factor=stopper_target,
                learning_rate=learning_rate,
                activation_fun_name=str(base_params['ACTIVATION_FUN']),
                x_test_eval=x_test_emb,
                y_test_eval=y_test_enc,
            )
            val_acc = result.get('accuracy_val', 0.0)
            train_acc = result.get('accuracy_train', 0.0)
            test_acc = result.get('accuracy_test', float('nan'))
            log_row['accuracy_train'] = train_acc
            log_row['accuracy_val'] = val_acc
            log_row['accuracy_test'] = test_acc
            return (emb_idx, val_acc, train_acc, result, combo, log_row)
        except Exception as e:
            logging.warning(f"  [embedding {emb_idx+1}] FAILED: {e}")
            return (emb_idx, None, None, None, combo, log_row)
    finally:
        if mute_output and devnull is not None:
            sys.stdout, sys.stderr = saved_out, saved_err
            devnull.close()


def _run_one_embedding_worker(args):
    """Picklable entry point for ProcessPoolExecutor: args = (emb_idx, combo, dataset_name, ...)."""
    return _run_one_embedding(*args)


def run_single_experiment(
    dataset_name,
    x_train_raw,
    y_train,
    x_test_raw,
    y_test,
    base_params,
    word_length,
    alphabet_size,
    embedding_dim,
    batch_size,
    learning_rate,
    verbose_safe=False,
    n_workers=None,
    train_val_split_fraction=None,
    stopper_target_accuracy=None,
):
    """
    Run the full pipeline for one (dataset, base_params): try all embedding combos, pick best by validation accuracy.

    Parameters
    ----------
    dataset_name : str
        For logging and embedding log rows.
    x_train_raw, y_train, x_test_raw, y_test : np.ndarray
        Already loaded and optionally augmented (e.g. after classic_augmenter). Z-score normalized.
        The training split is further divided into train/validation (``train_val_split_fraction`` or
        ``DEFAULT_TRAIN_VAL_SPLIT_FRACTION``); the test split is held out for final accuracy only.
    base_params : dict
        Must contain: epochs, generations, hidden_size, simulation_set_size, simulation_time,
        simulation_epochs, simulation_scheduler_type, ACTIVATION_FUN, convolution, augmentation_mode,
        DATA_augmentation_factor_jitter, DATA_augmentation_factor_warp, DATA_augmentation_factor_rgw,
        DATA_MIN_SAMPLES_PER_CLASS, optimization_factor.
        Optional: ``STOPPER_TARGET_ACCURACY`` (float) for ``gnn.AccuracyStopper``; if omitted,
        ``DEFAULT_STOPPER_TARGET_ACCURACY`` from config is used.
    stopper_target_accuracy : float, optional
        If not None, overrides ``base_params['STOPPER_TARGET_ACCURACY']``; otherwise uses that key or
        ``DEFAULT_STOPPER_TARGET_ACCURACY``.
    word_length, alphabet_size, embedding_dim, batch_size, learning_rate : int/float
        Resolved SAFE and training params (not 'X').
    verbose_safe : bool
        Passed to SafeTransformer(verbose=...).
    n_workers : int, optional
        Number of threads for embedding search. If None, uses EMBEDDING_SEARCH_MAX_WORKERS from config.
        1 = sequential; >1 = parallel.
    train_val_split_fraction : float, optional
        Fraction of the training pool used for validation (must be in (0, 1)).
        If None, uses ``DEFAULT_TRAIN_VAL_SPLIT_FRACTION`` from config.

    Returns
    -------
    best_result_dict : dict
        Keys: accuracy_train, accuracy_val, accuracy_test, params (from GrowingNN), plus best embedding params.
    embedding_log_rows : list of dict
        One row per embedding attempt (dataset, experiment_id, SAFE_*, accuracy_train, accuracy_val, accuracy_test, chosen).
    best_embedding_combo : tuple
        (emb_epochs, emb_method, emb_ws, emb_ngram, emb_dm).
    """
    requested_workers = EMBEDDING_SEARCH_MAX_WORKERS if n_workers is None else n_workers
    workers_to_use, available_cpus = resolve_embedding_workers(requested_workers)
    use_processes = workers_to_use > 1 and EMBEDDING_SEARCH_USE_PROCESSES
    if workers_to_use <= 1:
        logging.info(f"Embedding search: available {available_cpus} CPUs, using 1 worker (sequential)")
    else:
        kind = "processes" if use_processes else "threads"
        logging.info(
            f"Embedding search: available {available_cpus} CPUs, using {workers_to_use} {kind} parallel (max_workers = queue limit)"
        )
    n_workers = workers_to_use
    # When parallel, don't pass verbose into workers (they mute output; processes mute in-worker, threads via wrapper)
    effective_verbose = (
        verbose_safe
        if not (workers_to_use > 1 and (MUTE_VERBOSE_INSIDE_THREAD or use_processes))
        else False
    )
    tv_frac = (
        float(train_val_split_fraction)
        if train_val_split_fraction is not None
        else float(DEFAULT_TRAIN_VAL_SPLIT_FRACTION)
    )
    if not (0 < tv_frac < 1):
        raise ValueError(f"train_val_split_fraction must be in (0, 1), got {tv_frac}")
    if stopper_target_accuracy is not None:
        stopper_effective = float(stopper_target_accuracy)
    else:
        stopper_effective = float(
            base_params.get("STOPPER_TARGET_ACCURACY", DEFAULT_STOPPER_TARGET_ACCURACY)
        )
    if not (0 < stopper_effective <= 1.0):
        raise ValueError(f"stopper target must be in (0, 1], got {stopper_effective}")
    split_rs = _train_val_random_state(dataset_name)
    x_tr_raw, x_val_raw, y_tr, y_val = _split_train_val(
        x_train_raw, y_train, tv_frac, split_rs
    )
    n_cls_tr = len(np.unique(y_tr))
    n_cls_val = len(np.unique(y_val))
    n_all = len(x_train_raw)
    actual_val_frac = len(x_val_raw) / n_all if n_all else 0.0
    logging.info(
        f"Train/val split: {len(x_tr_raw)} train ({n_cls_tr} classes), "
        f"{len(x_val_raw)} val ({n_cls_val} classes), "
        f"target_frac≈{tv_frac}, actual_val_frac={actual_val_frac:.4f}, "
        f"random_state={split_rs}"
    )
    if n_cls_val < n_cls_tr:
        logging.warning(
            f"  Validation has fewer classes than training ({n_cls_val} vs {n_cls_tr}) — "
            "usually because some classes have only one sample in the training split; "
            "embedding selection by val accuracy may be noisier."
        )

    embedding_combos = list(iter_embedding_param_combos())
    n_embedding_combos = len(embedding_combos)
    embedding_log_rows = []
    best_val_acc = -1.0
    best_result = None
    best_embedding_combo = None
    experiment_id = f"{dataset_name}_{hash(str(sorted(base_params.items())))}"[:24]

    # Thread-only: when parallel with threads and MUTE_VERBOSE_INSIDE_THREAD, redirect worker stdout/stderr
    muted_threads = set() if (workers_to_use > 1 and not use_processes and MUTE_VERBOSE_INSIDE_THREAD) else None
    real_stdout = real_stderr = stdout_writer = stderr_writer = None
    if muted_threads is not None:
        real_stdout, real_stderr = sys.stdout, sys.stderr
        stdout_writer = _ThreadMutingWriter(real_stdout, muted_threads)
        stderr_writer = _ThreadMutingWriter(real_stderr, muted_threads)
        sys.stdout, sys.stderr = stdout_writer, stderr_writer

    def task(emb_idx_combo):
        emb_idx, combo = emb_idx_combo
        if muted_threads is not None:
            t = threading.current_thread()
            muted_threads.add(t)
        try:
            return _run_one_embedding(
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
                x_test_raw,
                y_test,
                tv_frac,
                stopper_effective,
                batch_size,
                learning_rate,
                effective_verbose,
            )
        finally:
            if muted_threads is not None:
                muted_threads.discard(t)

    try:
        if n_workers <= 1:
            for emb_idx, combo in enumerate(embedding_combos):
                emb_idx_ret, val_acc, train_acc, result, combo_ret, log_row = task((emb_idx, combo))
                embedding_log_rows.append(log_row)
                if val_acc is not None:
                    logging.info(
                        f"  [embedding {emb_idx_ret+1}/{n_embedding_combos}] "
                        f"epochs={combo_ret[0]} method={combo_ret[1]} window={combo_ret[2]} -> val_acc={val_acc:.4f}"
                    )
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_result = result
                        best_embedding_combo = combo_ret
            # keep log rows in combo order
            embedding_log_rows.sort(key=lambda r: (
                r['SAFE_embedding_epochs'], r['SAFE_embedding_method'], r['SAFE_window_size'],
                r.get('SAFE_window_overlap_fraction', 0), r.get('SAFE_fasttext_max_ngram') or '', r['SAFE_doc2vec_dm']))
        else:
            results_by_idx = {}
            if use_processes:
                # Process pool: real multi-core CPU parallelism (not GIL-limited). Each worker gets one combo;
                # max_workers caps concurrency (queue semantics). Workers mute their own stdout/stderr.
                def _args(emb_idx, combo):
                    return (
                        emb_idx, combo, dataset_name, experiment_id,
                        word_length, alphabet_size, embedding_dim, base_params,
                        x_tr_raw, y_tr, x_val_raw, y_val, x_test_raw, y_test,
                        tv_frac,
                        stopper_effective,
                        batch_size, learning_rate, effective_verbose, True,  # mute_output=True
                    )
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(_run_one_embedding_worker, _args(emb_idx, combo)): emb_idx
                        for emb_idx, combo in enumerate(embedding_combos)
                    }
                    for future in as_completed(futures):
                        emb_idx_ret, val_acc, train_acc, result, combo_ret, log_row = future.result()
                        results_by_idx[emb_idx_ret] = (val_acc, train_acc, result, combo_ret, log_row)
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(task, (emb_idx, combo)): emb_idx
                        for emb_idx, combo in enumerate(embedding_combos)
                    }
                    for future in as_completed(futures):
                        emb_idx_ret, val_acc, train_acc, result, combo_ret, log_row = future.result()
                        results_by_idx[emb_idx_ret] = (val_acc, train_acc, result, combo_ret, log_row)
            embedding_log_rows = [results_by_idx[i][4] for i in range(n_embedding_combos)]
            for emb_idx in range(n_embedding_combos):
                val_acc, train_acc, result, combo_ret, _ = results_by_idx[emb_idx]
                if val_acc is not None:
                    logging.info(
                        f"  [embedding {emb_idx+1}/{n_embedding_combos}] "
                        f"epochs={combo_ret[0]} method={combo_ret[1]} window={combo_ret[2]} -> val_acc={val_acc:.4f}"
                    )
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_result = result
                        best_embedding_combo = combo_ret
    finally:
        if muted_threads is not None:
            sys.stdout, sys.stderr = real_stdout, real_stderr
            stdout_writer.close()
            stderr_writer.close()

    if best_embedding_combo is None:
        return None, embedding_log_rows, None

    emb_epochs_b, emb_method_b, emb_ws_b, emb_ngram_b, emb_dm_b, emb_overlap_b = best_embedding_combo
    for row in embedding_log_rows:
        if (row['SAFE_embedding_epochs'], row['SAFE_embedding_method'], row['SAFE_window_size'],
            row.get('SAFE_window_overlap_fraction', 0), row.get('SAFE_fasttext_max_ngram') or '', row['SAFE_doc2vec_dm']) == (
                emb_epochs_b, emb_method_b, emb_ws_b, emb_overlap_b, (emb_ngram_b or ''), emb_dm_b):
            row['chosen'] = True
            break

    best_result_dict = {
        'accuracy_train': best_result.get('accuracy_train', 0.0),
        'accuracy_val': best_result.get('accuracy_val', 0.0),
        'accuracy_test': best_result.get('accuracy_test', float('nan')),
        'params': best_result.get('params', 0.0),
        'SAFE_embedding_epochs': emb_epochs_b,
        'SAFE_embedding_method': emb_method_b,
        'SAFE_window_size': emb_ws_b,
        'SAFE_window_overlap_fraction': emb_overlap_b,
        'SAFE_stride': _word_extraction_stride(
            word_length, emb_overlap_b
        ),
        'SAFE_fasttext_max_ngram': emb_ngram_b,
        'SAFE_doc2vec_dm': emb_dm_b,
        'TRAIN_VAL_SPLIT_FRACTION': tv_frac,
        'STOPPER_TARGET_ACCURACY': stopper_effective,
    }
    return best_result_dict, embedding_log_rows, best_embedding_combo
