"""
SAFEgrowingNN: embedding grid and default config.
All embedding param combinations tried per experiment; best by validation accuracy is selected.
"""

import os
from itertools import product

# Default train/val fraction when callers do not pass `train_val_split_fraction` to `run_single_experiment`.
DEFAULT_TRAIN_VAL_SPLIT_FRACTION = 0.2
TRAIN_VAL_SPLIT_RANDOM_STATE = 42

# Target accuracy passed to GrowingNN `AccuracyStopper` when `base_params` has no `STOPPER_TARGET_ACCURACY`.
DEFAULT_STOPPER_TARGET_ACCURACY = 0.85

# Number of threads/workers for embedding search (1 = sequential; >1 = parallel).
# At runtime this is capped to the machine's available CPU count.
EMBEDDING_SEARCH_MAX_WORKERS = 10

# When True (default), mute verbose output from each thread during parallel embedding search
# so logs are not interleaved. Sequential runs always use the caller's verbose_safe.
MUTE_VERBOSE_INSIDE_THREAD = True

# Use processes instead of threads for parallel embedding search. Processes give real multi-core
# CPU usage (threads are limited by Python's GIL). Set False to use threads (less memory per worker).
EMBEDDING_SEARCH_USE_PROCESSES = True


def get_available_cpu_count():
    """Return the number of CPUs available on this machine, or 1 if undetected."""
    n = os.cpu_count()
    return n if n is not None and n >= 1 else 1


def resolve_embedding_workers(requested):
    """
    Resolve how many workers to use for embedding search: clamp requested to [1, available CPUs].
    Returns (workers_to_use, available_cpus) for logging.
    """
    available = get_available_cpu_count()
    requested = 1 if requested is None or requested < 1 else int(requested)
    workers_to_use = min(requested, available)
    return workers_to_use, available

# Embedding param grid (tried exhaustively per run)
EMBEDDING_EPOCHS = [ 10, 20, 30, 40, 50]
EMBEDDING_METHODS = ['word2vec', 'fasttext', 'ppmi_svd']
EMBEDDING_WINDOW_SIZES = [1, 2, 3, 4, 5, 6, 7]
EMBEDDING_FASTTEXT_MAX_NGRAM = [4]  # originally was 2,3
EMBEDDING_DOC2VEC_DM = [1]  # originally was 0,1

# Stride overlap: fraction of the word-extraction window that overlaps when sliding (0 = no overlap, 0.5 = 50%).
# stride = word_length * (1 - overlap); e.g. word_length=4, overlap=0.5 -> stride=2 (2 new elements per step).
EMBEDDING_WINDOW_OVERLAP_FRACTION = [0, 0.1, 0.25, 0.5]


def _word_extraction_stride(word_length: int, overlap_fraction: float) -> int:
    """Stride for sliding window: max(1, int(word_length * (1 - overlap_fraction)))."""
    return max(1, int(word_length * (1.0 - overlap_fraction)))


def iter_embedding_param_combos():
    """Yield (embedding_epochs, embedding_method, window_size, fasttext_max_ngram, doc2vec_dm, window_overlap_fraction)."""
    for method in EMBEDDING_METHODS:
        if method == 'ppmi_svd':
            for ws, ov in product(EMBEDDING_WINDOW_SIZES, EMBEDDING_WINDOW_OVERLAP_FRACTION):
                yield (200, method, ws, None, 0, ov)
            continue
        if method == 'fasttext':
            for epochs, ws, ng, ov in product(
                EMBEDDING_EPOCHS, EMBEDDING_WINDOW_SIZES, EMBEDDING_FASTTEXT_MAX_NGRAM, EMBEDDING_WINDOW_OVERLAP_FRACTION
            ):
                yield (epochs, method, ws, ng, 0, ov)
        elif method == 'doc2vec':
            for epochs, ws, dm, ov in product(
                EMBEDDING_EPOCHS, EMBEDDING_WINDOW_SIZES, EMBEDDING_DOC2VEC_DM, EMBEDDING_WINDOW_OVERLAP_FRACTION
            ):
                yield (epochs, method, ws, None, dm, ov)
        else:
            for epochs, ws, ov in product(EMBEDDING_EPOCHS, EMBEDDING_WINDOW_SIZES, EMBEDDING_WINDOW_OVERLAP_FRACTION):
                yield (epochs, method, ws, None, 0, ov)
