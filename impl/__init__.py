"""
SAFEgrowingNN: pipeline that (1) tries all embedding param combinations, (2) uses SAFE for transform,
(3) trains GrowingNN, and (4) selects the best embedding by validation accuracy.

Usage (from repo root):
    from SAFEgrowingNN import run_single_experiment, iter_embedding_param_combos
    from SAFEgrowingNN.config import EMBEDDING_EPOCHS, EMBEDDING_METHODS, ...
"""

from .config import (
    EMBEDDING_EPOCHS,
    EMBEDDING_METHODS,
    EMBEDDING_WINDOW_SIZES,
    EMBEDDING_WINDOW_OVERLAP_FRACTION,
    EMBEDDING_FASTTEXT_MAX_NGRAM,
    EMBEDDING_DOC2VEC_DM,
    EMBEDDING_SEARCH_MAX_WORKERS,
    EMBEDDING_SEARCH_USE_PROCESSES,
    MUTE_VERBOSE_INSIDE_THREAD,
    DEFAULT_TRAIN_VAL_SPLIT_FRACTION,
    DEFAULT_STOPPER_TARGET_ACCURACY,
    TRAIN_VAL_SPLIT_RANDOM_STATE,
    iter_embedding_param_combos,
    _word_extraction_stride,
)
from .pipeline import run_single_experiment, encode_labels

__all__ = [
    'run_single_experiment',
    'encode_labels',
    'iter_embedding_param_combos',
    'EMBEDDING_EPOCHS',
    'EMBEDDING_METHODS',
    'EMBEDDING_WINDOW_SIZES',
    'EMBEDDING_WINDOW_OVERLAP_FRACTION',
    '_word_extraction_stride',
    'EMBEDDING_FASTTEXT_MAX_NGRAM',
    'EMBEDDING_DOC2VEC_DM',
    'EMBEDDING_SEARCH_MAX_WORKERS',
    'EMBEDDING_SEARCH_USE_PROCESSES',
    'MUTE_VERBOSE_INSIDE_THREAD',
    'DEFAULT_TRAIN_VAL_SPLIT_FRACTION',
    'DEFAULT_STOPPER_TARGET_ACCURACY',
    'TRAIN_VAL_SPLIT_RANDOM_STATE',
]
