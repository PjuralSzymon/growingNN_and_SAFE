# SAFEgrowingNN

Self-contained module that implements the **SAFE + GrowingNN** method used in `growingnn_parameter_analysis_10.py`:

1. **Embedding search** – For each run, try all combinations of SAFE embedding params (epochs, method, window_size, fasttext_max_ngram, doc2vec_dm). This step is **parallelized** when `n_workers` > 1: a **bounded pool** (queue) runs at most `n_workers` trainings at a time; when one finishes, the next combo is started. The training split is divided into train/validation (fraction from `run_single_experiment(..., train_val_split_fraction=...)` or `DEFAULT_TRAIN_VAL_SPLIT_FRACTION` in `config.py`); after all combos complete for one dataset, the best by **validation** accuracy is chosen. **Test** accuracy is reported on the held-out test set only (not used for selection).
2. **SAFE** – Time series → SAX words → embedding (word2vec / fasttext / doc2vec / ppmi_svd) via `safe2.SafeTransformer`.
3. **GrowingNN** – Train on embedded vectors using `kerne.train_growingnn`.
4. **Selection** – Keep the embedding configuration with **best validation accuracy**; return that result (including train, val, and test accuracies) and a per-attempt log.

## Dependencies

- Run from the **repository root** so that `safe2` and `data_loader` are on the path.
- **kerne** is duplicated inside this package as `SAFEgrowingNN.kerne`: it provides `LabelEncoder` (sklearn re-export) and `train_growingnn` (GrowingNN training). The external `growingnn` library must be available (path `D:/repos/growingnn` or set `GROWINGNN_PATH`).

## API

- **`run_single_experiment(..., verbose_safe=False, n_workers=None)`**  
  Runs all embedding combos for one (dataset, base_params), trains GrowingNN for each, selects best by validation accuracy.  
  **`n_workers`**: number of threads for embedding search (default: `EMBEDDING_SEARCH_MAX_WORKERS`). Use `1` for sequential, `>1` for parallel. The value is capped to the machine’s available CPU count; a log line reports “available N CPUs, using M threads parallel”.  
  Returns: `(best_result_dict, embedding_log_rows, best_embedding_combo)` or `(None, log_rows, None)` if all failed.

- **`iter_embedding_param_combos()`**  
  Generator of `(embedding_epochs, method, window_size, fasttext_max_ngram, doc2vec_dm)`.

- **`encode_labels(y_train, y_test)`**  
  Encode labels to integers (uses `kerne.LabelEncoder`).

**Config:** `EMBEDDING_EPOCHS`, `EMBEDDING_METHODS`, ... **`DEFAULT_TRAIN_VAL_SPLIT_FRACTION`** / **`TRAIN_VAL_SPLIT_RANDOM_STATE`** (defaults when `train_val_split_fraction` is not passed), **`EMBEDDING_SEARCH_MAX_WORKERS`** (queue limit, capped to CPUs), **`EMBEDDING_SEARCH_USE_PROCESSES`** (default `True`: processes = real multi-core; `False` = threads), **`MUTE_VERBOSE_INSIDE_THREAD`**.

(Config list above is canonical.) `EMBEDDING_EPOCHS`, `EMBEDDING_METHODS`, `EMBEDDING_WINDOW_SIZES`, `EMBEDDING_FASTTEXT_MAX_NGRAM`, `EMBEDDING_DOC2VEC_DM`, **`EMBEDDING_SEARCH_MAX_WORKERS`** (capped to available CPUs), **`MUTE_VERBOSE_INSIDE_THREAD`** (default `True`; mutes verbose output from parallel threads to avoid interleaved logs)’
## Usage from the playground

```python
# From repo root
from SAFEgrowingNN import run_single_experiment
from data_loader import TimeSeriesLoader, TimeSeriesAugmenter

loader = TimeSeriesLoader(dataset_path='dataset', normalize='zscore')
x_train, y_train = loader.load('BeetleFly', split='train')
x_test, y_test = loader.load('BeetleFly', split='test')

base_params = {'epochs': 200, 'generations': 8, 'hidden_size': 512, ...}  # full base config

best_result, log_rows, best_combo = run_single_experiment(
    'BeetleFly', x_train, y_train, x_test, y_test,
    base_params,
    word_length=6, alphabet_size=7, embedding_dim=4,
    batch_size=4, learning_rate=0.0001,
    n_workers=4,  # optional: parallel embedding search (default from config)
)
```

See `growingnn_parameter_analysis_11.py` for a full experiment loop that uses this package.
