# Examples (SAFE + GrowingNN on UCR time series)

Scripts in this folder load **UCR** classification datasets and run the pipeline in `impl/` (SAFE embedding search + GrowingNN training via `run_single_experiment`).

## Where to run from

Run commands from the **repository root** (`growingNN_and_SAFE`), not from inside `example/`. The script adds the repo root to `sys.path` so `impl` imports work.

```bash
cd /path/to/growingNN_and_SAFE
python -m pip install -r requirements.txt
```

## Data layout

For a dataset named `MyDataset`, the loader expects:

```text
example/data/MyDataset/MyDataset_TRAIN.txt
example/data/MyDataset/MyDataset_TEST.txt
```

Each row is **one time series**: first column = class label (integer), remaining columns = values (UCR `.txt` format).

The `example/data/` directory is **gitignored** (see `example/.gitignore`). You can download or copy datasets locally without committing large files.

### Where to get the data

- **UCR Time Series Classification Archive** — [https://www.cs.ucr.edu/~eamonn/time_series_data_2018/](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)  
  Unzip a problem folder and copy the `*_TRAIN.txt` / `*_TEST.txt` files into `example/data/<ProblemName>/` using the same `<ProblemName>` prefix as the archive (e.g. `Earthquakes`, `DistalPhalanxOutlineAgeGroup`).

## Run the example

```bash
# Default dataset: Earthquakes (quick embedding grid)
python example/run_example_on_data.py

# Another bundled dataset name (if you have the files under example/data/...)
python example/run_example_on_data.py --dataset DistalPhalanxOutlineAgeGroup

# Full SAFE embedding grid from config (very slow on CPU)
python example/run_example_on_data.py --dataset Earthquakes --full

# Parallel embedding search (optional)
python example/run_example_on_data.py --n-workers 4
```

Use `--help` for all options.

## Requirements

Same as the main project: `requirements.txt` at the repo root (includes `growingnn`, `numpy`, `gensim`, `saxpy`, etc.).
