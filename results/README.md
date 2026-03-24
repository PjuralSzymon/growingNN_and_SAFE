# Experiment results

This folder holds **CSV tables** from batch runs, plus a few **reference** CSVs that document hyperparameter defaults. Run outputs are **generated** by your driver; the meta-parameter CSVs are **edited** when settings change.

## How these CSVs are produced (general)

Runs are driven by a **small experiment script** (outside this folder) that you execute from the **repository root** with dependencies installed (`pip install -r requirements.txt`). Typical flow:

1. **Load** time-series classification data (e.g. UCR-style train/test splits on disk).
2. **Call** the project pipeline `run_single_experiment` from `impl` (SAFE embedding search + GrowingNN training; best embedding chosen by validation accuracy).
3. **Sweep** settings you care about—datasets, train/validation split fraction, stopper target, and any base GrowingNN/SAFE knobs your script exposes.
4. **Append** one row per finished run to a CSV such as `parameter_analysis_results.csv` (dataset name, train/val/test accuracy, chosen SAFE embedding fields, and the grid parameters you varied).

Parallel workers or resume logic are optional; the idea is always: **one script orchestrates loads + `run_single_experiment` + CSV writes**.

For a **minimal, single-dataset** run, see `example/README.md` and `example/run_example_on_data.py`.

## What each `.csv` file is

| File | Short description |
|------|-------------------|
| **`parameter_analysis_results.csv`** | **Run output.** One row per finished experiment (dataset × grid point): train/val/test accuracy, GrowingNN `params` count, grid columns you swept (`TRAIN_VAL_SPLIT_FRACTION`, `STOPPER_TARGET_ACCURACY`, …), and the **chosen** SAFE embedding fields (`SAFE_embedding_method`, window, stride, etc.). Produced by your batch driver after each `run_single_experiment` completes. |
| **`size_results.csv`** | **Reference or comparison table.** Per dataset, columns for baselines / methods (e.g. FCN, ResNet, Encoder, ROCKET, SAFE) and a **GrowingNN** column (often parameter count or a size metric—depends how you exported it). Not written by `run_single_experiment` itself; fill from papers or separate tooling. |
| **`per_dataset_meta_parameters.csv`** | **Config reference.** One row per dataset: default `SAFE_embedding_dim`, `SAX_alphabet_size`, `SAFE_word_length`, `batch_size`, `learning_rate` when your harness uses `'X'` placeholders in the grid. Edit when you tune per-dataset defaults. |
| **`base_parameter_grid.csv`** | **Config reference.** One row per shared hyperparameter (`epochs`, `hidden_size`, `TRAIN_VAL_SPLIT_FRACTION`, `STOPPER_TARGET_ACCURACY`, …) for a single grid snapshot; documents the fixed base point your driver iterates over or combines with sweeps. |

In the table above, the first two files are **experiment outputs or comparisons**; the last two are **meta-parameter snapshots** (settings documentation), not automatic pipeline outputs. Edit those two when your harness changes; they are not produced by `run_single_experiment`.

Adjust file names in your own harness if you prefer; this README only describes the usual pattern.
