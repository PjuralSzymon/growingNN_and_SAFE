# GrowingNN and SAFE

SAFE + GrowingNN for univariate time-series classification.

Published reference tables live in `results/` (do not overwrite). New runs write to `results_adaptive/` (gitignored).

## Setup

```bash
pip install -r requirements.txt
```

## Run experiments

Uses paper hyperparameters from `results/` and a budgeted embedding search over the same grid as the paper:

```bash
python experiments/run_experiments.py --list --download
python experiments/run_experiments.py
python experiments/run_experiments.py --compare-only
```

Outputs: `results_adaptive/`
