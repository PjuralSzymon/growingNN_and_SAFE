# GrowingNN and SAFE

SAFE + GrowingNN for univariate time-series classification.

Published reference tables live in `results/` (do not overwrite). New runs write to `results_adaptive/` (gitignored).

## Setup

```bash
pip install -r requirements.txt
```

## Run experiments

Uses paper hyperparameters from `results/` and a budgeted embedding search over the same grid as the paper.

Search stops when **validation** reaches the paper Table 1 level. The number compared to the paper is the chosen config's **test** accuracy.

```bash
python experiments/run_experiments.py --list --download
python experiments/run_experiments.py
python experiments/run_experiments.py --compare-only
```

Outputs: `results_adaptive/`
