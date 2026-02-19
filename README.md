# xgboost2ww

Convert XGBoost boosting dynamics into WeightWatcher-style correlation matrices (W1/W2/W7/W8).

## Install

Development install:

```bash
pip install -e .
pip install weightwatcher torch
```

Minimal runtime install (for a future PyPI install):

```bash
pip install xgboost2ww
pip install weightwatcher
```

## Quickstart (compute_matrices)

```python
import numpy as np
import xgboost as xgb

from xgboost2ww import compute_matrices

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 12)).astype(np.float32)
logits = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * rng.normal(size=300)
y = (logits > 0).astype(np.int32)

dtrain = xgb.DMatrix(X, label=y)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "seed": 0,
    "verbosity": 0,
}
rounds = 40
bst = xgb.train(params, dtrain, num_boost_round=rounds)

# Reproducibility knobs for fold training inside compute_matrices / convert
train_params = params
num_boost_round = rounds

mats = compute_matrices(
    bst,
    X,
    y,
    nfolds=5,
    t_points=40,
    random_state=0,
    train_params=train_params,
    num_boost_round=num_boost_round,
)

W7 = mats.W7
print(W7.shape)
```

## Quickstart (convert + WeightWatcher)

```python
import weightwatcher as ww

from xgboost2ww import convert

layer = convert(
    bst,
    X,
    y,
    W="W7",
    return_type="torch",
    nfolds=5,
    t_points=40,
    random_state=0,
    train_params=train_params,
    num_boost_round=num_boost_round,
)

watcher = ww.WeightWatcher(model=layer)
details_df = watcher.analyze(randomize=True, plot=False)

alpha = details_df["alpha"].iloc[0]
rand_num_spikes = details_df["rand_num_spikes"].iloc[0]
print({"alpha": alpha, "rand_num_spikes": rand_num_spikes})
```

For initial evaluation, you do not need `detX=True`. If you want determinant-based diagnostics, you can pass `detX=True`.

## Notes / limitations

- Binary classification is the default workflow.
- Multiclass requires setting `multiclass` explicitly (supported modes: `"per_class"`, `"stack"`, `"avg"`).
- `convert(..., multiclass="per_class", return_type="torch")` is unsupported and raises; for multiclass per-class output, use `return_type="numpy"`.
- `torch` is optional unless you need `convert(..., return_type="torch")`.
