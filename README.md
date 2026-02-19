# xgboost2ww

Compute WeightWatcher-style correlation matrices for XGBoost models: **W1, W2, W7, W8**.

These matrices are derived from **out-of-fold (OOF) margin increments** along the boosting
trajectory, then centered / de-trended as in our workflow.

## Install (dev)
```bash
pip install -e .
```

## Quickstart
```python
import numpy as np
import xgboost as xgb
from xgboost2ww import compute_matrices, convert

X = np.random.randn(200, 10).astype(np.float32)
y = (X[:, 0] + 0.25*np.random.randn(200) > 0).astype(np.int32)

dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.2, "verbosity": 0, "seed": 0}
bst = xgb.train(params, dtrain, num_boost_round=50)

mats = compute_matrices(bst, X, y, nfolds=5, t_points=50)
W7 = mats.W7

layer = convert(bst, X, y, W="W7", return_type="torch")  # torch.nn.Sequential
```

## Notes
- Current implementation assumes **binary labels (0/1)**.
- `torch` is optional unless you use `convert(..., return_type="torch")` or `to_linear_layer`.
