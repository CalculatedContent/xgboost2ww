import numpy as np
import xgboost as xgb

from xgboost2ww import compute_matrices

def test_compute_matrices_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 6), dtype=np.float32)
    y = (X[:, 0] + 0.1 * rng.standard_normal(120) > 0).astype(np.int32)

    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3, "verbosity": 0, "seed": 0}
    bst = xgb.train(params, dtrain, num_boost_round=30)

    mats = compute_matrices(bst, X, y, nfolds=3, t_points=20, random_state=0, verbose=False)

    assert mats.W1.shape == (120, len(mats.endpoints))
    assert mats.W2.shape == mats.W1.shape
    assert mats.W7.shape == mats.W1.shape
    assert mats.W8.shape == mats.W1.shape
    assert mats.m_final.shape == (120,)
