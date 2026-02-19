import numpy as np
import pytest
import xgboost as xgb


@pytest.fixture
def toy_binary_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((140, 8), dtype=np.float32)
    y = (X[:, 0] - 0.4 * X[:, 1] + 0.2 * rng.standard_normal(140) > 0).astype(np.int32)
    return X, y


@pytest.fixture
def booster(toy_binary_data):
    X, y = toy_binary_data
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "max_depth": 2,
        "eta": 0.2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "verbosity": 0,
        "seed": 7,
    }
    return xgb.train(params, dtrain, num_boost_round=24)
