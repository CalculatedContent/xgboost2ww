# tests/test_matrices_basic.py
import numpy as np
import pytest

from xgboost2ww import compute_matrices, convert


@pytest.mark.parametrize("nfolds,t_points", [(2, 20), (5, 20)])
def test_matrices_shapes_dtypes_finite(booster, toy_binary_data, nfolds, t_points):
    X, y = toy_binary_data
    mats = compute_matrices(
        booster, X, y, nfolds=nfolds, t_points=t_points, random_state=0, verbose=False
    )

    N = X.shape[0]
    T = len(mats.endpoints)

    for name in ("W1", "W2", "W7", "W8", "W9"):
        A = getattr(mats, name)
        assert isinstance(A, np.ndarray)
        assert A.shape == (N, T)
        assert A.dtype == np.float32
        assert np.isfinite(A).all()

    assert mats.m_final.shape == (N,)
    assert mats.m_final.dtype == np.float32
    assert np.isfinite(mats.m_final).all()


def test_centering_invariants(booster, toy_binary_data):
    X, y = toy_binary_data
    mats = compute_matrices(booster, X, y, nfolds=5, t_points=25, random_state=0)

    assert np.allclose(mats.W1.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(mats.W7.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(mats.W2.mean(axis=1), 0.0, atol=1e-4)

    assert np.isfinite(mats.W8).all()
    assert np.isfinite(mats.W9).all()
    assert float(np.linalg.norm(mats.W8)) > 0.0


def test_compute_matrices_with_training_overrides_runs(toy_binary_data):
    import xgboost as xgb

    X, y = toy_binary_data
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.15,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "verbosity": 0,
        "seed": 11,
    }
    rounds = 18
    bst = xgb.train(params, dtrain, num_boost_round=rounds)

    mats_default = compute_matrices(bst, X, y, nfolds=3, t_points=16, random_state=0)
    mats_override = compute_matrices(
        bst,
        X,
        y,
        nfolds=3,
        t_points=16,
        random_state=0,
        train_params=params,
        num_boost_round=rounds,
    )

    for mats in (mats_default, mats_override):
        T = len(mats.endpoints)
        assert mats.W1.shape == (X.shape[0], T)
        assert mats.W2.shape == (X.shape[0], T)
        assert mats.W7.shape == (X.shape[0], T)
        assert mats.W8.shape == (X.shape[0], T)
        assert mats.W9.shape == (X.shape[0], T)
        assert mats.m_final.shape == (X.shape[0],)
        assert np.isfinite(mats.W1).all()
        assert np.isfinite(mats.W2).all()
        assert np.isfinite(mats.W7).all()
        assert np.isfinite(mats.W8).all()
        assert np.isfinite(mats.W9).all()


def test_convert_defaults_to_w1(booster, toy_binary_data):
    X, y = toy_binary_data
    W_default = convert(booster, X, y, return_type="numpy")
    W_w1 = convert(booster, X, y, W="W1", return_type="numpy")

    assert W_default.shape == W_w1.shape
    assert np.allclose(W_default, W_w1, atol=1e-7, rtol=1e-7)
