import numpy as np
import pytest
import xgboost as xgb

from xgboost2ww import compute_w10, convert


def _train_binary(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((180, 10), dtype=np.float32)
    y = (X[:, 0] - 0.6 * X[:, 1] + 0.2 * rng.standard_normal(180) > 0).astype(np.int32)
    bst = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3, "eta": 0.2, "verbosity": 0, "seed": seed},
        xgb.DMatrix(X, label=y),
        num_boost_round=20,
    )
    return X, y, bst


def test_w10_smoke():
    X, y, bst = _train_binary()
    W10 = compute_w10(bst, X, y)
    assert W10.ndim == 2
    assert W10.shape[0] == X.shape[0]
    assert W10.shape[1] > 0
    assert W10.dtype == np.float32
    assert np.isfinite(W10).all()


def test_convert_w10_numpy_and_torch():
    X, y, bst = _train_binary(1)
    W10 = convert(bst, X, y, W="W10", return_type="numpy")
    assert isinstance(W10, np.ndarray)
    try:
        import torch  # noqa: F401
    except Exception:
        return
    layer = convert(bst, X, y, W="W10", return_type="torch")
    assert hasattr(layer, "forward")


def test_w10_weighted_mean_zero_and_whitening():
    X, y, bst = _train_binary(2)
    W10, info = compute_w10(bst, X, y, return_info=True)
    h = 1.0 / (1.0 + np.exp(-bst.predict(xgb.DMatrix(X), output_margin=True)))
    h = np.clip(h * (1.0 - h), 1e-6, None)
    m = h / h.sum()

    widths = info["tree_block_widths"]
    start = 0
    for width in widths:
        block = W10[:, start : start + width]
        start += width
        centered = block / np.sqrt(m)[:, None]
        assert np.allclose((m[:, None] * centered).sum(axis=0), 0.0, atol=5e-3)
        cov = block.T @ block
        assert np.allclose(cov, np.eye(width), atol=8e-2)


def test_w10_zero_support_leaf_handling():
    X, y, bst = _train_binary(3)
    X_small = X[:30]
    y_small = y[:30]
    W10, info = compute_w10(bst, X_small, y_small, return_info=True)
    assert W10.shape[0] == X_small.shape[0]
    assert np.isfinite(W10).all()
    assert info["skipped_trees"] >= 0


def test_w10_unsupported_objective_message():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((120, 5), dtype=np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    bst = xgb.train({"objective": "binary:logistic", "verbosity": 0}, xgb.DMatrix(X, label=y), num_boost_round=8)
    with pytest.raises(NotImplementedError, match="supports objective"):
        compute_w10(bst, X, y, objective="rank:pairwise")


def test_w10_backend_parity_torch_cpu():
    X, y, bst = _train_binary(5)
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch unavailable")
    W_np = compute_w10(bst, X, y, backend="numpy")
    W_t = compute_w10(bst, X, y, backend="torch", device="cpu")
    assert np.allclose(W_np, W_t, atol=1e-4, rtol=1e-4)
