import numpy as np
import pytest
import xgboost as xgb

from xgboost2ww import compute_matrices, convert


@pytest.fixture
def multiclass_setup():
    rng = np.random.default_rng(123)
    n = 120
    X = rng.standard_normal((n, 6), dtype=np.float32)
    scores = np.stack(
        [
            X[:, 0] - 0.3 * X[:, 1],
            X[:, 1] + 0.2 * X[:, 2],
            -X[:, 0] + 0.4 * X[:, 2],
        ],
        axis=1,
    )
    y = np.argmax(scores + 0.1 * rng.standard_normal(scores.shape), axis=1).astype(np.int32)

    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 2,
        "eta": 0.25,
        "verbosity": 0,
        "seed": 7,
    }
    bst = xgb.train(params, dtrain, num_boost_round=20)
    return bst, X, y


def test_multiclass_per_class_shapes(multiclass_setup):
    bst, X, y = multiclass_setup
    mats = compute_matrices(bst, X, y, nfolds=3, t_points=12, random_state=0, multiclass="per_class")

    assert isinstance(mats, dict)
    assert sorted(mats.keys()) == [0, 1, 2]
    for k in range(3):
        m = mats[k]
        assert m.W1.shape == (X.shape[0], len(m.endpoints))
        assert m.W2.shape == m.W1.shape
        assert m.W7.shape == m.W1.shape
        assert m.W8.shape == m.W1.shape
        assert m.m_final.shape == (X.shape[0],)
        assert m.W1.dtype == np.float32
        assert np.all(np.isfinite(m.W1))
        assert np.all(np.isfinite(m.W2))
        assert np.all(np.isfinite(m.W7))
        assert np.all(np.isfinite(m.W8))


def test_multiclass_stack_shape(multiclass_setup):
    bst, X, y = multiclass_setup
    mats = compute_matrices(bst, X, y, nfolds=3, t_points=12, random_state=0, multiclass="stack")
    T = len(mats.endpoints)

    assert mats.W1.shape == (X.shape[0] * 3, T)
    assert mats.W2.shape == mats.W1.shape
    assert mats.W7.shape == mats.W1.shape
    assert mats.W8.shape == mats.W1.shape
    assert mats.m_final.shape == (X.shape[0], 3)


def test_multiclass_avg_shape(multiclass_setup):
    bst, X, y = multiclass_setup
    mats = compute_matrices(bst, X, y, nfolds=3, t_points=12, random_state=0, multiclass="avg")

    assert mats.W1.shape == (X.shape[0], len(mats.endpoints))
    assert mats.W2.shape == mats.W1.shape
    assert mats.W7.shape == mats.W1.shape
    assert mats.W8.shape == mats.W1.shape
    assert mats.m_final.shape == (X.shape[0], 3)


def test_multiclass_error_mode_raises(multiclass_setup):
    bst, X, y = multiclass_setup
    with pytest.raises(ValueError, match="Detected multiclass labels"):
        compute_matrices(bst, X, y, nfolds=3, t_points=12, random_state=0, multiclass="error")


def test_multiclass_folds_change_output(multiclass_setup):
    bst, X, y = multiclass_setup
    mats_0 = compute_matrices(bst, X, y, nfolds=3, t_points=12, random_state=0, multiclass="per_class")
    mats_1 = compute_matrices(bst, X, y, nfolds=3, t_points=12, random_state=1, multiclass="per_class")

    assert not np.allclose(mats_0[0].W1, mats_1[0].W1)


def test_convert_per_class_torch_raises(multiclass_setup):
    bst, X, y = multiclass_setup
    with pytest.raises(ValueError, match="unsupported"):
        convert(
            bst,
            X,
            y,
            W="W7",
            nfolds=3,
            t_points=12,
            random_state=0,
            multiclass="per_class",
            return_type="torch",
        )
