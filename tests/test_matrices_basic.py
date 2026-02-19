# tests/test_matrices_basic.py
import numpy as np
import pytest

from xgboost2ww import compute_matrices


@pytest.mark.parametrize("nfolds,t_points", [(2, 20), (5, 20)])
def test_matrices_shapes_dtypes_finite(booster, toy_binary_data, nfolds, t_points):
    X, y = toy_binary_data
    mats = compute_matrices(
        booster, X, y, nfolds=nfolds, t_points=t_points, random_state=0, verbose=False
    )

    N = X.shape[0]
    T = len(mats.endpoints)

    for name in ("W1", "W2", "W7", "W8"):
        A = getattr(mats, name)
        assert isinstance(A, np.ndarray)
        assert A.shape == (N, T)
        assert A.dtype == np.float32
        assert np.isfinite(A).all()

    assert mats.m_final.shape == (N,)
    assert mats.m_final.dtype == np.float32
    assert np.isfinite(mats.m_final).all()


def test_centering_invariants(booster, toy_binary_data):
    """
    Verify the algebraic centering properties implied by our definitions:
      - W1 is column-centered
      - W2 is row-centered
      - W7 is column-centered
      - W8 should be (approximately) weighted column-centered before scaling,
        but after scaling it won't necessarily have exact zero column means.
        So we check W7 centering and basic sanity for W8.
    """
    X, y = toy_binary_data
    mats = compute_matrices(booster, X, y, nfolds=5, t_points=25, random_state=0)

    # W1 and W7 are column-centered by construction
    assert np.allclose(mats.W1.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(mats.W7.mean(axis=0), 0.0, atol=1e-4)

    # W2 is row-centered by construction
    assert np.allclose(mats.W2.mean(axis=1), 0.0, atol=1e-4)

    # W8 sanity: finite and non-trivial norm
    assert np.isfinite(mats.W8).all()
    assert float(np.linalg.norm(mats.W8)) > 0.0
