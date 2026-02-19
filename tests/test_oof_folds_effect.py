# tests/test_oof_folds_effect.py
import numpy as np
import pytest

from xgboost2ww import compute_matrices


def _mean_abs_diff(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.mean(np.abs(A - B)))


@pytest.mark.parametrize("name", ["W1", "W2", "W7", "W8"])
def test_changing_nfolds_changes_matrices(booster, toy_binary_data, name):
    """
    Strong evidence OOF folds are actually used:
    same model/data, different nfolds => different OOF partitions => different OOF increments => different matrices.
    """
    X, y = toy_binary_data

    mats_2 = compute_matrices(booster, X, y, nfolds=2, t_points=20, random_state=0, verbose=False)
    mats_5 = compute_matrices(booster, X, y, nfolds=5, t_points=20, random_state=0, verbose=False)

    A = getattr(mats_2, name)
    B = getattr(mats_5, name)

    # Not nearly identical
    assert not np.allclose(A, B, rtol=1e-6, atol=1e-7), f"{name} unexpectedly identical across nfolds"
    # Quantitative non-trivial difference threshold
    assert _mean_abs_diff(A, B) > 1e-5, f"{name} mean abs diff too small across nfolds"


@pytest.mark.parametrize("name", ["W1", "W2", "W7", "W8"])
def test_changing_random_state_changes_matrices(booster, toy_binary_data, name):
    """
    Another strong OOF verification:
    same nfolds, different random_state => different stratified splits => different OOF matrices.
    """
    X, y = toy_binary_data

    mats_a = compute_matrices(booster, X, y, nfolds=5, t_points=20, random_state=0, verbose=False)
    mats_b = compute_matrices(booster, X, y, nfolds=5, t_points=20, random_state=123, verbose=False)

    A = getattr(mats_a, name)
    B = getattr(mats_b, name)

    assert not np.allclose(A, B, rtol=1e-6, atol=1e-7), f"{name} unexpectedly identical across random_state"
    assert _mean_abs_diff(A, B) > 1e-5, f"{name} mean abs diff too small across random_state"


def test_oof_m_final_changes_with_splits(booster, toy_binary_data):
    X, y = toy_binary_data
    a = compute_matrices(booster, X, y, nfolds=5, t_points=20, random_state=0)
    b = compute_matrices(booster, X, y, nfolds=5, t_points=20, random_state=999)

    assert not np.allclose(a.m_final, b.m_final, rtol=1e-6, atol=1e-7)
    assert float(np.mean(np.abs(a.m_final - b.m_final))) > 1e-6
