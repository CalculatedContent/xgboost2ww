import json
from types import SimpleNamespace

import numpy as np
import pytest
import xgboost as xgb

from xgboost2ww import compute_matrices, convert
from xgboost2ww.core import _infer_params, _out_of_fold_increments
import xgboost2ww.core as core_mod




@pytest.fixture
def multiclass_setup_local():
    rng = np.random.default_rng(123)
    n = 120
    X = rng.standard_normal((n, 6), dtype=np.float32)
    scores = np.stack([X[:, 0] - 0.3 * X[:, 1], X[:, 1] + 0.2 * X[:, 2], -X[:, 0] + 0.4 * X[:, 2]], axis=1)
    y = np.argmax(scores + 0.1 * rng.standard_normal(scores.shape), axis=1).astype(np.int32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "multi:softprob", "num_class": 3, "max_depth": 2, "eta": 0.25, "verbosity": 0, "seed": 7}
    bst = xgb.train(params, dtrain, num_boost_round=20)
    return bst, X, y

def _leaf_node_ids(tree_json: str) -> set[int]:
    tree = json.loads(tree_json)

    out: set[int] = set()

    def _walk(node):
        if "leaf" in node:
            out.add(int(node["nodeid"]))
            return
        for child in node.get("children", []):
            _walk(child)

    _walk(tree)
    return out


def test_w9_binary_smoke(booster, toy_binary_data):
    X, y = toy_binary_data
    mats = compute_matrices(booster, X, y, nfolds=3, t_points=18, random_state=0)

    assert mats.W9.shape == mats.W8.shape
    assert mats.W9.dtype == np.float32
    assert np.isfinite(mats.W9).all()


def test_convert_w9_numpy_and_torch(booster, toy_binary_data):
    X, y = toy_binary_data

    W9 = convert(booster, X, y, W="W9", return_type="numpy", nfolds=3, t_points=16, random_state=0)
    assert isinstance(W9, np.ndarray)

    torch = pytest.importorskip("torch")
    layer = convert(booster, X, y, W="W9", return_type="torch", nfolds=3, t_points=16, random_state=0)
    assert isinstance(layer, torch.nn.Sequential)


def test_convert_reports_missing_requested_matrix(monkeypatch, booster, toy_binary_data):
    X, y = toy_binary_data

    fake = SimpleNamespace(
        W1=np.zeros((X.shape[0], 3), dtype=np.float32),
        W2=np.zeros((X.shape[0], 3), dtype=np.float32),
        W7=np.zeros((X.shape[0], 3), dtype=np.float32),
        W8=np.zeros((X.shape[0], 3), dtype=np.float32),
    )

    monkeypatch.setattr(core_mod, "compute_matrices", lambda *args, **kwargs: fake)

    with pytest.raises(AttributeError, match="expected W9, upgrade/reinstall xgboost2ww"):
        convert(booster, X, y, W="W9", return_type="numpy")


def test_w9_formula_reconstruction_binary(booster, toy_binary_data):
    X, y = toy_binary_data
    nfolds = 3
    t_points = 14
    random_state = 2

    mats = compute_matrices(booster, X, y, nfolds=nfolds, t_points=t_points, random_state=random_state)
    dF_oof, m_final, _, _, multiclass_output, gamma_diag = _out_of_fold_increments(
        booster,
        X,
        y,
        nfolds=nfolds,
        t_points=t_points,
        random_state=random_state,
    )
    assert not multiclass_output

    p = 1.0 / (1.0 + np.exp(-m_final))
    h = np.clip(p * (1.0 - p), 1e-6, None)
    A = dF_oof - ((h[:, None] * dF_oof).sum(axis=0, keepdims=True) / (h.sum() + 1e-12))
    W9_recon = (np.sqrt(h)[:, None] * A) / np.sqrt(np.clip(gamma_diag, 1e-6, None))[None, :]

    assert np.allclose(mats.W9, W9_recon.astype(np.float32), atol=1e-5, rtol=1e-5)


def test_w9_endpoint_block_gamma_uses_full_block(booster, toy_binary_data):
    X, y = toy_binary_data
    nfolds = 2
    t_points = 5
    random_state = 3

    dF_oof, m_final, endpoints, _, multiclass_output, gamma_diag = _out_of_fold_increments(
        booster,
        X,
        y,
        nfolds=nfolds,
        t_points=t_points,
        random_state=random_state,
    )
    assert dF_oof.shape[0] == m_final.shape[0]
    assert not multiclass_output

    from sklearn.model_selection import StratifiedKFold

    params = _infer_params(booster)
    num_rounds = int(booster.num_boosted_rounds())
    reg_lambda = float(params.get("lambda", 1.0))
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)

    gamma_full = np.zeros_like(gamma_diag, dtype=np.float64)
    gamma_endpoint_only = np.zeros_like(gamma_diag, dtype=np.float64)

    for tr_idx, va_idx in skf.split(np.zeros(X.shape[0]), y):
        bst = xgb.train(params, xgb.DMatrix(X[tr_idx], label=y[tr_idx]), num_boost_round=num_rounds, verbose_eval=False)
        dump_json = bst.get_dump(dump_format="json")
        leaf_mass = np.array(
            [sum(float(v) ** 2 for v in _leaf_values(t)) for t in dump_json], dtype=np.float64
        )
        starts = np.concatenate(([0], endpoints[:-1]))
        fold_full = np.array([leaf_mass[int(s):int(e)].sum() for s, e in zip(starts, endpoints)], dtype=np.float64)
        fold_end = np.array([leaf_mass[int(e) - 1] for e in endpoints], dtype=np.float64)
        weight = len(va_idx) / X.shape[0]
        gamma_full += fold_full * weight
        gamma_endpoint_only += fold_end * weight

    gamma_full *= reg_lambda
    gamma_endpoint_only *= reg_lambda

    assert np.allclose(gamma_diag, gamma_full.astype(np.float32), atol=1e-5, rtol=1e-5)
    assert np.any(np.abs(gamma_diag - gamma_endpoint_only.astype(np.float32)) > 1e-6)


def _leaf_values(tree_json: str) -> list[float]:
    tree = json.loads(tree_json)
    out: list[float] = []

    def _walk(node):
        if "leaf" in node:
            out.append(float(node["leaf"]))
            return
        for child in node.get("children", []):
            _walk(child)

    _walk(tree)
    return out


def test_multiclass_w9_shapes_and_ordering_check(multiclass_setup_local):
    bst, X, y = multiclass_setup_local
    mats = compute_matrices(bst, X, y, nfolds=3, t_points=12, random_state=0, multiclass="per_class")
    for k in range(3):
        assert mats[k].W9.shape == mats[k].W8.shape
        assert mats[k].W9.dtype == np.float32

    mats_stack = compute_matrices(bst, X, y, nfolds=3, t_points=12, random_state=0, multiclass="stack")
    mats_avg = compute_matrices(bst, X, y, nfolds=3, t_points=12, random_state=0, multiclass="avg")
    assert mats_stack.W9.shape == (X.shape[0] * 3, len(mats_stack.endpoints))
    assert mats_avg.W9.shape == (X.shape[0], len(mats_avg.endpoints))

    pred_leaf = bst.predict(xgb.DMatrix(X[:1]), pred_leaf=True, strict_shape=True)
    assert pred_leaf.shape[1] == int(bst.num_boosted_rounds())
    assert pred_leaf.shape[2] == 3

    dump_json = bst.get_dump(dump_format="json")
    first_round_tree_ids = [0, 1, 2]
    for class_idx, tree_id in enumerate(first_round_tree_ids):
        leaf_ids = _leaf_node_ids(dump_json[tree_id])
        observed_leaf = int(pred_leaf[0, 0, class_idx, 0])
        assert observed_leaf in leaf_ids
