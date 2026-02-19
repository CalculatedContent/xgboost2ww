"""xgboost2ww: compute WeightWatcher-style correlation matrices for XGBoost.

This library computes the XGBoost "correlation" matrices we've been studying:
W1, W2, W7, W8 derived from out-of-fold (OOF) margin increments along the
boosting trajectory.

Public API:
  - compute_matrices(...)
  - convert(...): optionally returns a torch.nn.Sequential with the chosen matrix
  - to_linear_layer(...): build a torch Linear layer from a matrix

Notes
-----
* The OOF procedure retrains models on each fold using the *same* training
  parameters as the provided model (as best as we can infer).
* Supports binary classification out-of-the-box (labels 0/1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, Tuple, Union

import json
import re

import numpy as np

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    raise ImportError("xgboost2ww requires xgboost to be installed.") from e

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold


ArrayLike = Union[np.ndarray, "np.typing.NDArray[np.floating]"]


@dataclass(frozen=True)
class Matrices:
    """Container for computed matrices and intermediates."""

    W1: np.ndarray
    W2: np.ndarray
    W7: np.ndarray
    W8: np.ndarray
    m_final: np.ndarray
    endpoints: np.ndarray


def _as_numpy(X: Any) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    if hasattr(X, "values"):
        return np.asarray(X.values)
    return np.asarray(X)


def _center_cols(A: np.ndarray) -> np.ndarray:
    return A - A.mean(axis=0, keepdims=True)


def _row_center(A: np.ndarray) -> np.ndarray:
    return A - A.mean(axis=1, keepdims=True)


def _weighted_center_cols(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    denom = float(np.sum(w)) + 1e-12
    mu = (w[:, None] * A).sum(axis=0, keepdims=True) / denom
    return A - mu


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _stage_endpoints(num_rounds: int, t_points: int) -> np.ndarray:
    t_points = int(min(max(2, t_points), num_rounds))
    return np.unique(np.round(np.linspace(1, num_rounds, t_points)).astype(int))


def _margins_over_time(bst: "xgb.Booster", dX: "xgb.DMatrix", endpoints: np.ndarray) -> np.ndarray:
    F = np.empty((dX.num_row(), len(endpoints)), dtype=np.float32)
    for j, t in enumerate(endpoints):
        F[:, j] = bst.predict(dX, iteration_range=(0, int(t)), output_margin=True).astype(np.float32)
    return F


def _infer_num_rounds(model: "xgb.Booster") -> int:
    try:
        n = int(model.num_boosted_rounds())
        if n > 0:
            return n
    except Exception:
        pass

    attrs = {}
    try:
        attrs = model.attributes() or {}
    except Exception:
        attrs = {}

    for k in ("best_iteration", "best_ntree_limit", "num_boost_round"):
        if k in attrs and attrs[k] is not None:
            try:
                n = int(attrs[k])
                if n > 0:
                    return n
            except Exception:
                pass

    try:
        return len(model.get_dump())
    except Exception:
        return 0


def _infer_params(model: "xgb.Booster") -> Dict[str, Any]:
    """Infer training params from an existing Booster via save_config()."""
    cfg = json.loads(model.save_config())
    learner = cfg.get("learner", {})
    out: Dict[str, Any] = {}

    obj = learner.get("objective", {}).get("name", None)
    if obj:
        out["objective"] = obj

    gbm = learner.get("gradient_booster", {})
    booster_name = gbm.get("name", None)
    if booster_name:
        out["booster"] = booster_name

    def _cast(v: Any) -> Any:
        if isinstance(v, str):
            if re.fullmatch(r"-?\d+", v):
                return int(v)
            if re.fullmatch(r"-?\d+\.\d*", v):
                return float(v)
        return v

    def _update_from(d: Mapping[str, Any], keys: Iterable[str]) -> None:
        for k in keys:
            if k in d:
                out[k] = _cast(d[k])

    gen = learner.get("generic_param", {})
    _update_from(gen, ("seed", "nthread", "verbosity"))

    ttp = gbm.get("tree_train_param", {})
    _update_from(
        ttp,
        (
            "eta",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "lambda",
            "alpha",
            "max_delta_step",
        ),
    )
    gbt = gbm.get("gbtree_train_param", {})
    _update_from(gbt, ("num_parallel_tree",))

    out.setdefault("seed", 0)
    out.setdefault("verbosity", 0)
    out.setdefault("objective", "binary:logistic")
    out.setdefault("eval_metric", "logloss")
    return out


def _compute_W_matrices(
    W1_centered: np.ndarray, v_top: np.ndarray, m_oof_final: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Given centered W1, top right singular vector, and final margins, compute W1/W2/W7/W8."""
    W1 = W1_centered
    W2 = _row_center(W1)

    W7 = _center_cols(W1 - (W1 @ v_top)[:, None] * v_top[None, :])

    p = _sigmoid(m_oof_final)
    w = (p * (1.0 - p)).astype(np.float32)
    w = np.clip(w, 1e-6, None)
    sqrtw = np.sqrt(w).astype(np.float32)

    W7_wcent = _weighted_center_cols(W7, w)
    W8 = (sqrtw[:, None] * W7_wcent).astype(np.float32)
    return W1.astype(np.float32), W2.astype(np.float32), W7.astype(np.float32), W8.astype(np.float32)


def _out_of_fold_increments(
    model: "xgb.Booster",
    X: np.ndarray,
    y: np.ndarray,
    nfolds: int = 5,
    t_points: int = 160,
    random_state: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute OOF margin increments and final margins."""
    if y.ndim != 1:
        y = y.reshape(-1)
    N = X.shape[0]
    if N != y.shape[0]:
        raise ValueError(f"X and y have mismatched rows: {N} vs {y.shape[0]}")

    unique = np.unique(y)
    if not (np.array_equal(unique, [0, 1]) or np.array_equal(unique, [0]) or np.array_equal(unique, [1])):
        raise ValueError("Currently assumes binary labels encoded as 0/1.")

    num_rounds = _infer_num_rounds(model)
    if num_rounds <= 0:
        raise ValueError("Could not infer num_boost_round from provided model.")

    params = _infer_params(model)
    endpoints = _stage_endpoints(num_rounds, t_points)
    T = len(endpoints)

    dF_oof = np.zeros((N, T), dtype=np.float32)
    m_final = np.zeros(N, dtype=np.float32)

    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(N), y), start=1):
        if verbose:
            print(f"[xgboost2ww] Fold {fold}/{nfolds}: train={len(tr_idx)} val={len(va_idx)}")
        dtr = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X[va_idx], label=y[va_idx])

        bst = xgb.train(params=params, dtrain=dtr, num_boost_round=num_rounds, verbose_eval=False)

        Fva = _margins_over_time(bst, dva, endpoints)
        dFva = np.empty_like(Fva)
        dFva[:, 0] = Fva[:, 0]
        dFva[:, 1:] = Fva[:, 1:] - Fva[:, :-1]

        dF_oof[va_idx] = dFva
        m_final[va_idx] = Fva[:, -1]

    return dF_oof, m_final, endpoints


def compute_matrices(
    model: "xgb.Booster",
    data: Any,
    labels: Any,
    *,
    nfolds: int = 5,
    t_points: int = 160,
    random_state: int = 0,
    verbose: bool = False,
) -> Matrices:
    """Compute W1/W2/W7/W8 matrices for an XGBoost model using OOF margin increments."""
    if not isinstance(model, xgb.Booster):
        raise TypeError("model must be an xgboost.Booster")

    X = _as_numpy(data)
    y = _as_numpy(labels).astype(np.int32).reshape(-1)

    dF_oof, m_final, endpoints = _out_of_fold_increments(
        model, X, y, nfolds=nfolds, t_points=t_points, random_state=random_state, verbose=verbose
    )

    W1_centered = _center_cols(dF_oof)

    svd = TruncatedSVD(n_components=1, random_state=random_state)
    svd.fit(W1_centered)
    v_top = svd.components_[0].astype(np.float32)
    v_top = v_top / (np.linalg.norm(v_top) + 1e-12)

    W1, W2, W7, W8 = _compute_W_matrices(W1_centered, v_top, m_final)
    return Matrices(W1=W1, W2=W2, W7=W7, W8=W8, m_final=m_final.astype(np.float32), endpoints=endpoints)


def to_linear_layer(W: np.ndarray):
    """Build a torch.nn.Linear layer with weight=W and bias=False (torch is optional)."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise ImportError("to_linear_layer requires torch to be installed.") from e

    W = np.asarray(W, dtype=np.float32)
    out_dim, in_dim = W.shape
    linear = torch.nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(W))
    return linear


def convert(
    model: "xgb.Booster",
    data: Any,
    labels: Any,
    *,
    W: Literal["W1", "W2", "W7", "W8"] = "W7",
    nfolds: int = 5,
    t_points: int = 160,
    random_state: int = 0,
    return_type: Literal["numpy", "torch"] = "torch",
    verbose: bool = False,
):
    """Compute matrices and return either the selected numpy matrix or a torch layer."""
    mats = compute_matrices(
        model, data, labels, nfolds=nfolds, t_points=t_points, random_state=random_state, verbose=verbose
    )
    Wmat = getattr(mats, W)
    if return_type == "numpy":
        return Wmat
    if return_type == "torch":
        layer = to_linear_layer(Wmat)
        import torch  # noqa: F401
        return torch.nn.Sequential(layer)
    raise ValueError("return_type must be 'numpy' or 'torch'")
