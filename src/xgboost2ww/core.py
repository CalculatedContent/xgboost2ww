"""xgboost2ww: compute WeightWatcher-style correlation matrices for XGBoost.

This library computes the XGBoost "correlation" matrices we've been studying:
W1, W2, W7, W8, W9 derived from out-of-fold (OOF) margin increments along the
boosting trajectory.

Public API:
  - compute_matrices(...)
  - convert(...): optionally returns a torch.nn.Sequential with the chosen matrix
  - to_linear_layer(...): build a torch Linear layer from a matrix

Notes
-----
* The OOF procedure retrains models on each fold using the *same* training
  parameters as the provided model (as best as we can infer).
* Supports binary classification out-of-the-box.
* Supports multiclass when compute_matrices(..., multiclass=...) is explicitly
  provided as "per_class", "stack", or "avg".
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
    W9: np.ndarray
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


def _softmax_rows(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    shifted = M - M.max(axis=1, keepdims=True)
    expm = np.exp(shifted)
    den = np.sum(expm, axis=1, keepdims=True) + 1e-12
    return expm / den


def _stage_endpoints(num_rounds: int, t_points: int) -> np.ndarray:
    t_points = int(min(max(2, t_points), num_rounds))
    return np.unique(np.round(np.linspace(1, num_rounds, t_points)).astype(int))


def _margins_over_time(
    bst: "xgb.Booster",
    dX: "xgb.DMatrix",
    endpoints: np.ndarray,
    num_class: int,
    multiclass_output: bool,
) -> np.ndarray:
    nrows = dX.num_row()
    if not multiclass_output:
        F = np.empty((nrows, len(endpoints)), dtype=np.float32)
    else:
        F = np.empty((nrows, len(endpoints), num_class), dtype=np.float32)

    for j, t in enumerate(endpoints):
        pred = bst.predict(dX, iteration_range=(0, int(t)), output_margin=True)
        pred = np.asarray(pred, dtype=np.float32)
        if not multiclass_output:
            F[:, j] = pred.reshape(-1)
            continue

        if pred.ndim == 1:
            pred = pred.reshape(nrows, num_class)
        F[:, j, :] = pred

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

    learner_train_param = learner.get("learner_train_param", {})
    _update_from(learner_train_param, ("objective", "num_class"))

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


def _sum_tree_leaf_outputs_sq_from_json(tree_json: str) -> float:
    tree = json.loads(tree_json)

    def _walk(node: Mapping[str, Any]) -> float:
        if "leaf" in node:
            v = float(node["leaf"])
            return v * v
        total = 0.0
        for child in node.get("children", []):
            total += _walk(child)
        return total

    return _walk(tree)


def _classwise_block_gamma_diag(
    bst: "xgb.Booster",
    endpoints: np.ndarray,
    num_rounds: int,
    output_classes: int,
    reg_lambda: float,
) -> np.ndarray:
    """Compute blockwise stage regularizer mass using only L2 leaf penalty (lambda).

    This is the W9 v1 local-quadratic approximation and intentionally ignores
    the non-smooth L1 term (alpha).
    """
    dump_json = bst.get_dump(dump_format="json")
    ntrees = len(dump_json)
    if output_classes == 1:
        expected = num_rounds
        if ntrees != expected:
            raise ValueError(f"Expected {expected} trees for binary model, got {ntrees}.")
        leaf_mass = np.array([_sum_tree_leaf_outputs_sq_from_json(t) for t in dump_json], dtype=np.float64)
        classwise = leaf_mass.reshape(1, num_rounds)
    else:
        expected = num_rounds * output_classes
        if ntrees != expected:
            raise ValueError(
                f"Expected {expected} trees for multiclass model with {output_classes} classes, got {ntrees}."
            )
        leaf_mass = np.array([_sum_tree_leaf_outputs_sq_from_json(t) for t in dump_json], dtype=np.float64)
        classwise = np.empty((output_classes, num_rounds), dtype=np.float64)
        for k in range(output_classes):
            classwise[k] = leaf_mass[k::output_classes]

    block_start = np.concatenate(([0], endpoints[:-1]))
    gamma = np.empty((output_classes, len(endpoints)), dtype=np.float64)
    for j, (s, e) in enumerate(zip(block_start, endpoints)):
        gamma[:, j] = reg_lambda * classwise[:, int(s) : int(e)].sum(axis=1)
    return gamma


def _compute_W9_from_raw_oof(dF_oof: np.ndarray, m_final: np.ndarray, gamma_diag: np.ndarray) -> np.ndarray:
    p = _sigmoid(m_final)
    h = np.clip(p * (1.0 - p), 1e-6, None).astype(np.float32)
    A = _weighted_center_cols(dF_oof, h)
    gamma_diag = np.clip(np.asarray(gamma_diag, dtype=np.float32), 1e-6, None)
    return ((np.sqrt(h)[:, None] * A) / np.sqrt(gamma_diag)[None, :]).astype(np.float32)


def _out_of_fold_increments(
    model: "xgb.Booster",
    X: np.ndarray,
    y: np.ndarray,
    nfolds: int = 5,
    t_points: int = 160,
    random_state: int = 0,
    train_params: dict | None = None,
    num_boost_round: int | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, bool, np.ndarray]:
    """Compute OOF margin increments and final margins."""
    if y.ndim != 1:
        y = y.reshape(-1)
    N = X.shape[0]
    if N != y.shape[0]:
        raise ValueError(f"X and y have mismatched rows: {N} vs {y.shape[0]}")

    classes = np.unique(y)
    if classes.size == 0:
        raise ValueError("labels cannot be empty")

    if not np.array_equal(classes, np.arange(classes.size)):
        remap = {int(c): i for i, c in enumerate(classes.tolist())}
        y_encoded = np.array([remap[int(v)] for v in y], dtype=np.int32)
    else:
        y_encoded = y.astype(np.int32)

    K = int(classes.size)

    num_rounds = int(num_boost_round) if num_boost_round is not None else _infer_num_rounds(model)
    if num_rounds <= 0:
        raise ValueError("Could not infer num_boost_round from provided model.")

    params = dict(train_params) if train_params is not None else _infer_params(model)
    objective = str(params.get("objective", ""))
    num_class_param = int(params.get("num_class", K)) if params.get("num_class", K) is not None else K
    multiclass_output = bool(objective in {"multi:softprob", "multi:softmax"} and num_class_param == 2)
    if multiclass_output:
        params["num_class"] = 2

    if K > 2:
        params["objective"] = params.get("objective") or "multi:softprob"
        if params["objective"] not in {"multi:softprob", "multi:softmax"}:
            params["objective"] = "multi:softprob"
        params["num_class"] = int(params.get("num_class", K))
        multiclass_output = True

    endpoints = _stage_endpoints(num_rounds, t_points)
    T = len(endpoints)

    output_classes = (2 if K == 2 else K) if multiclass_output else 1

    if not multiclass_output:
        dF_oof = np.zeros((N, T), dtype=np.float32)
        m_final = np.zeros(N, dtype=np.float32)
        gamma_fold_weighted = np.zeros((1, T), dtype=np.float64)
    else:
        dF_oof = np.zeros((N, T, output_classes), dtype=np.float32)
        m_final = np.zeros((N, output_classes), dtype=np.float32)
        gamma_fold_weighted = np.zeros((output_classes, T), dtype=np.float64)

    reg_lambda = float(params.get("lambda", 1.0))

    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(N), y_encoded), start=1):
        if verbose:
            print(f"[xgboost2ww] Fold {fold}/{nfolds}: train={len(tr_idx)} val={len(va_idx)}")
        dtr = xgb.DMatrix(X[tr_idx], label=y_encoded[tr_idx])
        dva = xgb.DMatrix(X[va_idx], label=y_encoded[va_idx])

        bst = xgb.train(params=params, dtrain=dtr, num_boost_round=num_rounds, verbose_eval=False)

        Fva = _margins_over_time(
            bst, dva, endpoints, num_class=output_classes, multiclass_output=multiclass_output
        )
        dFva = np.empty_like(Fva)
        dFva[:, 0] = Fva[:, 0]
        dFva[:, 1:] = Fva[:, 1:] - Fva[:, :-1]

        dF_oof[va_idx] = dFva
        m_final[va_idx] = Fva[:, -1]

        fold_gamma = _classwise_block_gamma_diag(
            bst=bst,
            endpoints=endpoints,
            num_rounds=num_rounds,
            output_classes=output_classes,
            reg_lambda=reg_lambda,
        )
        gamma_fold_weighted += fold_gamma * (len(va_idx) / N)

    if not multiclass_output:
        gamma_diag_out = gamma_fold_weighted[0].astype(np.float32)
    else:
        gamma_diag_out = gamma_fold_weighted.astype(np.float32)

    return dF_oof, m_final, endpoints, output_classes, multiclass_output, gamma_diag_out


def _matrices_from_increments(
    dF: np.ndarray, m_final: np.ndarray, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1_centered = _center_cols(dF)
    svd = TruncatedSVD(n_components=1, random_state=random_state)
    svd.fit(W1_centered)
    v_top = svd.components_[0].astype(np.float32)
    v_top = v_top / (np.linalg.norm(v_top) + 1e-12)
    return _compute_W_matrices(W1_centered, v_top, m_final)


def compute_matrices(
    model: "xgb.Booster",
    data: Any,
    labels: Any,
    *,
    nfolds: int = 5,
    t_points: int = 160,
    random_state: int = 0,
    train_params: dict | None = None,
    num_boost_round: int | None = None,
    multiclass: Literal["error", "per_class", "stack", "avg"] = "error",
    verbose: bool = False,
) -> Union[Matrices, Dict[int, Matrices]]:
    """Compute W1/W2/W7/W8/W9 matrices for an XGBoost model using OOF margin increments.

    For multiclass labels (K > 2), set ``multiclass`` to one of:
      - "per_class": return ``Dict[int, Matrices]``
      - "stack": return one ``Matrices`` with class-wise matrices vertically stacked
      - "avg": return one ``Matrices`` with class-wise matrices averaged
    """
    if not isinstance(model, xgb.Booster):
        raise TypeError("model must be an xgboost.Booster")

    X = _as_numpy(data)
    y = _as_numpy(labels).astype(np.int32).reshape(-1)

    dF_oof, m_final, endpoints, K, multiclass_output, gamma_diag = _out_of_fold_increments(
        model,
        X,
        y,
        nfolds=nfolds,
        t_points=t_points,
        random_state=random_state,
        train_params=train_params,
        num_boost_round=num_boost_round,
        verbose=verbose,
    )

    if not multiclass_output:
        W1, W2, W7, W8 = _matrices_from_increments(dF_oof, m_final, random_state=random_state)
        W9 = _compute_W9_from_raw_oof(dF_oof=dF_oof, m_final=m_final, gamma_diag=gamma_diag)
        return Matrices(W1=W1, W2=W2, W7=W7, W8=W8, W9=W9, m_final=m_final.astype(np.float32), endpoints=endpoints)

    if multiclass == "error":
        raise ValueError(
            "Detected multiclass outputs. Set multiclass to one of: 'per_class', 'stack', or 'avg'."
        )

    p = _softmax_rows(m_final)
    per_class: Dict[int, Matrices] = {}
    for k in range(K):
        dFk = dF_oof[:, :, k]
        W1k = _center_cols(dFk)
        svd = TruncatedSVD(n_components=1, random_state=random_state)
        svd.fit(W1k)
        v_topk = svd.components_[0].astype(np.float32)
        v_topk = v_topk / (np.linalg.norm(v_topk) + 1e-12)

        W2k = _row_center(W1k)
        W7k = _center_cols(W1k - (W1k @ v_topk)[:, None] * v_topk[None, :])

        wk = (p[:, k] * (1.0 - p[:, k])).astype(np.float32)
        wk = np.clip(wk, 1e-6, None)
        sqrtwk = np.sqrt(wk).astype(np.float32)
        W7_wcent_k = _weighted_center_cols(W7k, wk)
        W8k = (sqrtwk[:, None] * W7_wcent_k).astype(np.float32)
        W9k = _compute_W9_from_raw_oof(dF_oof=dFk, m_final=m_final[:, k], gamma_diag=gamma_diag[k])

        per_class[k] = Matrices(
            W1=W1k.astype(np.float32),
            W2=W2k.astype(np.float32),
            W7=W7k.astype(np.float32),
            W8=W8k.astype(np.float32),
            W9=W9k.astype(np.float32),
            m_final=m_final[:, k].astype(np.float32),
            endpoints=endpoints,
        )

    if multiclass == "per_class":
        return per_class

    if multiclass == "stack":
        return Matrices(
            W1=np.vstack([per_class[k].W1 for k in range(K)]).astype(np.float32),
            W2=np.vstack([per_class[k].W2 for k in range(K)]).astype(np.float32),
            W7=np.vstack([per_class[k].W7 for k in range(K)]).astype(np.float32),
            W8=np.vstack([per_class[k].W8 for k in range(K)]).astype(np.float32),
            W9=np.vstack([per_class[k].W9 for k in range(K)]).astype(np.float32),
            m_final=m_final.astype(np.float32),
            endpoints=endpoints,
        )

    if multiclass == "avg":
        return Matrices(
            W1=np.mean(np.stack([per_class[k].W1 for k in range(K)], axis=0), axis=0).astype(np.float32),
            W2=np.mean(np.stack([per_class[k].W2 for k in range(K)], axis=0), axis=0).astype(np.float32),
            W7=np.mean(np.stack([per_class[k].W7 for k in range(K)], axis=0), axis=0).astype(np.float32),
            W8=np.mean(np.stack([per_class[k].W8 for k in range(K)], axis=0), axis=0).astype(np.float32),
            W9=np.mean(np.stack([per_class[k].W9 for k in range(K)], axis=0), axis=0).astype(np.float32),
            m_final=m_final.astype(np.float32),
            endpoints=endpoints,
        )

    raise ValueError("multiclass must be one of 'error', 'per_class', 'stack', or 'avg'")


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
    W: Literal["W1", "W2", "W7", "W8", "W9"] = "W7",
    nfolds: int = 5,
    t_points: int = 160,
    random_state: int = 0,
    train_params: dict | None = None,
    num_boost_round: int | None = None,
    multiclass: Literal["error", "per_class", "stack", "avg"] = "error",
    return_type: Literal["numpy", "torch"] = "torch",
    verbose: bool = False,
):
    """Compute matrices and return either the selected numpy matrix or a torch layer."""
    supported = ("W1", "W2", "W7", "W8", "W9")
    if W not in supported:
        raise ValueError(f"W must be one of {supported}; got {W!r}.")

    mats = compute_matrices(
        model,
        data,
        labels,
        nfolds=nfolds,
        t_points=t_points,
        random_state=random_state,
        train_params=train_params,
        num_boost_round=num_boost_round,
        multiclass=multiclass,
        verbose=verbose,
    )

    if multiclass == "per_class":
        if return_type == "torch":
            raise ValueError("convert(..., multiclass='per_class', return_type='torch') is unsupported; use return_type='numpy'.")
        out: Dict[int, np.ndarray] = {}
        for k, v in mats.items():
            if not hasattr(v, W):
                available = tuple(name for name in supported if hasattr(v, name))
                raise AttributeError(
                    f"Requested matrix {W!r} is unavailable on class {k}. "
                    f"This environment exposes {available}. "
                    "If you expected W9, upgrade/reinstall xgboost2ww from a revision that includes W9."
                )
            out[k] = getattr(v, W)
        return out

    if not hasattr(mats, W):
        available = tuple(name for name in supported if hasattr(mats, name))
        raise AttributeError(
            f"Requested matrix {W!r} is unavailable on Matrices. "
            f"This environment exposes {available}. "
            "If you expected W9, upgrade/reinstall xgboost2ww from a revision that includes W9."
        )

    Wmat = getattr(mats, W)
    if return_type == "numpy":
        return Wmat
    if return_type == "torch":
        layer = to_linear_layer(Wmat)
        import torch  # noqa: F401

        return torch.nn.Sequential(layer)
    raise ValueError("return_type must be 'numpy' or 'torch'")
