"""xgboost2ww core matrix construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple, Union

import json
import re
import warnings

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
    W10: Optional[np.ndarray] = None
    W10_info: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class _BackendContext:
    name: Literal["numpy", "torch"]
    device: str
    torch: Any = None


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


def _resolve_backend_context(backend: Literal["auto", "numpy", "torch"], device: str) -> _BackendContext:
    if backend == "numpy":
        return _BackendContext(name="numpy", device="cpu", torch=None)

    torch_mod = None
    try:
        import torch as torch_mod  # type: ignore
    except Exception:
        if backend == "torch":
            raise ImportError("backend='torch' requested but torch is not installed.")
        return _BackendContext(name="numpy", device="cpu", torch=None)

    req_device = device
    if req_device == "auto":
        if torch_mod.cuda.is_available():
            req_device = "cuda"
        elif getattr(torch_mod.backends, "mps", None) is not None and torch_mod.backends.mps.is_available():
            req_device = "mps"
        else:
            req_device = "cpu"

    if req_device == "cuda" and not torch_mod.cuda.is_available():
        warnings.warn("Requested CUDA backend unavailable; falling back to CPU.")
        req_device = "cpu"
    if req_device == "mps":
        mps_ok = getattr(torch_mod.backends, "mps", None) is not None and torch_mod.backends.mps.is_available()
        if not mps_ok:
            warnings.warn("Requested MPS backend unavailable; falling back to CPU.")
            req_device = "cpu"

    return _BackendContext(name="torch", device=req_device, torch=torch_mod)


def _top_right_singular_vector_power(
    A: np.ndarray,
    backend_ctx: _BackendContext,
    n_iter: int = 8,
    tol: float = 1e-6,
    random_state: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_features = A.shape[1]
    v = rng.standard_normal(n_features).astype(np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)

    if backend_ctx.name == "torch" and backend_ctx.torch is not None:
        torch = backend_ctx.torch
        device = torch.device(backend_ctx.device)
        At = torch.as_tensor(np.asarray(A, dtype=np.float32), device=device)
        vt = torch.as_tensor(v.astype(np.float32), device=device)
        for _ in range(n_iter):
            v_prev = vt
            u = At @ vt
            u = u / (torch.linalg.norm(u) + 1e-12)
            vt = At.T @ u
            vt = vt / (torch.linalg.norm(vt) + 1e-12)
            if torch.linalg.norm(vt - v_prev).item() < tol:
                break
        out = vt.detach().cpu().numpy().astype(np.float32)
        return out / (np.linalg.norm(out) + 1e-12)

    for _ in range(n_iter):
        v_prev = v.copy()
        u = A @ v
        u = u / (np.linalg.norm(u) + 1e-12)
        v = A.T @ u
        v = v / (np.linalg.norm(v) + 1e-12)
        if np.linalg.norm(v - v_prev) < tol:
            break
    return v.astype(np.float32)


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
    dF: np.ndarray,
    m_final: np.ndarray,
    random_state: int,
    backend_ctx: Optional[_BackendContext] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1_centered = _center_cols(dF)
    if backend_ctx is not None and backend_ctx.name == "torch":
        v_top = _top_right_singular_vector_power(W1_centered, backend_ctx, random_state=random_state)
    else:
        svd = TruncatedSVD(n_components=1, random_state=random_state)
        svd.fit(W1_centered)
        v_top = svd.components_[0].astype(np.float32)
        v_top = v_top / (np.linalg.norm(v_top) + 1e-12)
    return _compute_W_matrices(W1_centered, v_top, m_final)


def _infer_objective_for_w10(
    model: "xgb.Booster",
    objective: Optional[str] = None,
    train_params: Optional[dict] = None,
) -> str:
    if objective is not None:
        return str(objective)
    if train_params is not None and train_params.get("objective") is not None:
        return str(train_params["objective"])
    params = _infer_params(model)
    return str(params.get("objective", "binary:logistic"))


def _compute_final_margin(model: "xgb.Booster", X: np.ndarray) -> np.ndarray:
    dX = xgb.DMatrix(X)
    pred = model.predict(dX, output_margin=True)
    out = np.asarray(pred)
    if out.ndim > 1:
        if out.shape[1] != 1:
            raise NotImplementedError("W10 currently supports binary logistic and squared-error regression only.")
        out = out[:, 0]
    return out.reshape(-1).astype(np.float64)


def _compute_hessian_weights(m_final: np.ndarray, objective: str, eps: float = 1e-6) -> np.ndarray:
    if objective == "binary:logistic":
        p = 1.0 / (1.0 + np.exp(-m_final))
        h = np.clip(p * (1.0 - p), eps, None)
        return h.astype(np.float64)
    if objective in {"reg:squarederror", "reg:linear"}:
        return np.ones_like(m_final, dtype=np.float64)
    if objective.startswith("multi:"):
        raise NotImplementedError(
            "W10 multiclass support is deferred: it requires classwise curvature and class-group leaf handling."
        )
    raise NotImplementedError(
        f"W10 currently supports objective='binary:logistic' and 'reg:squarederror' only; got {objective!r}."
    )


def _predict_leaf_assignments(model: "xgb.Booster", X: np.ndarray) -> np.ndarray:
    dX = xgb.DMatrix(X)
    try:
        leaves = model.predict(dX, pred_leaf=True, strict_shape=True)
    except TypeError:
        leaves = model.predict(dX, pred_leaf=True)
    leaves = np.asarray(leaves)
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    if leaves.ndim > 2:
        leaves = leaves.reshape(leaves.shape[0], -1)
    return leaves.astype(np.int64)


def _helmert_basis(L: int, dtype=np.float64) -> np.ndarray:
    if L <= 1:
        return np.zeros((L, 0), dtype=dtype)
    Q = np.zeros((L, L - 1), dtype=dtype)
    for j in range(1, L):
        Q[:j, j - 1] = 1.0 / np.sqrt(j * (j + 1.0))
        Q[j, j - 1] = -j / np.sqrt(j * (j + 1.0))
    return Q


def _remap_active_leaves(assignments: np.ndarray, weights: np.ndarray, support_tol: float):
    uniq, inv = np.unique(assignments, return_inverse=True)
    p = np.bincount(inv, weights=weights, minlength=uniq.size).astype(np.float64)
    active_mask = p > support_tol
    active_pos = np.flatnonzero(active_mask)
    if active_pos.size <= 1:
        return None
    lut = np.full(uniq.size, -1, dtype=np.int64)
    lut[active_pos] = np.arange(active_pos.size, dtype=np.int64)
    remapped = lut[inv]
    return remapped, p[active_pos], int(active_pos.size)


def _build_w10_tree_block(
    assignments: np.ndarray,
    m: np.ndarray,
    sqrtm: np.ndarray,
    backend_ctx: _BackendContext,
    eps: float,
    eig_tol: float,
    support_tol: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    remap = _remap_active_leaves(assignments, m, support_tol=support_tol)
    if remap is None:
        return None, {"active_leaves": 0, "retained_rank": 0, "kept": False}
    remapped, p_t, L_active = remap

    Q = _helmert_basis(L_active, dtype=np.float64)
    rows = Q[remapped]
    mu_t = p_t @ Q

    Rp = (Q * p_t[:, None]).T @ Q - np.outer(mu_t, mu_t)
    if backend_ctx.name == "torch" and backend_ctx.torch is not None:
        torch = backend_ctx.torch
        try:
            Rt = torch.as_tensor(Rp, dtype=torch.float64, device=backend_ctx.device)
            evals_t, evecs_t = torch.linalg.eigh(Rt)
            evals = evals_t.detach().cpu().numpy()
            evecs = evecs_t.detach().cpu().numpy()
        except Exception:
            evals, evecs = np.linalg.eigh(Rp)
    else:
        evals, evecs = np.linalg.eigh(Rp)

    keep = evals > eig_tol
    if not np.any(keep):
        return None, {"active_leaves": int(L_active), "retained_rank": 0, "kept": False}

    inv_sqrt = evecs[:, keep] @ np.diag(1.0 / np.sqrt(np.maximum(evals[keep], eps)))
    block = (rows - mu_t[None, :]) @ inv_sqrt
    block = sqrtm[:, None] * block
    block = block.astype(np.float32)
    return block, {"active_leaves": int(L_active), "retained_rank": int(np.sum(keep)), "kept": True}


def _concat_blocks(blocks: Iterable[np.ndarray], dtype: str = "float32") -> np.ndarray:
    blocks = [b for b in blocks if b is not None and b.size > 0]
    if not blocks:
        return np.zeros((0, 0), dtype=np.dtype(dtype))
    return np.concatenate(blocks, axis=1).astype(np.dtype(dtype), copy=False)


def compute_w10(
    model: "xgb.Booster",
    data: Any,
    labels: Any = None,
    *,
    objective: str | None = None,
    backend: Literal["auto", "numpy", "torch"] = "auto",
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
    dtype: Literal["float32", "float64"] = "float32",
    eps: float = 1e-6,
    eig_tol: float = 1e-8,
    support_tol: float = 1e-12,
    return_info: bool = False,
    return_gram: bool = False,
):
    if not isinstance(model, xgb.Booster):
        raise TypeError("model must be an xgboost.Booster")

    X = _as_numpy(data)
    N = X.shape[0]
    backend_ctx = _resolve_backend_context(backend=backend, device=device)
    objective_used = _infer_objective_for_w10(model=model, objective=objective)

    m_final = _compute_final_margin(model, X)
    h = _compute_hessian_weights(m_final, objective_used, eps=eps)
    m = h / (np.sum(h) + 1e-24)
    sqrtm = np.sqrt(m)

    leaf_assign = _predict_leaf_assignments(model, X)
    blocks = []
    leaf_counts = []
    contrast_ranks = []
    skipped_trees = 0
    for t in range(leaf_assign.shape[1]):
        block, stats = _build_w10_tree_block(
            leaf_assign[:, t], m=m, sqrtm=sqrtm, backend_ctx=backend_ctx, eps=eps, eig_tol=eig_tol, support_tol=support_tol
        )
        leaf_counts.append(stats["active_leaves"])
        contrast_ranks.append(stats["retained_rank"])
        if block is None:
            skipped_trees += 1
            continue
        blocks.append(block)

    W10 = _concat_blocks(blocks, dtype=dtype)
    if W10.shape[0] == 0:
        W10 = np.zeros((N, 0), dtype=np.dtype(dtype))

    if not np.isfinite(W10).all():
        raise FloatingPointError("W10 contains NaN or Inf values after stabilization.")

    G10 = None
    if return_gram:
        G10 = (W10.T @ W10).astype(np.dtype(dtype), copy=False)

    info = {
        "objective": objective_used,
        "backend": backend_ctx.name,
        "device": backend_ctx.device,
        "num_trees": int(leaf_assign.shape[1]),
        "active_trees": int(leaf_assign.shape[1] - skipped_trees),
        "skipped_trees": int(skipped_trees),
        "per_tree_active_leaf_counts": leaf_counts,
        "per_tree_retained_contrast_ranks": contrast_ranks,
        "total_width": int(W10.shape[1]),
        "computed_gram": bool(return_gram),
        "tree_block_widths": [b.shape[1] for b in blocks],
    }
    if return_gram and return_info:
        info["G10"] = G10

    if return_info and return_gram:
        return W10, info
    if return_info:
        return W10, info
    if return_gram:
        return W10, G10
    return W10


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
    include_W10: bool = False,
    w10_backend: Literal["auto", "numpy", "torch"] = "auto",
    w10_device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
    w10_return_gram: bool = False,
    verbose: bool = False,
) -> Union[Matrices, Dict[int, Matrices]]:
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

    sv_backend = _resolve_backend_context("auto", "auto")

    if not multiclass_output:
        W1, W2, W7, W8 = _matrices_from_increments(dF_oof, m_final, random_state=random_state, backend_ctx=sv_backend)
        W9 = _compute_W9_from_raw_oof(dF_oof=dF_oof, m_final=m_final, gamma_diag=gamma_diag)
        W10 = None
        W10_info = None
        if include_W10:
            if w10_return_gram:
                W10, info = compute_w10(
                    model,
                    X,
                    y,
                    objective=(train_params or {}).get("objective") if train_params else None,
                    backend=w10_backend,
                    device=w10_device,
                    return_info=True,
                    return_gram=True,
                )
                W10_info = info
            else:
                W10, W10_info = compute_w10(
                    model,
                    X,
                    y,
                    objective=(train_params or {}).get("objective") if train_params else None,
                    backend=w10_backend,
                    device=w10_device,
                    return_info=True,
                )
        return Matrices(
            W1=W1,
            W2=W2,
            W7=W7,
            W8=W8,
            W9=W9,
            m_final=m_final.astype(np.float32),
            endpoints=endpoints,
            W10=W10,
            W10_info=W10_info,
        )

    if multiclass == "error":
        raise ValueError(
            "Detected multiclass outputs. Set multiclass to one of: 'per_class', 'stack', or 'avg'."
        )

    p = _softmax_rows(m_final)
    per_class: Dict[int, Matrices] = {}
    for k in range(K):
        dFk = dF_oof[:, :, k]
        W1k, W2k, W7k, W8k = _matrices_from_increments(dFk, m_final[:, k], random_state=random_state, backend_ctx=sv_backend)
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
    W: Literal["W1", "W2", "W7", "W8", "W9", "W10"] = "W7",
    nfolds: int = 5,
    t_points: int = 160,
    random_state: int = 0,
    train_params: dict | None = None,
    num_boost_round: int | None = None,
    multiclass: Literal["error", "per_class", "stack", "avg"] = "error",
    return_type: Literal["numpy", "torch"] = "torch",
    verbose: bool = False,
):
    supported = ("W1", "W2", "W7", "W8", "W9", "W10")
    if W not in supported:
        raise ValueError(f"W must be one of {supported}; got {W!r}.")

    if W == "W10":
        Wmat = compute_w10(model, data, labels, objective=(train_params or {}).get("objective") if train_params else None)
        if return_type == "numpy":
            return Wmat
        if return_type == "torch":
            layer = to_linear_layer(Wmat)
            import torch  # noqa: F401

            return torch.nn.Sequential(layer)
        raise ValueError("return_type must be 'numpy' or 'torch'")

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
