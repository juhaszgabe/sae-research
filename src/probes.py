"""Feature selection and probes: DiM, linear probe, SAE probes, baselines, controls.

Every probe is a scikit-learn estimator, so `sklearn.base.clone` works and feature selection,
scaling and thresholding happen inside `fit` (R2).
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import joblib
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, StandardScaler

from .config import Config
from .utils import get_logger, write_json

log = get_logger("probes")

SELECTION_METHODS = ("mean_diff", "std_mean_diff", "mutual_info", "auroc")

# sklearn ≥ 1.8 deprecates `penalty` in favour of `l1_ratio`; detected once.
_PENALTY_DEPRECATED = inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"


# --------------------------------------------------------------------------- helpers (§12.1)

def make_logreg(penalty: Literal["l1", "l2"], C: float, max_iter: int = 5000, seed: int = 0) -> LogisticRegression:
    """L2 → solver lbfgs; L1 → solver liblinear; class_weight="balanced", random_state=seed.
    Uses l1_ratio (1.0/0.0) instead of `penalty` if the installed sklearn deprecates `penalty`
    (detect once via inspect.signature)."""
    if penalty not in ("l1", "l2"):
        raise ValueError(f"penalty must be 'l1' or 'l2', got {penalty!r}")
    kwargs: dict[str, Any] = dict(
        C=float(C), solver="lbfgs" if penalty == "l2" else "liblinear", max_iter=max_iter,
        class_weight="balanced", random_state=seed,
    )
    if _PENALTY_DEPRECATED:
        kwargs["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    else:
        kwargs["penalty"] = penalty
    return LogisticRegression(**kwargs)


def balanced_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy on the given (training) scores. Every probe sets
    self.threshold_ with this after fit — same rule for all methods. Predict 1 iff score > threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(int)
    order = np.argsort(scores, kind="stable")
    s, yy = scores[order], y[order]
    n, n_pos = len(s), int(yy.sum())
    n_neg = n - n_pos
    if n == 0 or n_pos == 0 or n_neg == 0:
        return float(np.median(s)) if n else 0.0
    # cut k: rows s[k:] are predicted positive
    tn = np.concatenate([[0], np.cumsum(yy == 0)])
    tp = n_pos - np.concatenate([[0], np.cumsum(yy == 1)])
    bal = (tp / n_pos + tn / n_neg) / 2
    valid = np.concatenate([[True], s[1:] > s[:-1], [True]])
    bal = np.where(valid, bal, -np.inf)
    k = int(np.argmax(bal))
    if k == 0:
        return float(s[0] - 1.0)
    if k == n:
        return float(s[-1] + 1.0)
    return float((s[k - 1] + s[k]) / 2)


def _rows(X: Any, idx: np.ndarray) -> Any:
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _class_stats(X: Any, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(μ₁, μ₀, σ₁², σ₀²) per column for a dense array or csr matrix."""
    out = []
    for cls in (1, 0):
        sub = X[y == cls]
        if sp.issparse(sub):
            mean = np.asarray(sub.mean(axis=0)).ravel()
            sq = np.asarray(sub.multiply(sub).mean(axis=0)).ravel()
            var = np.maximum(sq - mean**2, 0.0)
        else:
            sub = np.asarray(sub, dtype=np.float64)
            mean, var = sub.mean(axis=0), sub.var(axis=0)
        out.append((mean.astype(np.float64), var.astype(np.float64)))
    (m1, v1), (m0, v0) = out
    return m1, m0, v1, v0


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("zero vector has no direction")
    return (v / norm).astype(np.float32)


# --------------------------------------------------------------------------- FeatureSelector (§12.2)

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Ranks features by a univariate score on the fit rows and keeps the top k.

    `sparse=True` (SAE codes) applies the firing-rate eligibility filter first."""

    def __init__(self, k: int | None = None, method: str = "mean_diff",
                 min_firing_count: int = 3, min_firing_frac: float = 0.01,
                 max_firing_frac: float = 0.9, sparse: bool = True):
        self.k = k
        self.method = method
        self.min_firing_count = min_firing_count
        self.min_firing_frac = min_firing_frac
        self.max_firing_frac = max_firing_frac
        self.sparse = sparse

    def fit(self, X, y):
        if self.method not in SELECTION_METHODS:
            raise ValueError(f"unknown selection method {self.method!r} (available: {SELECTION_METHODS})")
        y = np.asarray(y).astype(int)
        if sp.issparse(X):
            X = X.tocsr()
        n, n_features = X.shape
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        if n1 == 0 or n0 == 0:
            raise ValueError("FeatureSelector needs both classes in the fit rows")

        fire1 = np.asarray((X[y == 1] > 0).sum(axis=0)).ravel().astype(np.float64)
        fire0 = np.asarray((X[y == 0] > 0).sum(axis=0)).ravel().astype(np.float64)
        if self.sparse:
            count = fire1 + fire0
            min_count = max(self.min_firing_count, math.ceil(self.min_firing_frac * n))
            eligible = (count >= min_count) & (count / n <= self.max_firing_frac)
        else:
            eligible = np.ones(n_features, dtype=bool)
        self.eligible_ = np.flatnonzero(eligible)

        m1, m0, v1, v0 = _class_stats(X, y)
        mean_diff = np.abs(m1 - m0)
        if self.method == "mean_diff":
            scores = mean_diff
        elif self.method == "std_mean_diff":
            scores = mean_diff / (np.sqrt((v1 + v0) / 2) + 1e-8)
        elif self.method == "mutual_info":
            scores = _mutual_info(fire1, fire0, n1, n0)
        else:
            scores = np.zeros(n_features)
            if len(self.eligible_):
                cols = X[:, self.eligible_]
                cols = cols.toarray() if sp.issparse(cols) else np.asarray(cols)
                ranks = rankdata(cols, axis=0, method="average")
                u = ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2
                scores[self.eligible_] = np.abs(u / (n1 * n0) - 0.5)

        # score desc, ties → mean_diff desc, then index asc
        e = self.eligible_
        ranked = e[np.lexsort((e, -mean_diff[e], -scores[e]))]
        self.ranked_ = ranked
        self.scores_ = np.where(eligible, scores, np.nan)
        signs = np.sign(m1 - m0).astype(np.int8)
        if self.k is None:
            self.selected_ = self.eligible_
        else:
            self.selected_ = ranked[: self.k]
            if len(self.selected_) < self.k:
                log.warning("only %d eligible features for k=%d; padding with zero columns", len(self.selected_), self.k)
        self.signs_ = signs[self.selected_]
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        """k int → dense float32 [N, k] in selected_ order; k None → csr[:, eligible_]."""
        if self.k is None:
            return X[:, self.eligible_]
        cols = X[:, self.selected_]
        cols = cols.toarray() if sp.issparse(cols) else np.asarray(cols)
        out = np.zeros((X.shape[0], self.k), dtype=np.float32)
        out[:, : cols.shape[1]] = cols
        return out


def _mutual_info(fire1: np.ndarray, fire0: np.ndarray, n1: int, n0: int) -> np.ndarray:
    """Exact MI (nats) between 1[x > 0] and y from the 2×2 table, per feature."""
    n = n1 + n0
    mi = np.zeros_like(fire1)
    cells = ((fire1, n1), (fire0, n0), (n1 - fire1, n1), (n0 - fire0, n0))
    col_tot = {True: fire1 + fire0, False: n - fire1 - fire0}
    for i, (count, n_cls) in enumerate(cells):
        tot = col_tot[i < 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (count / n) * np.log((count * n) / (tot * n_cls))
        mi += np.where(count > 0, term, 0.0)
    return np.maximum(mi, 0.0)


# --------------------------------------------------------------------------- probes (§12.3)

class _Probe(ClassifierMixin, BaseEstimator):
    """Shared predict/threshold logic. decision_function: higher = behavior more likely."""

    def _finish(self, X, y, n_features_used: int):
        self.classes_ = np.array([0, 1])
        self.n_features_used_ = int(n_features_used)
        self.threshold_ = balanced_threshold(self.decision_function(X), y)
        return self

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) > self.threshold_).astype(int)


class DiffMeansProbe(_Probe):
    """Difference in means: w = μ₁ − μ₀, score X·w − w·(μ₁+μ₀)/2."""

    def fit(self, X, y, aux=None):
        y = np.asarray(y).astype(int)
        m1, m0, _, _ = _class_stats(np.asarray(X), y)
        self.w_ = m1 - m0
        self.b_ = float(self.w_ @ (m1 + m0) / 2)
        return self._finish(X, y, len(self.w_))

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.w_ - self.b_


class LogRegProbe(_Probe):
    """StandardScaler → L2 logistic regression on raw activations."""

    def __init__(self, C: float = 1.0, max_iter: int = 5000, seed: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X, y, aux=None):
        y = np.asarray(y).astype(int)
        self.scaler_ = StandardScaler().fit(X)
        self.clf_ = make_logreg("l2", self.C, self.max_iter, self.seed).fit(self.scaler_.transform(X), y)
        return self._finish(X, y, np.shape(X)[1])

    def decision_function(self, X) -> np.ndarray:
        return self.clf_.decision_function(self.scaler_.transform(X)).astype(np.float64)


class _SelectedLogReg(_Probe):
    """FeatureSelector → StandardScaler → logistic regression (shared by the top-k probes)."""

    _penalty = "l1"

    def _make_selector(self) -> FeatureSelector:
        raise NotImplementedError

    def fit(self, X, y, aux=None):
        y = np.asarray(y).astype(int)
        self.selector_ = self._make_selector().fit(X, y)
        Z = self.selector_.transform(X)
        self.scaler_ = StandardScaler().fit(Z)
        self.clf_ = make_logreg(self._penalty, self.C, self.max_iter, self.seed).fit(self.scaler_.transform(Z), y)
        self.selected_ = self.selector_.selected_
        self.signs_ = self.selector_.signs_
        self.selector_scores_ = self.selector_.scores_[self.selected_]
        return self._finish(X, y, len(self.selected_))

    def decision_function(self, X) -> np.ndarray:
        Z = self.scaler_.transform(self.selector_.transform(X))
        return self.clf_.decision_function(Z).astype(np.float64)


class DenseTopKProbe(_SelectedLogReg):
    """Dimension-matched control: top-k raw neurons by standardized mean difference → L2 logreg."""

    _penalty = "l2"

    def __init__(self, k: int = 16, C: float = 1.0, max_iter: int = 5000, seed: int = 0):
        self.k = k
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    def _make_selector(self) -> FeatureSelector:
        return FeatureSelector(k=self.k, method="std_mean_diff", sparse=False)


class SaeTopKProbe(_SelectedLogReg):
    """Kantamneni et al.: top-k SAE features by `selection` → L1 logreg."""

    _penalty = "l1"

    def __init__(self, k: int = 16, C: float = 1.0, selection: str = "mean_diff", min_firing_count: int = 3,
                 min_firing_frac: float = 0.01, max_firing_frac: float = 0.9, max_iter: int = 5000, seed: int = 0):
        self.k = k
        self.C = C
        self.selection = selection
        self.min_firing_count = min_firing_count
        self.min_firing_frac = min_firing_frac
        self.max_firing_frac = max_firing_frac
        self.max_iter = max_iter
        self.seed = seed

    def _make_selector(self) -> FeatureSelector:
        return FeatureSelector(self.k, self.selection, self.min_firing_count, self.min_firing_frac,
                               self.max_firing_frac, sparse=True)


class SaeAllProbe(_Probe):
    """All eligible SAE features → MaxAbsScaler → L1 logreg (liblinear, sparse)."""

    def __init__(self, C: float = 1.0, min_firing_count: int = 3, min_firing_frac: float = 0.01,
                 max_firing_frac: float = 0.9, max_iter: int = 5000, seed: int = 0):
        self.C = C
        self.min_firing_count = min_firing_count
        self.min_firing_frac = min_firing_frac
        self.max_firing_frac = max_firing_frac
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X, y, aux=None):
        y = np.asarray(y).astype(int)
        self.selector_ = FeatureSelector(None, "mean_diff", self.min_firing_count, self.min_firing_frac,
                                         self.max_firing_frac, sparse=True).fit(X, y)
        self.empty_ = len(self.selector_.eligible_) == 0
        if self.empty_:
            log.warning("sae_all: no eligible features; scores are constant")
            self.selected_ = np.array([], dtype=int)
            self.signs_ = np.array([], dtype=np.int8)
            self.selector_scores_ = np.array([])
            return self._finish(X, y, 0)
        Z = sp.csr_matrix(self.selector_.transform(X))
        self.scaler_ = MaxAbsScaler().fit(Z)
        self.clf_ = make_logreg("l1", self.C, self.max_iter, self.seed).fit(self.scaler_.transform(Z), y)
        coef = self.clf_.coef_.ravel()
        used = np.flatnonzero(coef)
        used = used[np.argsort(-np.abs(coef[used]), kind="stable")]
        self.selected_ = self.selector_.eligible_[used]
        self.signs_ = np.sign(coef[used]).astype(np.int8)
        self.selector_scores_ = self.selector_.scores_[self.selected_]
        return self._finish(X, y, len(used))

    def decision_function(self, X) -> np.ndarray:
        if self.empty_:
            return np.zeros(X.shape[0], dtype=np.float64)
        Z = self.scaler_.transform(sp.csr_matrix(self.selector_.transform(X)))
        return self.clf_.decision_function(Z).astype(np.float64)


class SaeSingleFeatureProbe(_Probe):
    """Best single eligible SAE feature; score = sign · x_j."""

    def __init__(self, selection: str = "mean_diff", min_firing_count: int = 3,
                 min_firing_frac: float = 0.01, max_firing_frac: float = 0.9):
        self.selection = selection
        self.min_firing_count = min_firing_count
        self.min_firing_frac = min_firing_frac
        self.max_firing_frac = max_firing_frac

    def fit(self, X, y, aux=None):
        y = np.asarray(y).astype(int)
        self.selector_ = FeatureSelector(1, self.selection, self.min_firing_count, self.min_firing_frac,
                                         self.max_firing_frac, sparse=True).fit(X, y)
        self.selected_ = self.selector_.selected_
        self.signs_ = self.selector_.signs_
        self.selector_scores_ = self.selector_.scores_[self.selected_]
        return self._finish(X, y, len(self.selected_))

    def decision_function(self, X) -> np.ndarray:
        x = self.selector_.transform(X)[:, 0].astype(np.float64)
        sign = float(self.signs_[0]) if len(self.signs_) else 0.0
        return (sign or 1.0) * x


class RefusalDirectionProbe(_Probe):
    """Arditi et al. direction from the PROMPT CATEGORY (not the behavior label):
    r = mean(harmful) − mean(harmless); borderline_safe rows are ignored. Score X·r/‖r‖."""

    def fit(self, X, y, aux=None):
        if aux is None:
            raise ValueError("RefusalDirectionProbe.fit needs aux = prompt categories")
        X = np.asarray(X, dtype=np.float64)
        aux = np.asarray(aux)
        harmful, harmless = aux == "harmful", aux == "harmless"
        if not harmful.any() or not harmless.any():
            raise ValueError("RefusalDirectionProbe needs both 'harmful' and 'harmless' rows in the fit set")
        r = X[harmful].mean(axis=0) - X[harmless].mean(axis=0)
        self.r_ = r / np.linalg.norm(r)
        return self._finish(X, np.asarray(y).astype(int), X.shape[1])

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.r_


class CategoryOnlyProbe(_Probe):
    """one-hot(prompt_category, source) → L2 logreg C=1: how predictable the behavior is from
    the prompt type alone. X is a DataFrame of metadata columns."""

    def __init__(self, columns: tuple[str, ...] = ("prompt_category", "source"), max_iter: int = 5000, seed: int = 0):
        self.columns = columns
        self.max_iter = max_iter
        self.seed = seed

    def _meta(self, X):
        return X[list(self.columns)].astype(str).to_numpy()

    def fit(self, X, y, aux=None):
        y = np.asarray(y).astype(int)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore").fit(self._meta(X))
        Z = self.encoder_.transform(self._meta(X))
        self.clf_ = make_logreg("l2", 1.0, self.max_iter, self.seed).fit(Z, y)
        return self._finish(X, y, Z.shape[1])

    def decision_function(self, X) -> np.ndarray:
        return self.clf_.decision_function(self.encoder_.transform(self._meta(X))).astype(np.float64)


class ShuffledLabelProbe(_Probe):
    """Control: fits `base` on permuted labels (expected AUROC ≈ 0.5)."""

    def __init__(self, base: BaseEstimator = None, seed: int = 0):
        self.base = base
        self.seed = seed

    def fit(self, X, y, aux=None):
        y = np.asarray(y).astype(int)
        y_perm = np.random.RandomState(self.seed % (2**32)).permutation(y)
        self.base_ = clone(self.base)
        if aux is not None:
            self.base_.fit(X, y_perm, aux=aux)
        else:
            self.base_.fit(X, y_perm)
        self.classes_ = np.array([0, 1])
        self.n_features_used_ = self.base_.n_features_used_
        self.threshold_ = self.base_.threshold_
        return self

    def decision_function(self, X) -> np.ndarray:
        return self.base_.decision_function(X)


class RandomFeatureProbe(_Probe):
    """Control: k random eligible SAE features → scaler → L1 logreg (as sae_topk)."""

    def __init__(self, k: int = 16, C: float = 1.0, min_firing_count: int = 3, min_firing_frac: float = 0.01,
                 max_firing_frac: float = 0.9, max_iter: int = 5000, seed: int = 0):
        self.k = k
        self.C = C
        self.min_firing_count = min_firing_count
        self.min_firing_frac = min_firing_frac
        self.max_firing_frac = max_firing_frac
        self.max_iter = max_iter
        self.seed = seed

    def _features(self, X) -> np.ndarray:
        cols = X[:, self.selected_]
        return (cols.toarray() if sp.issparse(cols) else np.asarray(cols)).astype(np.float32)

    def fit(self, X, y, aux=None):
        y = np.asarray(y).astype(int)
        selector = FeatureSelector(None, "mean_diff", self.min_firing_count, self.min_firing_frac,
                                   self.max_firing_frac, sparse=True).fit(X, y)
        eligible = selector.eligible_
        if len(eligible) == 0:
            raise ValueError("RandomFeatureProbe: no eligible features")
        rng = np.random.RandomState(self.seed % (2**32))
        self.selected_ = np.sort(rng.choice(eligible, size=min(self.k, len(eligible)), replace=False))
        Z = self._features(X)
        self.scaler_ = StandardScaler().fit(Z)
        self.clf_ = make_logreg("l1", self.C, self.max_iter, self.seed).fit(self.scaler_.transform(Z), y)
        return self._finish(X, y, len(self.selected_))

    def decision_function(self, X) -> np.ndarray:
        return self.clf_.decision_function(self.scaler_.transform(self._features(X))).astype(np.float64)


SAE_PROBES = (SaeTopKProbe, SaeAllProbe, SaeSingleFeatureProbe, RandomFeatureProbe)


# --------------------------------------------------------------------------- method table

@dataclass
class MethodSpec:
    name: str
    make: Callable[..., BaseEstimator]  # make(**hparams)
    grid: list[dict]  # [{}] if no hyperparameters
    input: Literal["dense", "sae", "meta"]
    needs_aux: bool = False


class _Factory:
    """Picklable `make(**hparams)`: class + fixed keyword arguments (joblib needs to pickle specs)."""

    def __init__(self, cls: type, **fixed: Any):
        self.cls = cls
        self.fixed = fixed

    def __call__(self, **hparams: Any) -> BaseEstimator:
        return self.cls(**{**self.fixed, **hparams})


class _ShuffledFactory:
    def __init__(self, base: Callable[..., BaseEstimator], seed: int):
        self.base = base
        self.seed = seed

    def __call__(self, **hparams: Any) -> BaseEstimator:
        return ShuffledLabelProbe(self.base(**hparams), seed=self.seed)


def method_specs(cfg: Config, behavior: str) -> dict[str, MethodSpec]:
    """Builds the table above from cfg probes.*; drops refusal_dir unless behavior == 'refusal'."""
    pc = cfg["probes"]
    seed, max_iter = cfg["seed"], pc["max_iter"]
    firing = dict(min_firing_count=pc["min_firing_count"], min_firing_frac=pc["min_firing_frac"],
                  max_firing_frac=pc["max_firing_frac"])
    k_grid = [int(k) for k in pc["k_grid"]]

    table = {
        "dim": MethodSpec("dim", _Factory(DiffMeansProbe), [{}], "dense"),
        "logreg": MethodSpec("logreg", _Factory(LogRegProbe, max_iter=max_iter, seed=seed),
                             [{"C": float(c)} for c in pc["logreg_C"]], "dense"),
        "dense_topk": MethodSpec("dense_topk", _Factory(DenseTopKProbe, max_iter=max_iter, seed=seed),
                                 [{"k": k, "C": float(c)} for k in k_grid for c in pc["dense_topk_C"]], "dense"),
        "sae_topk": MethodSpec("sae_topk", _Factory(SaeTopKProbe, selection=pc["selection"], max_iter=max_iter,
                                                    seed=seed, **firing),
                               [{"k": k, "C": float(c)} for k in k_grid for c in pc["sae_C"]], "sae"),
        "sae_all": MethodSpec("sae_all", _Factory(SaeAllProbe, max_iter=max_iter, seed=seed, **firing),
                              [{"C": float(c)} for c in pc["sae_C"]], "sae"),
        "sae_single": MethodSpec("sae_single", _Factory(SaeSingleFeatureProbe, selection=pc["selection"], **firing),
                                 [{}], "sae"),
        "refusal_dir": MethodSpec("refusal_dir", _Factory(RefusalDirectionProbe), [{}], "dense", needs_aux=True),
        "category_only": MethodSpec("category_only", _Factory(CategoryOnlyProbe, max_iter=max_iter, seed=seed),
                                    [{}], "meta"),
    }
    unknown = [m for m in pc["methods"] if m not in table]
    if unknown:
        raise ValueError(f"unknown probe methods {unknown} (available: {sorted(table)})")
    return {m: table[m] for m in pc["methods"] if m != "refusal_dir" or behavior == "refusal"}


def shuffled_spec(spec: MethodSpec, seed: int) -> MethodSpec:
    """Shuffled-label control of `spec` (same grid, same input)."""
    return MethodSpec(f"shuffled_{spec.name}", _ShuffledFactory(spec.make, seed), spec.grid, spec.input, spec.needs_aux)


def random_feature_spec(cfg: Config, k: int, seed: int) -> MethodSpec:
    """RandomFeatureProbe control with fixed k and the sae_C grid."""
    pc = cfg["probes"]
    make = _Factory(RandomFeatureProbe, k=int(k), max_iter=pc["max_iter"], seed=seed,
                    min_firing_count=pc["min_firing_count"], min_firing_frac=pc["min_firing_frac"],
                    max_firing_frac=pc["max_firing_frac"])
    return MethodSpec("random_features", make, [{"C": float(c)} for c in pc["sae_C"]], "sae")


# --------------------------------------------------------------------------- persistence / directions

def save_probe(probe, path: Path, meta: dict) -> None:
    """joblib + path.with_suffix(".json")"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(probe, tmp)
    tmp.replace(path)
    write_json(path.with_suffix(".json"), meta)


def load_probe(path: Path) -> tuple[BaseEstimator, dict]:
    path = Path(path)
    with open(path.with_suffix(".json"), encoding="utf-8") as f:
        meta = json.load(f)
    return joblib.load(path), meta


def raw_direction(probe) -> np.ndarray:
    """Unit vector in raw activation space: DiM → w; LogReg → coef/scaler.scale_; RefusalDirection → r.
    Used by steering.py."""
    if isinstance(probe, DiffMeansProbe):
        return _unit(probe.w_)
    if isinstance(probe, LogRegProbe):
        return _unit(probe.clf_.coef_.ravel() / probe.scaler_.scale_)
    if isinstance(probe, RefusalDirectionProbe):
        return _unit(probe.r_)
    raise TypeError(f"no raw-space direction for {type(probe).__name__}")
