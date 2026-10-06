"""probes.py: feature selection, probe classes, method table, persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src import probes as P


def _gaussians(n: int = 400, d: int = 16, shift: float = 4.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    y = rng.permutation(np.repeat([0, 1], n // 2))
    X = rng.randn(n, d).astype(np.float32)
    X[:, 0] += shift * y
    X[:, 1] -= shift * y
    return X, y


# --------------------------------------------------------------------------- helpers

def test_make_logreg_solvers():
    l2, l1 = P.make_logreg("l2", 0.5, max_iter=123, seed=4), P.make_logreg("l1", 0.05)
    assert (l2.solver, l1.solver) == ("lbfgs", "liblinear")
    assert l2.C == 0.5 and l2.max_iter == 123 and l2.random_state == 4 and l2.class_weight == "balanced"
    X, y = _gaussians(200, 6)
    assert l2.fit(X, y).score(X, y) > 0.95
    sparse_fit = l1.fit(np.hstack([X, np.random.RandomState(1).randn(200, 40)]), y)
    assert (sparse_fit.coef_ == 0).sum() > 0  # L1 actually zeroes coefficients
    with pytest.raises(ValueError):
        P.make_logreg("elasticnet", 1.0)


def test_balanced_threshold():
    assert P.balanced_threshold(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 1, 1])) == 1.5
    # one positive outlier at the bottom: the best cut is still between 1 and 2
    assert P.balanced_threshold(np.array([-5.0, 0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1, 0, 0, 1, 1, 1])) == 1.5
    # inverted scores: predicting everything negative/positive is the best a threshold can do
    t = P.balanced_threshold(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1, 1, 0, 0]))
    assert t < 0.0 or t > 3.0
    assert np.isfinite(P.balanced_threshold(np.array([1.0, 2.0]), np.array([1, 1])))  # single class


# --------------------------------------------------------------------------- FeatureSelector

def _codes() -> tuple[sp.csr_matrix, np.ndarray]:
    """10 rows (5 per class), 6 features with hand-set firing patterns."""
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    X = np.zeros((10, 6), dtype=np.float32)
    X[:5, 0] = 2.0                 # fires only for class 1 (5 rows)           mean diff 2.0
    X[:, 1] = 1.0                  # fires everywhere → above max_firing_frac  ineligible
    X[0, 2] = 9.0                  # fires once → below min_firing_count       ineligible
    X[5:8, 3] = 1.0                # fires for 3 negatives                     mean diff 0.6, sign -1
    X[[0, 1, 5], 4] = [3.0, 3.0, 3.0]  # 2 positives, 1 negative               mean diff 0.6, sign +1
    X[[0, 1, 2, 5, 6, 7], 5] = 1.0     # balanced                              mean diff 0.0
    return sp.csr_matrix(X), y


def test_selector_eligibility_and_ranking():
    X, y = _codes()
    sel = P.FeatureSelector(k=3, method="mean_diff", min_firing_count=3, min_firing_frac=0.01, max_firing_frac=0.9).fit(X, y)
    assert sel.eligible_.tolist() == [0, 3, 4, 5]
    assert sel.selected_.tolist() == [0, 3, 4]  # score desc, ties by index
    assert sel.signs_.tolist() == [1, -1, 1]
    assert np.isnan(sel.scores_[[1, 2]]).all() and sel.scores_[0] == pytest.approx(2.0)
    Z = sel.transform(X)
    assert Z.shape == (10, 3) and Z.dtype == np.float32 and not sp.issparse(Z)
    assert np.array_equal(Z, X.toarray()[:, [0, 3, 4]])


def test_selector_pads_with_zero_columns_when_few_eligible():
    X, y = _codes()
    sel = P.FeatureSelector(k=6).fit(X, y)
    Z = sel.transform(X)
    assert len(sel.selected_) == 4 and Z.shape == (10, 6)
    assert not Z[:, 4:].any()


def test_selector_k_none_returns_eligible_sparse_columns():
    X, y = _codes()
    sel = P.FeatureSelector(k=None).fit(X, y)
    Z = sel.transform(X)
    assert sp.issparse(Z) and Z.shape == (10, 4)
    assert sel.selected_.tolist() == [0, 3, 4, 5]


def test_selector_dense_mode_has_no_firing_filter():
    X, y = _gaussians(200, 8)
    sel = P.FeatureSelector(k=2, method="std_mean_diff", sparse=False).fit(X, y)
    assert len(sel.eligible_) == 8 and set(sel.selected_.tolist()) == {0, 1}
    assert dict(zip(sel.selected_.tolist(), sel.signs_.tolist())) == {0: 1, 1: -1}


@pytest.mark.parametrize("method", ["mean_diff", "mutual_info", "auroc"])
def test_every_selection_method_finds_the_planted_feature(synthetic_dataset, method):
    ds = synthetic_dataset
    sel = P.FeatureSelector(k=1, method=method).fit(ds.codes, ds.y)
    assert sel.selected_.tolist() == [ds.planted_feature]
    assert sel.signs_.tolist() == [1]


def test_selector_rejects_unknown_method_and_single_class():
    X, y = _codes()
    with pytest.raises(ValueError):
        P.FeatureSelector(method="chi2").fit(X, y)
    with pytest.raises(ValueError):
        P.FeatureSelector().fit(X, np.ones(10, dtype=int))


# --------------------------------------------------------------------------- probes

def test_dim_and_logreg_on_separable_gaussians():
    X, y = _gaussians(seed=0)
    X_test, y_test = _gaussians(seed=1)
    for probe in (P.DiffMeansProbe(), P.LogRegProbe(C=1.0)):
        probe.fit(X, y)
        scores = probe.decision_function(X_test)
        assert scores.shape == (400,) and scores.dtype == np.float64
        assert roc_auc_score(y_test, scores) > 0.95
        assert (probe.predict(X_test) == y_test).mean() > 0.95
        assert np.isfinite(probe.threshold_) and probe.n_features_used_ == 16


def test_dim_scores_are_centered_between_the_class_means():
    X, y = _gaussians()
    probe = P.DiffMeansProbe().fit(X, y)
    midpoint = (X[y == 1].mean(0) + X[y == 0].mean(0)) / 2
    assert probe.decision_function(midpoint[None])[0] == pytest.approx(0.0, abs=1e-4)
    assert probe.decision_function(X[y == 1]).mean() > 0 > probe.decision_function(X[y == 0]).mean()


def test_dense_topk_uses_the_informative_dimensions():
    X, y = _gaussians()
    probe = P.DenseTopKProbe(k=2, C=1.0).fit(X, y)
    assert set(probe.selected_.tolist()) == {0, 1} and probe.n_features_used_ == 2
    assert roc_auc_score(y, probe.decision_function(X)) > 0.95


def test_sae_single_finds_the_planted_feature(synthetic_dataset):
    ds = synthetic_dataset
    train = ~ds.is_test
    probe = P.SaeSingleFeatureProbe().fit(ds.codes[train], ds.y[train])
    assert probe.selected_.tolist() == [ds.planted_feature] and probe.signs_.tolist() == [1]
    assert probe.n_features_used_ == 1
    scores = probe.decision_function(ds.codes[ds.is_test])
    assert np.array_equal(scores, ds.codes[ds.is_test][:, ds.planted_feature].toarray().ravel())
    assert roc_auc_score(ds.y[ds.is_test], scores) > 0.75


def test_sae_single_negative_sign_flips_the_score():
    X, y = _codes()
    probe = P.SaeSingleFeatureProbe().fit(X, 1 - y)  # feature 0 now marks the negative class
    assert probe.selected_.tolist() == [0] and probe.signs_.tolist() == [-1]
    assert roc_auc_score(1 - y, probe.decision_function(X)) == 1.0


def test_sae_topk_and_sae_all(synthetic_dataset):
    ds = synthetic_dataset
    train, test = ~ds.is_test, ds.is_test
    topk = P.SaeTopKProbe(k=4, C=1.0).fit(ds.codes[train], ds.y[train])
    assert topk.selected_[0] == ds.planted_feature and len(topk.selected_) == 4
    assert len(topk.signs_) == len(topk.selector_scores_) == 4
    assert roc_auc_score(ds.y[test], topk.decision_function(ds.codes[test])) > 0.75

    sae_all = P.SaeAllProbe(C=1.0).fit(ds.codes[train], ds.y[train])
    assert ds.planted_feature in sae_all.selected_.tolist()
    assert sae_all.n_features_used_ == len(sae_all.selected_) <= 100
    assert roc_auc_score(ds.y[test], sae_all.decision_function(ds.codes[test])) > 0.75


def test_refusal_dir_ignores_borderline_safe():
    rng = np.random.RandomState(0)
    X = rng.randn(90, 8)
    aux = np.array(["harmful"] * 30 + ["harmless"] * 30 + ["borderline_safe"] * 30)
    X[:30, 0] += 3.0
    y = rng.randint(0, 2, 90)  # the direction must come from the category, not from the label
    a = P.RefusalDirectionProbe().fit(X, y, aux=aux)

    X_moved = X.copy()
    X_moved[60:] += 100.0  # only borderline_safe rows change
    b = P.RefusalDirectionProbe().fit(X_moved, 1 - y, aux=aux)

    expected = X[:30].mean(0) - X[30:60].mean(0)
    assert np.allclose(a.r_, expected / np.linalg.norm(expected))
    assert np.allclose(a.r_, b.r_)
    assert np.linalg.norm(a.r_) == pytest.approx(1.0)
    assert a.decision_function(X[:30]).mean() > a.decision_function(X[30:60]).mean()


def test_refusal_dir_needs_categories():
    X, y = _gaussians(40, 4)
    with pytest.raises(ValueError):
        P.RefusalDirectionProbe().fit(X, y)
    with pytest.raises(ValueError):
        P.RefusalDirectionProbe().fit(X, y, aux=np.array(["harmful"] * 40))


def test_category_only_probe():
    rng = np.random.RandomState(0)
    category = rng.choice(["harmful", "harmless"], 300)
    meta = pd.DataFrame({"prompt_category": category, "source": rng.choice(["s1", "s2"], 300)})
    y = ((category == "harmful") ^ (rng.rand(300) < 0.1)).astype(int)
    probe = P.CategoryOnlyProbe().fit(meta, y)
    assert roc_auc_score(y, probe.decision_function(meta)) > 0.8
    unseen = pd.DataFrame({"prompt_category": ["never_seen"], "source": ["s9"]})
    assert np.isfinite(probe.decision_function(unseen)).all()  # unknown categories are ignored, not an error


def test_shuffled_and_random_feature_controls(synthetic_dataset):
    ds = synthetic_dataset
    shuffled = P.ShuffledLabelProbe(P.SaeSingleFeatureProbe(), seed=3).fit(ds.codes, ds.y)
    assert shuffled.decision_function(ds.codes).shape == (200,)
    assert np.isfinite(shuffled.threshold_)
    # the base estimator given to the constructor is cloned, never fit in place
    assert not hasattr(shuffled.base, "selected_")

    a = P.RandomFeatureProbe(k=5, C=1.0, seed=1).fit(ds.codes, ds.y)
    b = P.RandomFeatureProbe(k=5, C=1.0, seed=1).fit(ds.codes, ds.y)
    c = P.RandomFeatureProbe(k=5, C=1.0, seed=2).fit(ds.codes, ds.y)
    assert a.selected_.tolist() == b.selected_.tolist() != c.selected_.tolist()
    assert len(a.selected_) == a.n_features_used_ == 5


def _all_probes() -> list:
    return [
        P.DiffMeansProbe(), P.LogRegProbe(C=0.5), P.DenseTopKProbe(k=3, C=0.5),
        P.SaeTopKProbe(k=3, C=0.5, selection="auroc"), P.SaeAllProbe(C=0.5),
        P.SaeSingleFeatureProbe(selection="mutual_info"), P.RefusalDirectionProbe(), P.CategoryOnlyProbe(),
        P.ShuffledLabelProbe(P.LogRegProbe(C=0.5), seed=5), P.RandomFeatureProbe(k=3, C=0.5, seed=5),
        P.FeatureSelector(k=3, method="auroc"),
    ]


@pytest.mark.parametrize("probe", _all_probes(), ids=lambda p: type(p).__name__)
def test_clone_works_for_every_probe(probe, synthetic_dataset):
    ds = synthetic_dataset
    copy = clone(probe)
    assert type(copy) is type(probe) and copy is not probe
    assert repr(copy.get_params()) == repr(probe.get_params())

    if isinstance(copy, P.FeatureSelector):
        assert copy.fit(ds.codes, ds.y).transform(ds.codes).shape == (200, 3)
        return
    if isinstance(copy, P.CategoryOnlyProbe):
        X = ds.df[["prompt_category", "source"]]
    elif isinstance(copy, P.SAE_PROBES):
        X = ds.codes
    else:
        X = ds.X
    if isinstance(copy, P.RefusalDirectionProbe):
        copy.fit(X, ds.y, aux=ds.categories)
    else:
        copy.fit(X, ds.y)
    assert copy.decision_function(X).shape == (200,)
    assert set(np.unique(copy.predict(X))) <= {0, 1}
    assert not hasattr(probe, "threshold_")  # fitting the clone leaves the original untouched


# --------------------------------------------------------------------------- method table / persistence

def test_method_specs(real_cfg):
    refusal = P.method_specs(real_cfg, "refusal")
    assert list(refusal) == real_cfg.get("probes.methods")
    fmt = P.method_specs(real_cfg, "format_break")
    assert "refusal_dir" not in fmt and set(refusal) - set(fmt) == {"refusal_dir"}

    assert {name: spec.input for name, spec in refusal.items()} == {
        "dim": "dense", "logreg": "dense", "dense_topk": "dense", "refusal_dir": "dense",
        "sae_topk": "sae", "sae_all": "sae", "sae_single": "sae", "category_only": "meta",
    }
    assert [name for name, spec in refusal.items() if spec.needs_aux] == ["refusal_dir"]
    assert refusal["dim"].grid == [{}] and refusal["sae_single"].grid == [{}]
    assert len(refusal["logreg"].grid) == 9
    assert len(refusal["sae_topk"].grid) == 9 * 7 and len(refusal["dense_topk"].grid) == 9 * 4
    assert all(isinstance(g["C"], float) and isinstance(g["k"], int) for g in refusal["sae_topk"].grid)

    probe = refusal["sae_topk"].make(k=8, C=0.1)
    assert isinstance(probe, P.SaeTopKProbe) and (probe.k, probe.C) == (8, 0.1)
    assert probe.selection == real_cfg.get("probes.selection") and probe.max_iter == real_cfg.get("probes.max_iter")

    shuffled = P.shuffled_spec(refusal["logreg"], seed=1)
    assert shuffled.grid == refusal["logreg"].grid and isinstance(shuffled.make(C=1.0), P.ShuffledLabelProbe)
    rand = P.random_feature_spec(real_cfg, k=8, seed=1)
    assert rand.input == "sae" and rand.make(C=1.0).k == 8


def test_method_specs_are_picklable(real_cfg):
    import pickle

    for spec in P.method_specs(real_cfg, "refusal").values():  # joblib workers receive the specs
        assert pickle.loads(pickle.dumps(spec)).name == spec.name


def test_save_and_load_probe(tmp_path):
    X, y = _gaussians()
    probe = P.LogRegProbe(C=1.0).fit(X, y)
    path = tmp_path / "probes" / "logreg__L2__last__dense.joblib"
    P.save_probe(probe, path, {"key": "logreg__L2__last__dense", "layer": 2})
    assert path.exists() and path.with_suffix(".json").exists()
    loaded, meta = P.load_probe(path)
    assert meta == {"key": "logreg__L2__last__dense", "layer": 2}
    assert np.array_equal(loaded.decision_function(X), probe.decision_function(X))
    assert loaded.threshold_ == probe.threshold_


def test_raw_direction():
    X, y = _gaussians()
    for probe in (P.DiffMeansProbe().fit(X, y), P.LogRegProbe(C=1.0).fit(X, y),
                  P.RefusalDirectionProbe().fit(X, y, aux=np.where(y == 1, "harmful", "harmless"))):
        v = P.raw_direction(probe)
        assert v.shape == (16,) and v.dtype == np.float32
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
        assert v[0] > 0.3 and v[1] < -0.3  # points from class 0 to class 1 in raw activation space
    with pytest.raises(TypeError):
        P.raw_direction(P.DenseTopKProbe(k=2).fit(X, y))
