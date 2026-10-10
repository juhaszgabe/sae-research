"""eval.py: bootstrap statistics, Holm correction, the evaluation protocol and its tables."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import eval_kwargs
from sklearn.metrics import roc_auc_score

from src import eval as ev
from src import probes as P


def _scores(n: int = 120, signal: float = 1.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    y = rng.permutation(np.repeat([0, 1], n // 2))
    return y, signal * y + rng.randn(n), np.arange(n) // 3


LOGREG = P.MethodSpec("logreg", lambda **hp: P.LogRegProbe(**hp), [{"C": c} for c in (0.01, 1.0, 100.0)], "dense")
DIM = P.MethodSpec("dim", lambda **hp: P.DiffMeansProbe(**hp), [{}], "dense")


# --------------------------------------------------------------------------- statistics

def test_fast_auroc_matches_sklearn():
    y, s, _ = _scores()
    assert ev._fast_auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    ties = np.round(s)  # heavy ties: average ranks must match sklearn too
    assert ev._fast_auroc(y, ties) == pytest.approx(roc_auc_score(y, ties))
    assert ev.auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(ev._fast_auroc(np.ones(5, dtype=int), np.arange(5.0)))


def test_bootstrap_auroc():
    y, s, groups = _scores()
    point, lo, hi = ev.bootstrap_auroc(y, s, groups, n_boot=300, seed=1)
    assert point == pytest.approx(roc_auc_score(y, s))
    assert 0.5 < lo < point < hi <= 1.0
    assert ev.bootstrap_auroc(y, s, groups, n_boot=300, seed=1) == (point, lo, hi)  # deterministic
    assert ev.bootstrap_auroc(y, s, groups, n_boot=300, seed=2) != (point, lo, hi)
    narrow = ev.bootstrap_auroc(y, s, groups, n_boot=300, seed=1, ci=0.5)
    assert narrow[2] - narrow[1] < hi - lo


def test_cluster_bootstrap_is_wider_than_iid_when_groups_are_duplicates():
    # 20 distinct prompts, each repeated 6 times in one group: only 20 independent units
    y0, s0, _ = _scores(20, seed=3)
    y, s, groups = np.repeat(y0, 6), np.repeat(s0, 6), np.repeat(np.arange(20), 6)
    _, lo_c, hi_c = ev.bootstrap_auroc(y, s, groups, n_boot=400, seed=0)
    _, lo_i, hi_i = ev.bootstrap_auroc(y, s, np.arange(120), n_boot=400, seed=0)
    assert hi_c - lo_c > 1.5 * (hi_i - lo_i)


def test_bootstrap_raises_when_resamples_are_single_class():
    with pytest.raises(ValueError):
        ev.bootstrap_auroc(np.ones(10, dtype=int), np.arange(10.0), np.arange(10))
    # two groups, one per class: half of all resamples contain a single class
    y = np.array([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match="single-class"):
        ev.bootstrap_auroc(y, np.arange(10.0), y, n_boot=200)


def test_paired_delta_of_a_method_with_itself():
    y, s, groups = _scores()
    d = ev.paired_delta(y, s, s, groups, n_boot=200)
    assert d == {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0}


def test_paired_delta_detects_a_real_difference():
    y, good, groups = _scores(signal=3.0)
    noise = np.random.RandomState(9).randn(len(y))
    d = ev.paired_delta(y, good, noise, groups, n_boot=400, seed=0)
    assert d["delta"] == pytest.approx(roc_auc_score(y, good) - roc_auc_score(y, noise))
    assert d["lo"] > 0 and d["p"] == pytest.approx(1 / 400)  # floored at 1 / n_boot
    rev = ev.paired_delta(y, noise, good, groups, n_boot=400, seed=0)
    assert rev["delta"] == pytest.approx(-d["delta"]) and rev["hi"] == pytest.approx(-d["lo"])


def test_holm():
    assert ev.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert ev.holm([0.5, 0.9]) == [1.0, 1.0]  # capped at 1
    assert ev.holm([0.02]) == [0.02]
    assert ev.holm([]) == []
    assert ev.holm([0.03, 0.01, 0.04]) == [0.06, 0.03, 0.06]  # input order is preserved


def test_bootstrap_rate_and_paired_rate_delta():
    x = np.array([1] * 30 + [0] * 70)
    rate, lo, hi = ev.bootstrap_rate(x, n_boot=500, seed=0)
    assert rate == 0.3 and 0.15 < lo < 0.3 < hi < 0.45
    assert ev.bootstrap_rate(x, n_boot=500, seed=0) == (rate, lo, hi)
    assert all(np.isnan(v) for v in ev.bootstrap_rate(np.array([])))

    same = ev.paired_rate_delta(x, x, n_boot=200)
    assert same == {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0}
    better = ev.paired_rate_delta(np.ones(100), x, n_boot=500, seed=0)
    assert better["delta"] == pytest.approx(0.7) and better["lo"] > 0.5 and better["p"] == pytest.approx(1 / 500)


# --------------------------------------------------------------------------- protocol

def test_evaluate_method_is_deterministic(synthetic_dataset):
    ds = synthetic_dataset
    a = ev.evaluate_method(ds.X, **eval_kwargs(ds), spec=LOGREG, seed=5, n_boot=100)
    b = ev.evaluate_method(ds.X, **eval_kwargs(ds), spec=LOGREG, seed=5, n_boot=100)
    drop = {"fit_seconds"}
    assert {k: v for k, v in a.row.items() if k not in drop} == {k: v for k, v in b.row.items() if k not in drop}
    assert np.array_equal(a.test_scores, b.test_scores)
    assert a.cv.equals(b.cv)


def test_evaluate_method_outputs(synthetic_dataset):
    ds = synthetic_dataset
    out = ev.evaluate_method(ds.X, **eval_kwargs(ds), spec=LOGREG, seed=0, n_boot=100)
    assert set(out.row) == {"hparams", "cv_auroc_mean", "cv_auroc_std", "test_auroc", "test_lo", "test_hi",
                            "test_auprc", "test_bal_acc", "n_train", "n_test", "n_features_used", "fit_seconds"}
    assert (out.row["n_train"], out.row["n_test"]) == (int((~ds.is_test).sum()), int(ds.is_test.sum()))
    assert out.row["n_features_used"] == 32
    assert json.loads(out.row["hparams"]) in LOGREG.grid

    y_test = ds.y[ds.is_test]
    assert out.test_scores.shape == y_test.shape and out.test_scores.dtype == np.float64
    assert out.row["test_auroc"] == pytest.approx(roc_auc_score(y_test, out.test_scores))
    assert out.row["test_lo"] <= out.row["test_auroc"] <= out.row["test_hi"]
    assert out.row["test_auroc"] > 0.85 and out.row["cv_auroc_mean"] > 0.85 and out.row["test_bal_acc"] > 0.7
    # the returned probe is the final model: it reproduces the test scores
    assert np.array_equal(out.probe.decision_function(ds.X[ds.is_test]), out.test_scores)

    assert out.cv["fold"].tolist() == [0, 1, 2, 3, 4]
    assert out.row["cv_auroc_mean"] == pytest.approx(out.cv["auroc"].mean())
    assert all(json.loads(h) in LOGREG.grid for h in out.cv["hparams"])


def test_evaluate_method_without_hyperparameters(synthetic_dataset):
    ds = synthetic_dataset
    out = ev.evaluate_method(ds.X, **eval_kwargs(ds), spec=DIM, seed=0, n_boot=50)
    assert out.row["hparams"] == "{}" and set(out.cv["hparams"]) == {"{}"}
    # CV estimate of a method without a search = plain per-fold fit and score
    train = np.flatnonzero(~ds.is_test)
    fit, val = train[ds.folds[train] != 2], train[ds.folds[train] == 2]
    probe = P.DiffMeansProbe().fit(ds.X[fit], ds.y[fit])
    assert out.cv.set_index("fold").loc[2, "auroc"] == pytest.approx(roc_auc_score(ds.y[val], probe.decision_function(ds.X[val])))


def test_evaluate_method_fixed_hyperparameters(synthetic_dataset):
    ds = synthetic_dataset
    spec = P.MethodSpec("sae_topk", lambda **hp: P.SaeTopKProbe(**hp),
                        [{"k": k, "C": c} for k in (1, 4, 16) for c in (0.1, 1.0)], "sae")
    out = ev.evaluate_method(ds.codes, **eval_kwargs(ds), spec=spec, seed=0, n_boot=50, fixed={"k": 4})
    assert json.loads(out.row["hparams"])["k"] == 4 and out.row["n_features_used"] == 4
    assert all(json.loads(h)["k"] == 4 for h in out.cv["hparams"])
    # a fixed point outside the grid is used as is
    off_grid = ev.evaluate_method(ds.codes, **eval_kwargs(ds), spec=spec, seed=0, n_boot=50, fixed={"k": 2, "C": 0.5})
    assert json.loads(off_grid.row["hparams"]) == {"C": 0.5, "k": 2}
    with pytest.raises(ValueError):
        ev.evaluate_method(ds.codes, **eval_kwargs(ds), spec=spec, seed=0, n_boot=50, fixed={"k": 3})


def test_grid_order_breaks_ties_towards_smaller_k_then_c():
    spec = P.MethodSpec("x", lambda **hp: None, [{"k": 16, "C": 1.0}, {"k": 1, "C": 10.0}, {"k": 1, "C": 0.1}], "sae")
    grid = ev._restrict_grid(spec, None)
    assert grid == [{"k": 1, "C": 0.1}, {"k": 1, "C": 10.0}, {"k": 16, "C": 1.0}]
    assert ev._best(np.array([0.9, 0.9, 0.9])) == 0  # ties → first = smallest k, then smallest C
    assert ev._best(np.array([0.7, np.nan, 0.9])) == 2
    assert ev._best(np.array([np.nan, np.nan])) == 0


def test_needs_aux_is_enforced(synthetic_dataset):
    ds = synthetic_dataset
    spec = P.MethodSpec("refusal_dir", lambda **hp: P.RefusalDirectionProbe(**hp), [{}], "dense", needs_aux=True)
    with pytest.raises(ValueError, match="aux"):
        ev.evaluate_method(ds.X, **eval_kwargs(ds), spec=spec, seed=0, n_boot=50)
    out = ev.evaluate_method(ds.X, **eval_kwargs(ds), spec=spec, seed=0, n_boot=50, aux=ds.categories)
    assert np.isfinite(out.row["test_auroc"])


def test_category_only_takes_a_dataframe(synthetic_dataset):
    ds = synthetic_dataset
    spec = P.MethodSpec("category_only", lambda **hp: P.CategoryOnlyProbe(**hp), [{}], "meta")
    out = ev.evaluate_method(ds.df[["prompt_category", "source"]], **eval_kwargs(ds), spec=spec, seed=0, n_boot=50)
    assert 0.2 < out.row["test_auroc"] < 0.8  # categories are independent of the label here


# --------------------------------------------------------------------------- tables

def _results() -> pd.DataFrame:
    rows = [
        # method, layer, position, width, cv, test
        ("logreg", 1, "last", "dense", 0.80, 0.95),
        ("logreg", 2, "last", "dense", 0.90, 0.70),   # best by CV although its test AUROC is lower
        ("logreg", 3, "mean_user", "dense", 0.99, 0.99),  # other position: ignored
        ("sae_topk", 1, "last", "16k", 0.85, 0.80),
        ("sae_topk", 2, "last", "16k", 0.75, 0.90),
        ("sae_topk", 2, "last", "65k", 0.88, 0.60),
        ("category_only", -1, "-", "-", 0.55, 0.50),
    ]
    return pd.DataFrame([
        {"behavior": "refusal", "method": m, "layer": layer, "position": pos, "width": w,
         "key": ev.config_key(m, None if layer < 0 else layer, pos, None if w in ("dense", "-") else w),
         "cv_auroc_mean": cv, "test_auroc": test}
        for m, layer, pos, w, cv, test in rows
    ])


def test_config_key():
    assert ev.config_key("logreg", 13, "last", None) == "logreg__L13__last__dense"
    assert ev.config_key("sae_topk", 7, "model_tag", "65k") == "sae_topk__L7__model_tag__65k"
    assert ev.config_key("category_only", None, None, None) == "category_only__L-__-__-"


def test_best_layers_uses_cv_never_test():
    best = ev.best_layers(_results()).set_index(["method", "width"])
    assert len(best) == 4
    assert best.loc[("logreg", "dense"), "layer"] == 2
    assert best.loc[("sae_topk", "16k"), "layer"] == 1
    assert best.loc[("sae_topk", "65k"), "layer"] == 2
    assert best.loc[("category_only", "-"), "key"] == "category_only__L-__-__-"
    assert ev.best_layers(_results(), position="mean_user").set_index("method").loc["logreg", "layer"] == 3


def test_inner_cv_possible():
    y = np.array([0, 1] * 30)
    groups = np.arange(60)
    folds = np.arange(60) % 5
    assert ev._inner_cv_possible(y, groups, folds, n_inner=3)
    assert not ev._inner_cv_possible(y[:6], groups[:6], folds[:6], n_inner=3)  # folds without both classes
    assert not ev._inner_cv_possible(y, np.zeros(60, dtype=int), folds, n_inner=3)  # a single group
    assert not ev._inner_cv_possible(y, groups, np.zeros(60, dtype=int), n_inner=3)  # a single outer fold
