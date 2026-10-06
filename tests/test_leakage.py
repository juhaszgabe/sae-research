"""R2: selection, scaling and hyperparameter search only ever see the training rows of the current fold."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from conftest import eval_kwargs

from src import eval as ev
from src import probes as P


def _sae_topk_spec(grid: list[dict] | None = None, cls: type = P.SaeTopKProbe) -> P.MethodSpec:
    return P.MethodSpec("sae_topk", lambda **hp: cls(**hp), grid or [{"k": k, "C": 1.0} for k in (1, 4)], "sae")


def test_feature_predictive_only_on_test_rows_is_never_selected(synthetic_dataset):
    """Feature 0 equals the label on the test rows and is silent on every training row. A pipeline
    that selected features on all rows would pick it and score ≈ 1.0 on the test split."""
    ds = synthetic_dataset
    rng = np.random.RandomState(1)
    codes = ((rng.rand(200, 60) < 0.15) * rng.rand(200, 60)).astype(np.float32)  # pure noise
    codes[:, 0] = np.where(ds.is_test, ds.y * 5.0, 0.0)
    codes = sp.csr_matrix(codes)

    leaky = P.FeatureSelector(k=1).fit(codes, ds.y)
    assert leaky.selected_.tolist() == [0]  # sanity: selecting on all rows WOULD find it

    out = ev.evaluate_method(codes, **eval_kwargs(ds), spec=_sae_topk_spec(), seed=0, n_boot=50)
    assert 0 not in out.probe.selected_.tolist()
    assert out.row["test_auroc"] < 0.8
    assert out.row["cv_auroc_mean"] < 0.8


def test_feature_predictive_only_on_one_fold_does_not_inflate_that_fold(synthetic_dataset):
    """Same idea inside the cross-validation: the feature fires (as the label) only on outer fold 0,
    so a probe fit on the other folds can never have selected it when fold 0 is scored."""
    ds = synthetic_dataset
    rng = np.random.RandomState(2)
    codes = ((rng.rand(200, 60) < 0.15) * rng.rand(200, 60)).astype(np.float32)
    codes[:, 0] = np.where(ds.folds == 0, ds.y * 5.0, 0.0)
    out = ev.evaluate_method(sp.csr_matrix(codes), **eval_kwargs(ds), spec=_sae_topk_spec(), seed=0, n_boot=50)
    fold0 = out.cv.set_index("fold").loc[0, "auroc"]
    assert fold0 < 0.85, out.cv


class RecordingSelector(P.FeatureSelector):
    """Logs which rows it was fit on and which rows it later transformed. Column 0 of the matrix
    carries row id + 1 (it fires on every row, so the firing filter never selects it)."""

    log: list[tuple[str, frozenset, frozenset]] = []

    @staticmethod
    def _ids(X) -> frozenset:
        return frozenset(int(v) - 1 for v in np.asarray(X[:, 0].todense()).ravel())

    def fit(self, X, y):
        self.fit_ids_ = self._ids(X)
        RecordingSelector.log.append(("fit", self.fit_ids_, self.fit_ids_))
        return super().fit(X, y)

    def transform(self, X):
        RecordingSelector.log.append(("transform", self.fit_ids_, self._ids(X)))
        return super().transform(X)


class RecordingProbe(P.SaeTopKProbe):
    def _make_selector(self) -> P.FeatureSelector:
        return RecordingSelector(self.k, self.selection, self.min_firing_count, self.min_firing_frac,
                                 self.max_firing_frac, sparse=True)


def test_fit_rows_and_scored_rows_are_disjoint_in_every_fold(synthetic_dataset):
    ds = synthetic_dataset
    codes = ds.codes.tolil()
    codes[:, 0] = (np.arange(200) + 1.0)[:, None]
    codes = codes.tocsr()
    RecordingSelector.log = []

    spec = _sae_topk_spec([{"k": k, "C": c} for k in (2, 4) for c in (0.1, 1.0)], cls=RecordingProbe)
    out = ev.evaluate_method(codes, **eval_kwargs(ds), spec=spec, seed=0, n_inner=3, n_boot=50)

    log = RecordingSelector.log
    test_ids = frozenset(np.flatnonzero(ds.is_test).tolist())
    train_ids = frozenset(np.flatnonzero(~ds.is_test).tolist())
    fits = [fit_ids for kind, fit_ids, _ in log if kind == "fit"]
    scored = [(fit_ids, ids) for kind, fit_ids, ids in log if kind == "transform" and ids != fit_ids]

    # 5 outer folds × (3 inner folds + 1 outer refit) × 4 grid points + the final model
    assert len(fits) == 5 * (3 + 1) * 4 + 1
    assert all(not fit_ids & test_ids for fit_ids in fits), "a selector was fit on test rows"
    assert fits[-1] == train_ids  # the final model uses the whole train split
    assert 0 not in out.probe.selected_.tolist()

    # every transform is either on the selector's own fit rows (training-time) or on disjoint rows (scoring)
    assert scored, "no held-out scoring was recorded"
    assert all(not fit_ids & ids for fit_ids, ids in scored), "a probe was scored on rows it was fit on"
    assert len(scored) == len(fits)  # each fitted probe scores exactly one held-out set
    assert sum(ids == test_ids for _, ids in scored) == 1  # the test split is scored once

    # nested CV: every outer validation fold is scored only by probes that never saw it
    for fold in range(5):
        fold_ids = frozenset(np.flatnonzero((ds.folds == fold) & ~ds.is_test).tolist())
        assert any(ids == fold_ids for _, ids in scored)
        assert all(not fit_ids & fold_ids for fit_ids, ids in scored if ids <= fold_ids)


def test_shuffled_label_control_is_at_chance(synthetic_dataset):
    ds = synthetic_dataset
    logreg = P.MethodSpec("logreg", lambda **hp: P.LogRegProbe(**hp), [{"C": 0.01}, {"C": 1.0}], "dense")

    real = ev.evaluate_method(ds.X, **eval_kwargs(ds), spec=logreg, seed=0, n_boot=50)
    assert real.row["test_auroc"] > 0.85  # the signal is there...

    aurocs = [
        ev.evaluate_method(ds.X, **eval_kwargs(ds), spec=P.shuffled_spec(logreg, seed), seed=seed, n_boot=20).row["test_auroc"]
        for seed in range(12)
    ]
    assert abs(np.mean(aurocs) - 0.5) < 0.15, aurocs  # ...and is gone once the training labels are permuted
