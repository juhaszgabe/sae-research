"""plots.py: every figure draws from tables whose intervals touch or miss the point estimate."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import plots  # noqa: E402

TINY = 2.220446049250313e-16  # the rounding gap between the two AUROC routines


def test_ci_err_clips_at_zero():
    below, above = plots._ci_err([0.808, 0.5, 0.7], [0.743, 0.6, np.nan], [0.808 - TINY, 0.9, 0.8])
    assert np.all(below >= 0) and np.all(above >= 0)
    assert above[0] == 0.0  # interval ends at the point up to rounding
    assert below[1] == 0.0  # interval does not contain the point at all
    assert below[2] == 0.0  # missing bound: no bar instead of NaN
    assert below[0] == pytest.approx(0.065) and above[1] == pytest.approx(0.4)


def _results():
    return pd.DataFrame([
        {"method": "logreg", "width": "dense", "layer": 5, "position": "last", "key": "logreg__L5__last__dense",
         "cv_auroc_mean": 0.80, "cv_auroc_std": 0.02, "test_auroc": 0.808, "test_lo": 0.743, "test_hi": 0.808 - TINY},
        {"method": "sae_topk", "width": "16k", "layer": 5, "position": "last", "key": "sae_topk__L5__last__16k",
         "cv_auroc_mean": 0.75, "cv_auroc_std": 0.03, "test_auroc": 0.70, "test_lo": 0.72, "test_hi": 0.80},
        {"method": "sae_topk", "width": "65k", "layer": 5, "position": "last", "key": "sae_topk__L5__last__65k",
         "cv_auroc_mean": 0.76, "cv_auroc_std": 0.03, "test_auroc": 0.71, "test_lo": 0.65, "test_hi": 0.71},
    ])


def test_figures_with_intervals_that_end_at_or_miss_the_point(tmp_path):
    res = _results()
    k_curve = pd.DataFrame([{"method": "sae_topk", "width": "16k", "k": k, "test_auroc": 0.7 + 0.01 * i,
                             "test_lo": 0.7 + 0.01 * i + (TINY if i == 0 else -0.05), "test_hi": 0.7 + 0.01 * i - TINY}
                            for i, k in enumerate((1, 4, 16))])
    comparisons = pd.DataFrame([{"family": "headline", "method_a": "sae_topk", "key_a": "sae_topk__L5__last__16k",
                                 "method_b": "logreg", "key_b": "logreg__L5__last__dense", "delta": -0.1, "lo": -0.2,
                                 "hi": 0.0, "p": 0.04, "p_holm": 0.08}])
    sweep = pd.DataFrame([
        {"kind": "none", "direction": "induce", "vector": None, "alpha": 0.0, "behavior_rate": 0.1, "rate_lo": 0.1, "rate_hi": 0.1 - TINY},
        {"kind": "add", "direction": "induce", "vector": "dim", "alpha": 0.5, "behavior_rate": 0.6, "rate_lo": 0.61, "rate_hi": 0.7},
    ])
    shift = pd.DataFrame([{"method": "logreg", "width": "dense", "shift": "style_mix", "id_auroc": 0.9,
                           "ood_auroc": 0.8, "ood_lo": 0.8 + TINY, "ood_hi": 0.85}])
    plots.plot_method_comparison(res, comparisons, tmp_path / "comparison")
    plots.plot_k_curve(k_curve, tmp_path / "k_curve", (0.7, 0.8))
    plots.plot_width(res, tmp_path / "width")
    plots.plot_steering_curves(sweep, "induce", tmp_path / "steering")
    plots.plot_shift(shift, tmp_path / "shift")
    for name in ("comparison", "k_curve", "width", "steering", "shift"):
        assert (tmp_path / f"{name}.png").exists() and (tmp_path / f"{name}.pdf").exists()
