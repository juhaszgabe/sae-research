"""Nested-CV evaluation protocol, bootstrap/Holm statistics, layer sweeps, comparisons, OOD, monitor latency.

`evaluate_method` is the only evaluation path (R1): every method gets the identical split, folds,
metric and bootstrap.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from . import activations, model as model_lib, sae as sae_lib
from .config import Config
from .model import ModelBundle
from .probes import (
    SAE_PROBES, MethodSpec, ShuffledLabelProbe, load_probe, method_specs, random_feature_spec, save_probe,
    shuffled_spec,
)
from .sae import JumpReLUSAE
from .utils import derive_seed, get_logger, paths, read_parquet, write_json, write_parquet

log = get_logger("eval")

RESULT_COLUMNS = [
    "behavior", "method", "layer", "position", "width", "key", "hparams", "cv_auroc_mean", "cv_auroc_std",
    "test_auroc", "test_lo", "test_hi", "test_auprc", "test_bal_acc", "n_train", "n_test", "n_features_used",
    "fit_seconds",
]
_NO_LAYER, _NO_VALUE = -1, "-"  # layer / position / width of category_only


# --------------------------------------------------------------------------- statistics (§13.1)

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def _fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann–Whitney AUROC with average ranks (== roc_auc_score); NaN for a single class."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _group_resamples(y: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Cluster bootstrap: yields n_boot row-index arrays, resampling unique groups with replacement.
    Single-class resamples are redrawn; raises if more than 10 % of the draws had to be redrawn."""
    y = np.asarray(y).astype(int)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    if len(np.unique(y)) < 2:
        raise ValueError("bootstrap needs both classes")
    rng = np.random.RandomState(seed % (2**32))
    done = redraws = 0
    while done < n_boot:
        idx = np.concatenate([members[g] for g in rng.randint(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            redraws += 1
            if redraws > max(0.1 * n_boot, 10) and redraws > 0.1 * (done + redraws):
                raise ValueError(f"cluster bootstrap: > 10 % single-class resamples ({redraws} redraws, {done} valid)")
            continue
        done += 1
        yield idx


def _percentile_ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(lo), float(hi)


def _two_sided_p(deltas: np.ndarray) -> float:
    p = 2 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return float(max(min(1.0, p), 1.0 / len(deltas)))


def _auroc_samples(y, s, groups, n_boot: int, seed: int) -> np.ndarray:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
    return np.array([_fast_auroc(y[idx], s[idx]) for idx in _group_resamples(y, groups, n_boot, seed)])


def bootstrap_auroc(y, s, groups, n_boot: int = 1000, seed: int = 0, ci: float = 0.95
                    ) -> tuple[float, float, float]:
    """Cluster bootstrap: resample unique group_ids with replacement; redraw single-class
    resamples (raise if > 10 %). Returns (point, lo, hi) percentile CI."""
    lo, hi = _percentile_ci(_auroc_samples(y, s, groups, n_boot, seed), ci)
    return auroc(np.asarray(y).astype(int), np.asarray(s)), lo, hi


def paired_delta(y, s_a, s_b, groups, n_boot: int = 1000, seed: int = 0, ci: float = 0.95) -> dict:
    """Same resamples for both score vectors. {delta, lo, hi, p} with
    p = min(1, 2·min(P(Δ*≤0), P(Δ*≥0))), floored at 1/n_boot."""
    y = np.asarray(y).astype(int)
    s_a, s_b = np.asarray(s_a, dtype=np.float64), np.asarray(s_b, dtype=np.float64)
    deltas = np.array([
        _fast_auroc(y[idx], s_a[idx]) - _fast_auroc(y[idx], s_b[idx])
        for idx in _group_resamples(y, groups, n_boot, seed)
    ])
    lo, hi = _percentile_ci(deltas, ci)
    return {"delta": auroc(y, s_a) - auroc(y, s_b), "lo": lo, "hi": hi, "p": _two_sided_p(deltas)}


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return [round(float(v), 12) for v in out]


def bootstrap_rate(x: np.ndarray, n_boot=1000, seed=0, ci=0.95) -> tuple[float, float, float]:
    """Mean of a 0/1 (or real) vector with an i.i.d. bootstrap percentile CI."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed % (2**32))
    means = x[rng.randint(0, len(x), (n_boot, len(x)))].mean(axis=1)
    lo, hi = _percentile_ci(means, ci)
    return float(x.mean()), lo, hi


def paired_rate_delta(x_a: np.ndarray, x_b: np.ndarray, n_boot=1000, seed=0, ci=0.95) -> dict:
    """mean(x_a) − mean(x_b) on paired rows (same resampled indices). {delta, lo, hi, p}."""
    d = np.asarray(x_a, dtype=np.float64) - np.asarray(x_b, dtype=np.float64)
    if len(d) == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    rng = np.random.RandomState(seed % (2**32))
    deltas = d[rng.randint(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = _percentile_ci(deltas, ci)
    return {"delta": float(d.mean()), "lo": lo, "hi": hi, "p": _two_sided_p(deltas)}


# --------------------------------------------------------------------------- protocol (§13.2)

@dataclass
class EvalOutput:
    row: dict  # summary (columns of probe_results, §13.3)
    test_scores: np.ndarray  # float64 [n_test]
    probe: BaseEstimator  # final probe fit on the whole train split
    cv: pd.DataFrame  # per outer fold: fold, auroc, chosen hparams


def _take(X: Any, idx: np.ndarray) -> Any:
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _fit(spec: MethodSpec, hp: dict, X: Any, y: np.ndarray, aux: np.ndarray | None, idx: np.ndarray) -> BaseEstimator:
    probe = spec.make(**hp)
    if spec.needs_aux:
        if aux is None:
            raise ValueError(f"method {spec.name!r} needs aux (prompt categories)")
        return probe.fit(_take(X, idx), y[idx], aux=aux[idx])
    return probe.fit(_take(X, idx), y[idx])


def _fit_score(spec, hp, X, y, aux, fit_idx, score_idx) -> float:
    """AUROC on score_idx of a probe fit on fit_idx; NaN if either side has a single class."""
    if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[score_idx])) < 2:
        return float("nan")
    probe = _fit(spec, hp, X, y, aux, fit_idx)
    return _fast_auroc(y[score_idx], probe.decision_function(_take(X, score_idx)))


def _restrict_grid(spec: MethodSpec, fixed: dict | None) -> list[dict]:
    """spec.grid restricted by `fixed`, ordered so that ties resolve to smaller k, then smaller C."""
    grid = spec.grid
    if fixed:
        keys = set().union(*(g.keys() for g in spec.grid))
        use = {k: v for k, v in fixed.items() if k in keys}
        grid = [g for g in spec.grid if all(g.get(k) == v for k, v in use.items())]
        if not grid:
            if set(use) != keys:
                raise ValueError(f"{spec.name}: fixed={fixed} matches no grid point and does not set all of {keys}")
            grid = [use]
    return sorted(grid, key=lambda g: (g.get("k", 0), g.get("C", 0.0)))


def _best(scores: np.ndarray) -> int:
    """Index of the best mean score; first (= smallest k, then C) on ties; 0 if all NaN."""
    if np.all(np.isnan(scores)):
        return 0
    return int(np.nanargmax(np.round(scores, 12)))


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        counts = (~np.isnan(a)).sum(axis=axis)
        return np.where(counts > 0, np.nansum(a, axis=axis) / np.maximum(counts, 1), np.nan)


def evaluate_method(X, y: np.ndarray, groups: np.ndarray, folds: np.ndarray, is_test: np.ndarray,
                    spec: MethodSpec, seed: int, n_inner: int = 3, n_boot: int = 1000,
                    aux: np.ndarray | None = None, fixed: dict | None = None) -> EvalOutput:
    """Nested-CV estimate on the train folds, final model refit on the whole train split,
    test split scored once (spec §13.2)."""
    t0 = time.perf_counter()
    y = np.asarray(y).astype(int)
    groups, folds, is_test = np.asarray(groups), np.asarray(folds), np.asarray(is_test).astype(bool)
    aux = None if aux is None else np.asarray(aux)
    train = np.flatnonzero(~is_test)
    test = np.flatnonzero(is_test)
    outer = sorted(int(f) for f in np.unique(folds[train]))
    grid = _restrict_grid(spec, fixed)
    search = len(grid) > 1

    # outer_scores[f, g]: fit on TR \ f, AUROC on f. Used for the final model's hyperparameters
    # (step 2) and, at the inner-CV choice, as the nested estimate (step 1).
    outer_scores = np.full((len(outer), len(grid)), np.nan)
    chosen = []
    for i, f in enumerate(outer):
        fit_idx, val_idx = train[folds[train] != f], train[folds[train] == f]
        if search:
            inner = StratifiedGroupKFold(n_inner, shuffle=True, random_state=derive_seed(seed, spec.name, f))
            inner_scores = np.full((n_inner, len(grid)), np.nan)
            splits = inner.split(np.zeros(len(fit_idx)), y[fit_idx], groups[fit_idx])
            for j, (a, b) in enumerate(splits):
                for g, hp in enumerate(grid):
                    inner_scores[j, g] = _fit_score(spec, hp, X, y, aux, fit_idx[a], fit_idx[b])
            chosen.append(_best(_nanmean(inner_scores, 0)))
            for g, hp in enumerate(grid):
                outer_scores[i, g] = _fit_score(spec, hp, X, y, aux, fit_idx, val_idx)
        else:
            chosen.append(0)
            outer_scores[i, 0] = _fit_score(spec, grid[0], X, y, aux, fit_idx, val_idx)

    nested = np.array([outer_scores[i, g] for i, g in enumerate(chosen)])
    cv = pd.DataFrame({
        "fold": outer,
        "auroc": nested,
        "hparams": [json.dumps(grid[g], sort_keys=True) for g in chosen],
    })

    final_hp = grid[_best(_nanmean(outer_scores, 0))]
    probe = _fit(spec, final_hp, X, y, aux, train)

    test_scores = np.asarray(probe.decision_function(_take(X, test)), dtype=np.float64) if len(test) else np.array([])
    y_test = y[test]
    if len(test) and len(np.unique(y_test)) == 2:
        point, lo, hi = bootstrap_auroc(y_test, test_scores, groups[test], n_boot, derive_seed(seed, "bootstrap"))
        auprc = float(average_precision_score(y_test, test_scores))
        bal_acc = float(balanced_accuracy_score(y_test, (test_scores > probe.threshold_).astype(int)))
    else:
        point = lo = hi = auprc = bal_acc = float("nan")

    valid = nested[~np.isnan(nested)]
    row = {
        "hparams": json.dumps(final_hp, sort_keys=True),
        "cv_auroc_mean": float(valid.mean()) if len(valid) else float("nan"),
        "cv_auroc_std": float(valid.std()) if len(valid) else float("nan"),
        "test_auroc": point, "test_lo": lo, "test_hi": hi,
        "test_auprc": auprc, "test_bal_acc": bal_acc,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_features_used": int(getattr(probe, "n_features_used_", 0)),
        "fit_seconds": float(time.perf_counter() - t0),
    }
    return EvalOutput(row=row, test_scores=test_scores, probe=probe, cv=cv)


# --------------------------------------------------------------------------- sweeps (§13.3)

def config_key(method: str, layer: int | None, position: str | None, width: str | None) -> str:
    """f"{method}__L{layer}__{position}__{width or 'dense'}" (`L-` / `-` for category_only)."""
    if layer is None or layer == _NO_LAYER:
        return f"{method}__L-__-__-"
    return f"{method}__L{layer}__{position}__{width or 'dense'}"


def load_inputs(cfg: Config, behavior: str, layer: int | None, position: str | None,
                spec: MethodSpec, width: str | None = None, pool: str = "main"
                ) -> tuple[Any, pd.DataFrame]:
    """dense → activations.load_acts; sae → sae.load_codes; meta → dataset columns. Returns (X, index)."""
    p = paths(cfg, behavior, pool)
    if spec.input == "dense":
        return activations.load_acts(cfg, behavior, layer, position, pool)
    if spec.input == "sae":
        if width is None:
            raise ValueError("SAE methods need a width")
        X = sae_lib.load_codes(cfg, behavior, layer, position, width, pool=pool)
        index = read_parquet(p.acts_dir / "index.parquet")
        if X.shape[0] != len(index):
            raise ValueError(f"codes rows ({X.shape[0]}) != index rows ({len(index)})")
        return X, index
    if spec.input == "meta":
        df = read_parquet(p.pool_dataset).reset_index(drop=True)
        index = df[["prompt_id", "split", "fold", "label", "group_id", "prompt_category"]].copy()
        index.insert(0, "row", np.arange(len(df)))
        return df[["prompt_category", "source"]].copy(), index
    raise ValueError(f"unknown input kind {spec.input!r}")


def _eval_arrays(index: pd.DataFrame) -> dict:
    return dict(
        y=index["label"].to_numpy().astype(int),
        groups=index["group_id"].to_numpy(),
        folds=index["fold"].to_numpy().astype(int),
        is_test=(index["split"] == "test").to_numpy(),
    )


def _evaluate_config(cfg: Config, behavior: str, spec: MethodSpec, layer: int | None, position: str | None,
                     width: str | None, fixed: dict | None = None) -> tuple[EvalOutput, pd.DataFrame]:
    X, index = load_inputs(cfg, behavior, layer, position, spec, width)
    aux = index["prompt_category"].to_numpy() if spec.needs_aux else None
    out = evaluate_method(
        X, **_eval_arrays(index), spec=spec, seed=cfg["seed"], n_inner=cfg.get("dataset.n_inner_folds"),
        n_boot=cfg.get("eval.n_bootstrap"), aux=aux, fixed=fixed,
    )
    return out, index


def _identity(behavior: str, method: str, layer: int | None, position: str | None, width: str | None) -> dict:
    meta = layer is None
    return {
        "behavior": behavior, "method": method,
        "layer": _NO_LAYER if meta else int(layer),
        "position": _NO_VALUE if meta else position,
        "width": _NO_VALUE if meta else (width or "dense"),
        "key": config_key(method, layer, position, width),
    }


def _run_config(cfg: Config, behavior: str, method: str, layer: int | None, position: str | None,
                width: str | None) -> dict:
    """One sweep configuration: evaluate, save scores / probe / features, return the result row."""
    spec = method_specs(cfg, behavior)[method]
    out, index = _evaluate_config(cfg, behavior, spec, layer, position, width)
    row = {**_identity(behavior, method, layer, position, width), **out.row}
    key = row["key"]
    results_dir: Path = paths(cfg, behavior).results_dir

    test = index[index["split"] == "test"]
    write_parquet(pd.DataFrame({
        "prompt_id": test["prompt_id"].to_numpy(), "group_id": test["group_id"].to_numpy(),
        "y": test["label"].to_numpy().astype(int), "score": out.test_scores,
        "prompt_category": test["prompt_category"].to_numpy(),
    }), results_dir / "scores" / f"{key}.parquet")
    save_probe(out.probe, results_dir / "probes" / f"{key}.joblib", {**row, "cv": out.cv.to_dict("records")})
    if spec.input == "sae":
        write_json(results_dir / "features" / f"{key}.json", {
            "key": key, "layer": layer, "position": position, "width": width,
            "selected": np.asarray(out.probe.selected_).tolist(),
            "signs": np.asarray(out.probe.signs_).tolist(),
            "selector_scores": np.asarray(out.probe.selector_scores_).tolist(),
        })
    return row


def _append_result(path: Path, row: dict) -> pd.DataFrame:
    new = pd.DataFrame([row], columns=RESULT_COLUMNS)
    if path.exists():
        old = read_parquet(path)
        new = pd.concat([old[old["key"] != row["key"]], new], ignore_index=True)
    write_parquet(new, path)
    return new


def run_probe_sweep(cfg: Config, behavior: str, methods: Sequence[str] | None = None,
                    layers: Sequence[int] | None = None, positions: Sequence[str] | None = None,
                    widths: Sequence[str] | None = None, overwrite: bool = False) -> pd.DataFrame:
    """Loops over configurations: dense methods × layers_dense × positions; SAE methods ×
    layers_sae × widths × fixed positions (never mean_user); category_only once.
    Runs in parallel with joblib (cfg eval.n_jobs). After each configuration: append its row to
    results_dir/probe_results.parquet, save test scores to scores/{key}.parquet
    (prompt_id, group_id, y, score, prompt_category), probe to probes/{key}.joblib, and for SAE
    methods the selected features to features/{key}.json. Existing keys are skipped unless
    overwrite (resume after Colab disconnect). Returns the full results table."""
    specs = method_specs(cfg, behavior)
    if methods is not None:
        unknown = [m for m in methods if m not in specs]
        if unknown:
            raise ValueError(f"methods {unknown} not available for {behavior} (available: {list(specs)})")
        specs = {m: specs[m] for m in methods}
    positions = list(positions) if positions is not None else list(cfg.get("eval.positions"))
    widths = list(widths) if widths is not None else list(cfg.get("sae.widths"))
    dense_layers = [x for x in cfg.model.layers_dense if layers is None or x in layers]
    sae_layers = [x for x in cfg.model.layers_sae if layers is None or x in layers]
    fixed_positions = [p for p in positions if p in activations.POSITIONS]

    configs: list[tuple[str, int | None, str | None, str | None]] = []
    for name, spec in specs.items():
        if spec.input == "dense":
            configs += [(name, layer, pos, None) for layer in dense_layers for pos in positions]
        elif spec.input == "sae":
            configs += [(name, layer, pos, w) for w in widths for layer in sae_layers for pos in fixed_positions]
        else:
            configs.append((name, None, None, None))

    results_path = paths(cfg, behavior).results_dir / "probe_results.parquet"
    done = set(read_parquet(results_path)["key"]) if results_path.exists() and not overwrite else set()
    todo = [c for c in configs if config_key(*c) not in done]
    log.info("probe sweep %s: %d configurations, %d already done", behavior, len(configs), len(configs) - len(todo))

    if todo:
        jobs = (delayed(_run_config)(cfg, behavior, *c) for c in todo)
        for row in Parallel(n_jobs=cfg.get("eval.n_jobs"), return_as="generator_unordered")(jobs):
            _append_result(results_path, row)
            log.info("%s: cv %.3f, test %.3f", row["key"], row["cv_auroc_mean"], row["test_auroc"])
    if not results_path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return read_parquet(results_path).sort_values(["method", "width", "layer", "position"]).reset_index(drop=True)


def best_layers(results: pd.DataFrame, position: str = "last") -> pd.DataFrame:
    """Per (method, width): the row with the highest cv_auroc_mean (never chosen by test AUROC)."""
    sub = results[(results["position"] == position) | (results["position"] == _NO_VALUE)]
    sub = sub.dropna(subset=["cv_auroc_mean"]).sort_values(["cv_auroc_mean", "layer"], ascending=[False, True])
    return sub.groupby(["method", "width"], sort=True).head(1).sort_values(["method", "width"]).reset_index(drop=True)


def _load_scores(cfg: Config, behavior: str, key: str) -> pd.DataFrame:
    return read_parquet(paths(cfg, behavior).results_dir / "scores" / f"{key}.parquet")


def _compare(cfg: Config, behavior: str, family: str, row_a: pd.Series, row_b: pd.Series) -> dict:
    a, b = _load_scores(cfg, behavior, row_a["key"]), _load_scores(cfg, behavior, row_b["key"])
    b = b.set_index("prompt_id").loc[a["prompt_id"]].reset_index()  # shared test set, same order
    d = paired_delta(a["y"].to_numpy(), a["score"].to_numpy(), b["score"].to_numpy(), a["group_id"].to_numpy(),
                     cfg.get("eval.n_bootstrap"), derive_seed(cfg["seed"], "compare"), cfg.get("eval.ci"))
    return {"family": family, "method_a": row_a["method"], "key_a": row_a["key"],
            "method_b": row_b["method"], "key_b": row_b["key"], **d}


def compare_methods(cfg: Config, behavior: str, results: pd.DataFrame,
                    reference: str = "logreg", position: str = "last") -> pd.DataFrame:
    """Paired bootstrap on the shared test set, three families, Holm-adjusted per family:
    headline (each method at its best layer vs reference at its best layer),
    per_layer (sae_topk vs logreg at each SAE layer × width), width (sae_topk per width vs main_width).
    Columns: family, method_a, key_a, method_b, key_b, delta, lo, hi, p, p_holm. Saves comparisons.parquet."""
    best = best_layers(results, position)
    ref_rows = best[best["method"] == reference]
    if ref_rows.empty:
        raise ValueError(f"reference method {reference!r} has no results at position {position!r}")
    ref = ref_rows.iloc[0]
    rows = []

    for _, r in best.iterrows():
        if r["key"] != ref["key"]:
            rows.append(_compare(cfg, behavior, "headline", r, ref))

    at_pos = results[results["position"] == position]
    for _, r in at_pos[at_pos["method"] == "sae_topk"].sort_values(["width", "layer"]).iterrows():
        partner = at_pos[(at_pos["method"] == "logreg") & (at_pos["layer"] == r["layer"])]
        if not partner.empty:
            rows.append(_compare(cfg, behavior, "per_layer", r, partner.iloc[0]))

    sae_best = best[best["method"] == "sae_topk"]
    main = sae_best[sae_best["width"] == cfg.get("sae.main_width")]
    if not main.empty:
        for _, r in sae_best[sae_best["width"] != cfg.get("sae.main_width")].iterrows():
            rows.append(_compare(cfg, behavior, "width", r, main.iloc[0]))

    out = pd.DataFrame(rows, columns=["family", "method_a", "key_a", "method_b", "key_b", "delta", "lo", "hi", "p"])
    out["p_holm"] = np.nan
    for _, idx in out.groupby("family").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].tolist())
    write_parquet(out, paths(cfg, behavior).results_dir / "comparisons.parquet")
    return out


def within_category(cfg: Config, behavior: str, results: pd.DataFrame) -> pd.DataFrame:
    """For each best-layer probe: test AUROC within each prompt_category with ≥ 10 per class
    (else NaN). Key number for refusal: AUROC within borderline_safe and within harmful
    (a pure 'harmful topic' detector scores ≈ 0.5 there). Saves within_category.parquet."""
    rows = []
    for _, r in best_layers(results).iterrows():
        scores = _load_scores(cfg, behavior, r["key"])
        for category, sub in scores.groupby("prompt_category", sort=True):
            y = sub["y"].to_numpy().astype(int)
            n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
            enough = min(n_pos, n_neg) >= 10
            rows.append({
                "method": r["method"], "width": r["width"], "layer": r["layer"], "key": r["key"],
                "prompt_category": category, "n_pos": n_pos, "n_neg": n_neg,
                "auroc": _fast_auroc(y, sub["score"].to_numpy()) if enough else float("nan"),
            })
    out = pd.DataFrame(rows, columns=["method", "width", "layer", "key", "prompt_category", "n_pos", "n_neg", "auroc"])
    write_parquet(out, paths(cfg, behavior).results_dir / "within_category.parquet")
    return out


def _row_inputs(r: pd.Series) -> tuple[int | None, str | None, str | None]:
    """(layer, position, width) arguments for load_inputs from a results row."""
    if r["layer"] == _NO_LAYER:
        return None, None, None
    return int(r["layer"]), r["position"], None if r["width"] == "dense" else r["width"]


def k_curve(cfg: Config, behavior: str, results: pd.DataFrame) -> pd.DataFrame:
    """sae_topk and dense_topk at their best layers, re-evaluated with k fixed to each k_grid value
    (only C tuned). Saves k_curve.parquet."""
    specs = method_specs(cfg, behavior)
    best = best_layers(results)
    rows = []
    for _, r in best[best["method"].isin(["sae_topk", "dense_topk"])].iterrows():
        layer, position, width = _row_inputs(r)
        for k in cfg.get("probes.k_grid"):
            out, _ = _evaluate_config(cfg, behavior, specs[r["method"]], layer, position, width, fixed={"k": int(k)})
            rows.append({**_identity(behavior, r["method"], layer, position, width), "k": int(k), **out.row})
    out_df = pd.DataFrame(rows)
    write_parquet(out_df, paths(cfg, behavior).results_dir / "k_curve.parquet")
    return out_df


def run_controls(cfg: Config, behavior: str, results: pd.DataFrame) -> pd.DataFrame:
    """At the best `last` layer: shuffled-label logreg and sae_topk (expected AUROC ≈ 0.5; log an
    ERROR if the CI excludes [0.4, 0.6]) and RandomFeatureProbe with the final k of sae_topk.
    Saves controls.parquet."""
    specs = method_specs(cfg, behavior)
    best = best_layers(results)
    seed = derive_seed(cfg["seed"], "controls")
    rows = []

    def run(control: str, spec: MethodSpec, r: pd.Series) -> dict:
        layer, position, width = _row_inputs(r)
        out, _ = _evaluate_config(cfg, behavior, spec, layer, position, width)
        return {**_identity(behavior, r["method"], layer, position, width), "control": control, **out.row}

    for method in ("logreg", "sae_topk"):
        cand = best[best["method"] == method].sort_values("cv_auroc_mean", ascending=False)
        if cand.empty or method not in specs:
            continue
        r = cand.iloc[0]
        row = run("shuffled_labels", shuffled_spec(specs[method], seed), r)
        if row["test_hi"] < 0.4 or row["test_lo"] > 0.6:
            log.error("shuffled-label control for %s: test AUROC CI [%.3f, %.3f] excludes [0.4, 0.6] — "
                      "possible leakage", method, row["test_lo"], row["test_hi"])
        rows.append(row)
        if method == "sae_topk":
            k = json.loads(r["hparams"]).get("k", 16)
            rows.append(run("random_features", random_feature_spec(cfg, k, seed), r))

    out_df = pd.DataFrame(rows)
    write_parquet(out_df, paths(cfg, behavior).results_dir / "controls.parquet")
    return out_df


# --------------------------------------------------------------------------- robustness [TDK] (§13.4)

def evaluate_shift(cfg: Config, behavior: str, shift: str, results: pd.DataFrame) -> pd.DataFrame:
    """Loads the SAVED final probes (never refits), scores the ood_{shift} activations/codes at every
    layer, reports ood_auroc (+CI), id_auroc (test), id_auroc_matched (test rows whose base_id is in
    the OOD pool), drop = id − ood with a CI from independent bootstraps, plus Holm-adjusted paired
    method comparisons on the OOD set. Saves results_dir/shift_{shift}.parquet."""
    pool = f"ood_{shift}"
    p = paths(cfg, behavior)
    specs = method_specs(cfg, behavior)
    n_boot, ci, seed = cfg.get("eval.n_bootstrap"), cfg.get("eval.ci"), cfg["seed"]

    def base_keys(df: pd.DataFrame) -> pd.Series:
        return df["source"].astype(str) + ":" + df["base_id"].astype(str)

    main = read_parquet(p.dataset)
    main_key = dict(zip(main["prompt_id"], base_keys(main)))
    ood_bases = set(base_keys(read_parquet(p.ood(shift))))

    rows, ood_scores = [], {}
    for _, r in results.iterrows():
        if r["method"] not in specs:
            continue
        layer, position, width = _row_inputs(r)
        probe, _ = load_probe(p.results_dir / "probes" / f"{r['key']}.joblib")
        X, index = load_inputs(cfg, behavior, layer, position, specs[r["method"]], width, pool=pool)
        y, groups = index["label"].to_numpy().astype(int), index["group_id"].to_numpy()
        s = np.asarray(probe.decision_function(X), dtype=np.float64)
        ood_scores[r["key"]] = (index["prompt_id"].to_numpy(), y, s, groups)

        ident = _load_scores(cfg, behavior, r["key"])
        matched = ident[ident["prompt_id"].map(main_key).isin(ood_bases)]
        row = {**{c: r[c] for c in ("behavior", "method", "layer", "position", "width", "key")}, "shift": shift,
               "n_ood": int(len(y)), "id_auroc": r["test_auroc"],
               "id_auroc_matched": _fast_auroc(matched["y"].to_numpy().astype(int), matched["score"].to_numpy())}
        if len(np.unique(y)) == 2:
            ood_s = _auroc_samples(y, s, groups, n_boot, derive_seed(seed, "shift", "ood"))
            id_s = _auroc_samples(ident["y"].to_numpy(), ident["score"].to_numpy(), ident["group_id"].to_numpy(),
                                  n_boot, derive_seed(seed, "shift", "id"))
            row["ood_auroc"] = _fast_auroc(y, s)
            row["ood_lo"], row["ood_hi"] = _percentile_ci(ood_s, ci)
            row["drop"] = row["id_auroc"] - row["ood_auroc"]
            row["drop_lo"], row["drop_hi"] = _percentile_ci(id_s - ood_s, ci)
        else:
            row.update(ood_auroc=np.nan, ood_lo=np.nan, ood_hi=np.nan, drop=np.nan, drop_lo=np.nan, drop_hi=np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)

    # Paired comparisons on the OOD set: each method at its best (ID-chosen) layer vs the reference.
    out["delta_vs_ref"], out["p_vs_ref"], out["p_holm_vs_ref"] = np.nan, np.nan, np.nan
    best = best_layers(results)
    ref = best[best["method"] == cfg.get("eval.reference")]
    if not ref.empty and ref.iloc[0]["key"] in ood_scores:
        ids_b, y_b, s_b, groups_b = ood_scores[ref.iloc[0]["key"]]
        order_b = pd.Series(np.arange(len(ids_b)), index=ids_b)
        compared = []
        for key in best["key"]:
            if key == ref.iloc[0]["key"] or key not in ood_scores:
                continue
            ids_a, y_a, s_a, _ = ood_scores[key]
            if len(np.unique(y_a)) < 2:
                continue
            d = paired_delta(y_a, s_a, s_b[order_b.loc[ids_a].to_numpy()], groups_b[order_b.loc[ids_a].to_numpy()],
                             n_boot, derive_seed(seed, "shift", "compare"), ci)
            compared.append((key, d))
        for (key, d), p_adj in zip(compared, holm([d["p"] for _, d in compared])):
            out.loc[out["key"] == key, ["delta_vs_ref", "p_vs_ref", "p_holm_vs_ref"]] = [d["delta"], d["p"], p_adj]

    write_parquet(out, p.results_dir / f"shift_{shift}.parquet")
    return out


def _inner_cv_possible(y: np.ndarray, groups: np.ndarray, folds: np.ndarray, n_inner: int) -> bool:
    """True if every outer training set can be split into n_inner stratified group folds."""
    outer = np.unique(folds)
    if len(outer) < 2:
        return False
    for f in outer:
        keep = folds != f
        if len(np.unique(groups[keep])) < n_inner or min((y[keep] == 0).sum(), (y[keep] == 1).sum()) < n_inner:
            return False
        if len(np.unique(y[~keep])) < 2:
            return False
    return True


def run_regimes(cfg: Config, behavior: str, results: pd.DataFrame,
                methods: Sequence[str] = ("dim", "logreg", "sae_topk"),
                train_sizes=(16, 32, 64, 128, 256), pos_fracs=(0.05, 0.1, 0.2),
                noise_rates=(0.1, 0.2, 0.3), n_repeats: int = 5) -> pd.DataFrame:
    """Kantamneni et al.'s scarcity / imbalance / label-noise settings at the best `last` layer:
    modify only the training rows, evaluate on the unchanged test split. If inner CV is impossible,
    use fallback hparams {C: 1.0, k: 16} and mark hp_fallback. Saves regimes.parquet."""
    specs = method_specs(cfg, behavior)
    best = best_layers(results)
    n_inner, n_boot = cfg.get("dataset.n_inner_folds"), cfg.get("eval.n_bootstrap")
    settings = ([("scarcity", v) for v in train_sizes] + [("imbalance", v) for v in pos_fracs]
                + [("label_noise", v) for v in noise_rates])
    rows = []

    for method in methods:
        cand = best[best["method"] == method].sort_values("cv_auroc_mean", ascending=False)
        if cand.empty or method not in specs:
            log.warning("run_regimes: no results for %s, skipping", method)
            continue
        r = cand.iloc[0]
        layer, position, width = _row_inputs(r)
        spec = specs[method]
        X, index = load_inputs(cfg, behavior, layer, position, spec, width)
        arr = _eval_arrays(index)
        aux = index["prompt_category"].to_numpy() if spec.needs_aux else None
        train, test = np.flatnonzero(~arr["is_test"]), np.flatnonzero(arr["is_test"])
        pos, neg = train[arr["y"][train] == 1], train[arr["y"][train] == 0]

        for regime, value in settings:
            for rep in range(n_repeats):
                rng = np.random.RandomState(derive_seed(cfg["seed"], "regime", regime, value, rep) % (2**32))
                y = arr["y"].copy()
                if regime == "scarcity":
                    if value > len(train):
                        continue
                    n_pos = int(np.clip(round(value * len(pos) / len(train)), 1, min(len(pos), value - 1)))
                    keep = np.concatenate([rng.choice(pos, n_pos, replace=False),
                                           rng.choice(neg, min(len(neg), value - n_pos), replace=False)])
                elif regime == "imbalance":
                    n_pos = max(1, min(len(pos), int(round(value * len(neg) / (1 - value)))))
                    keep = np.concatenate([rng.choice(pos, n_pos, replace=False), neg])
                else:
                    keep = train
                    flip = rng.choice(train, int(round(value * len(train))), replace=False)
                    y[flip] = 1 - y[flip]
                rows_idx = np.sort(np.concatenate([keep, test]))
                sub = dict(y=y[rows_idx], groups=arr["groups"][rows_idx], folds=arr["folds"][rows_idx],
                           is_test=arr["is_test"][rows_idx])
                tr = ~sub["is_test"]
                fallback = not _inner_cv_possible(sub["y"][tr], sub["groups"][tr], sub["folds"][tr], n_inner)
                out = evaluate_method(
                    _take(X, rows_idx), **sub, spec=spec, seed=derive_seed(cfg["seed"], "regime", rep),
                    n_inner=n_inner, n_boot=n_boot, aux=None if aux is None else aux[rows_idx],
                    fixed={"C": 1.0, "k": 16} if fallback else None,
                )
                rows.append({**_identity(behavior, method, layer, position, width), "regime": regime,
                             "value": float(value), "repeat": rep, "hp_fallback": fallback, **out.row})

    out_df = pd.DataFrame(rows)
    write_parquet(out_df, paths(cfg, behavior).results_dir / "regimes.parquet")
    return out_df


# --------------------------------------------------------------------------- runtime monitor demo (§13.5)

class _StopForward(Exception):
    """Raised by the monitor hook to end the forward pass right after the monitored layer."""


def _is_sae_probe(probe: Any) -> bool:
    if isinstance(probe, ShuffledLabelProbe):
        probe = probe.base
    return isinstance(probe, SAE_PROBES)


@torch.inference_mode()
def monitor_scores(bundle: ModelBundle, probe, layer: int, position: str, texts: list[str],
                   sae: JumpReLUSAE | None = None, batch_size: int = 8) -> np.ndarray:
    """Prompt-only forward that stops after `layer` (a hook raises an internal exception caught
    here), extracts `position`, encodes with `sae` if the probe is an SAE probe, returns
    probe.decision_function."""
    if _is_sae_probe(probe) and sae is None:
        raise ValueError("an SAE probe needs `sae`")
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append((output[0] if isinstance(output, (tuple, list)) else output).detach())
        raise _StopForward

    feats = []
    formatted = [model_lib.format_prompt(bundle.tokenizer, t) for t in texts]
    for start in range(0, len(formatted), batch_size):
        enc = model_lib.tokenize(bundle, formatted[start:start + batch_size], max_len=10**6)
        captured.clear()
        handle = bundle.layers[layer].register_forward_hook(hook)
        try:
            bundle.model(
                input_ids=enc["input_ids"].to(bundle.device), attention_mask=enc["attention_mask"].to(bundle.device),
                position_ids=enc["position_ids"].to(bundle.device), use_cache=False,
            )
        except _StopForward:
            pass
        finally:
            handle.remove()
        h = captured[0].float()
        for b in range(h.shape[0]):
            pos = model_lib.find_positions(enc["input_ids"][b], enc["attention_mask"][b], bundle.ids)
            if position == "mean_user":
                a, e = pos["user_span"]
                feats.append(h[b, a:e].mean(0).cpu().numpy())
            else:
                feats.append(h[b, pos[position]].cpu().numpy())
    X = np.stack(feats).astype(np.float32)
    if _is_sae_probe(probe):
        return np.asarray(probe.decision_function(sae_lib.encode_matrix(sae, X)), dtype=np.float64)
    return np.asarray(probe.decision_function(X), dtype=np.float64)


def _sync(bundle: ModelBundle) -> None:
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()


def monitor_latency(bundle: ModelBundle, cfg: Config, behavior: str, method: str,
                    n: int = 50) -> dict:
    """On n test prompts: median monitor ms vs. median generation ms, overhead ratio,
    flagged-vs-observed confusion matrix. Asserts monitor scores match the saved test scores
    (|Δ| < 1e-3·std)."""
    p = paths(cfg, behavior)
    best = best_layers(read_parquet(p.results_dir / "probe_results.parquet"))
    cand = best[best["method"] == method].sort_values("cv_auroc_mean", ascending=False)
    if cand.empty:
        raise ValueError(f"no probe results for method {method!r}")
    r = cand.iloc[0]
    layer, position, width = _row_inputs(r)
    if layer is None:
        raise ValueError(f"{method!r} does not read activations; nothing to monitor")
    probe, _ = load_probe(p.results_dir / "probes" / f"{r['key']}.joblib")
    sae = sae_lib.load_sae(cfg, layer, width) if width is not None else None

    df = read_parquet(p.dataset)
    test = df[df["split"] == "test"].sort_values("prompt_id").head(n)
    texts = test["text"].tolist()
    saved = _load_scores(cfg, behavior, r["key"]).set_index("prompt_id")["score"]
    max_new = cfg.get(f"generation.max_new_tokens.{behavior}")

    try:
        monitor_scores(bundle, probe, layer, position, texts[:1], sae)  # warm-up
        monitor_ms, gen_ms, scores = [], [], []
        for text in texts:
            _sync(bundle)
            t0 = time.perf_counter()
            scores.append(float(monitor_scores(bundle, probe, layer, position, [text], sae)[0]))
            _sync(bundle)
            monitor_ms.append(1000 * (time.perf_counter() - t0))
            t0 = time.perf_counter()
            model_lib.generate(bundle, [text], max_new, batch_size=1, show_progress=False)
            _sync(bundle)
            gen_ms.append(1000 * (time.perf_counter() - t0))
    finally:
        if sae is not None:
            sae_lib.unload(sae)

    scores_arr = np.array(scores)
    ref = saved.loc[test["prompt_id"]].to_numpy()
    max_diff = float(np.abs(scores_arr - ref).max())
    tol = 1e-3 * float(saved.std())
    assert max_diff < tol, f"monitor scores differ from the saved test scores: max|Δ| {max_diff:.3g} ≥ {tol:.3g}"

    flagged = scores_arr > probe.threshold_
    observed = test["label"].to_numpy().astype(int) == 1
    return {
        "key": r["key"], "n": len(texts),
        "monitor_ms_median": float(np.median(monitor_ms)),
        "generation_ms_median": float(np.median(gen_ms)),
        "overhead_ratio": float(np.median(monitor_ms) / np.median(gen_ms)),
        "confusion_matrix": {
            "tp": int((flagged & observed).sum()), "fp": int((flagged & ~observed).sum()),
            "fn": int((~flagged & observed).sum()), "tn": int((~flagged & ~observed).sum()),
        },
        "max_abs_score_diff": max_diff,
    }
