"""All figures. Every function returns a matplotlib Figure and saves PDF + PNG (300 dpi) when `out` is given."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .eval import best_layers  # noqa: E402
from .sae import WIDTHS  # noqa: E402

STYLE = {"dim": ("#1b9e77", "o"), "logreg": ("#d95f02", "s"), "dense_topk": ("#7570b3", "^"),
         "refusal_dir": ("#a6761d", "v"), "category_only": ("#666666", "x"),
         "sae_topk": ("#e7298a", "D"), "sae_all": ("#66a61e", "P"), "sae_single": ("#e6ab02", "*"),
         "probe_dir": ("#d95f02", "s"), "sae_top1": ("#e7298a", "D"), "sae_combo5": ("#66a61e", "P"),
         "random0": ("#999999", "."), "prompt": ("#000000", "X")}

_DEFAULT = ("#444444", "o")
_SAE_METHODS = ("sae_topk", "sae_all", "sae_single")


def _style(name: str) -> tuple[str, str]:
    return STYLE.get(name, _DEFAULT)


def _save(fig: Figure, out: Path | None) -> Figure:
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=300)
    return fig


def _label(method: str, width: str) -> str:
    return method if width in ("dense", "-") else f"{method} ({width})"


def _width_order(widths) -> list[str]:
    return sorted(set(widths), key=lambda w: WIDTHS.get(w, 0))


def plot_layer_profile(results: pd.DataFrame, position: str, out: Path | None = None) -> Figure:
    """CV AUROC ± std vs layer; dense lines, SAE markers, dashed category_only (report figure 1)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sub = results[results["position"] == position]
    for (method, width), g in sub.groupby(["method", "width"], sort=True):
        g = g.sort_values("layer")
        color, marker = _style(method)
        if method in _SAE_METHODS:
            ax.errorbar(g["layer"], g["cv_auroc_mean"], yerr=g["cv_auroc_std"], fmt=marker, color=color, capsize=2,
                        alpha=0.9, label=_label(method, width))
        else:
            ax.plot(g["layer"], g["cv_auroc_mean"], color=color, marker=marker, markersize=3, label=method)
            ax.fill_between(g["layer"], g["cv_auroc_mean"] - g["cv_auroc_std"], g["cv_auroc_mean"] + g["cv_auroc_std"],
                            color=color, alpha=0.15, linewidth=0)
    cat = results[results["method"] == "category_only"]
    if not cat.empty:
        ax.axhline(cat["cv_auroc_mean"].iloc[0], color=_style("category_only")[0], linestyle="--", label="category_only")
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
    ax.set(xlabel="layer", ylabel="CV AUROC (mean ± std over folds)", title=f"Layer profile — position '{position}'")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, out)


def plot_method_comparison(results: pd.DataFrame, comparisons: pd.DataFrame, out: Path | None = None) -> Figure:
    """Test AUROC + CI per method at best layer, Δ vs logreg with Holm p (report figure 2)."""
    best = best_layers(results).sort_values("test_auroc", ascending=False).reset_index(drop=True)
    headline = comparisons[comparisons["family"] == "headline"].set_index("key_a") if len(comparisons) else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in best.iterrows():
        color, marker = _style(r["method"])
        ax.errorbar(i, r["test_auroc"], yerr=[[r["test_auroc"] - r["test_lo"]], [r["test_hi"] - r["test_auroc"]]],
                    fmt=marker, color=color, capsize=3, markersize=7)
        if r["key"] in headline.index:
            c = headline.loc[r["key"]]
            ax.annotate(f"Δ={c['delta']:+.3f}\np={c['p_holm']:.3f}", (i, r["test_hi"]), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=6)
    labels = [f"{_label(r['method'], r['width'])}\nL{r['layer']}" if r["layer"] >= 0 else r["method"] for _, r in best.iterrows()]
    ax.set_xticks(range(len(best)), labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
    ax.set(ylabel="test AUROC (95 % cluster-bootstrap CI)", title="Methods at their best layer (Δ vs reference, Holm p)")
    return _save(fig, out)


def plot_tradeoff(tradeoff_df: pd.DataFrame, direction: str, out: Path | None = None) -> Figure:
    """Effect vs MMLU drop, Pareto front, prompting and random highlighted (report figure 3)."""
    df = tradeoff_df[(tradeoff_df["direction"] == direction) & (tradeoff_df["kind"] != "none")]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    seen = set()
    for _, r in df.iterrows():
        name = "prompt" if r["kind"] == "prompt" else str(r["vector"])
        color, marker = _style("random0" if name.startswith("random") else name)
        highlight = name == "prompt" or name.startswith("random")
        ax.scatter(100 * r["cost"], r["effect"], color=color, marker=marker, s=70 if highlight else 35,
                   edgecolors="black" if highlight else "none", linewidths=0.8, alpha=0.85,
                   facecolors="none" if r["kind"] == "ablate" else color,
                   label=None if name in seen else name, zorder=3 if highlight else 2)
        seen.add(name)
    front = df[df["pareto"]].sort_values("cost")
    ax.plot(100 * front["cost"], front["effect"], color="black", linewidth=0.8, linestyle="--", label="Pareto front", zorder=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set(xlabel="MMLU accuracy drop (points)", ylabel=f"target effect ({direction}): Δ behavior rate",
           title=f"Steering trade-off — {direction}")
    ax.legend(fontsize=7)
    return _save(fig, out)


def plot_k_curve(k_curve_df: pd.DataFrame, out: Path | None = None, logreg_band: tuple[float, float] | None = None) -> Figure:
    """AUROC vs k (log₂), sae_topk vs dense_topk, logreg band (pass its (lo, hi) test CI)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for (method, width), g in k_curve_df.groupby(["method", "width"], sort=True):
        g = g.sort_values("k")
        color, marker = _style(method)
        ax.errorbar(g["k"], g["test_auroc"], yerr=[g["test_auroc"] - g["test_lo"], g["test_hi"] - g["test_auroc"]],
                    color=color, marker=marker, capsize=2, label=_label(method, width),
                    linestyle="-" if width in ("dense", _width_order(k_curve_df["width"])[-1]) else ":")
    if logreg_band is not None:
        ax.axhspan(*logreg_band, color=_style("logreg")[0], alpha=0.15, label="logreg (test CI)")
    ax.set_xscale("log", base=2)
    ax.set(xlabel="k (number of features)", ylabel="test AUROC", title="AUROC vs number of selected features")
    ax.legend(fontsize=7)
    return _save(fig, out)


def plot_width(results: pd.DataFrame, out: Path | None = None) -> Figure:
    """Test AUROC by SAE width."""
    best = best_layers(results)
    best = best[best["method"].isin(_SAE_METHODS)]
    widths = _width_order(best["width"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for method, g in best.groupby("method", sort=True):
        g = g.set_index("width").reindex(widths).dropna(subset=["test_auroc"])
        color, marker = _style(method)
        x = [widths.index(w) for w in g.index]
        ax.errorbar(x, g["test_auroc"], yerr=[g["test_auroc"] - g["test_lo"], g["test_hi"] - g["test_auroc"]],
                    color=color, marker=marker, capsize=3, label=method)
    ax.set_xticks(range(len(widths)), widths)
    ax.set(xlabel="SAE width", ylabel="test AUROC (best layer)", title="SAE probes by dictionary width")
    ax.legend(fontsize=7)
    return _save(fig, out)


def _heatmap(ax, table: pd.DataFrame, cbar_label: str, vmin: float = 0.5, vmax: float = 1.0):
    im = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=7)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < (vmin + vmax) / 2 else "black")
    ax.figure.colorbar(im, ax=ax, label=cbar_label)


def plot_position_heatmap(results: pd.DataFrame, method: str = "logreg", out: Path | None = None) -> Figure:
    """Layer × position CV AUROC."""
    sub = results[results["method"] == method]
    table = sub.pivot_table(index="position", columns="layer", values="cv_auroc_mean", aggfunc="max")
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * table.shape[1] + 2), 0.5 * table.shape[0] + 2))
    _heatmap(ax, table, "CV AUROC")
    ax.set(xlabel="layer", title=f"{method}: CV AUROC by layer and token position")
    return _save(fig, out)


def plot_within_category(wc_df: pd.DataFrame, out: Path | None = None) -> Figure:
    """Category × method AUROC."""
    df = wc_df.assign(label=[_label(m, w) for m, w in zip(wc_df["method"], wc_df["width"])])
    table = df.pivot_table(index="prompt_category", columns="label", values="auroc", aggfunc="first", dropna=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * table.shape[1] + 2), 0.5 * table.shape[0] + 2))
    _heatmap(ax, table, "test AUROC", vmin=0.3)
    ax.set(title="Within-category test AUROC (blank: < 10 per class)")
    return _save(fig, out)


def plot_steering_curves(sweep: pd.DataFrame, direction: str, out: Path | None = None) -> Figure:
    """Behavior rate vs α per vector."""
    df = sweep[sweep["direction"] == direction]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for vector, g in df[df["kind"] == "add"].groupby("vector", sort=True):
        g = g.assign(abs_alpha=g["alpha"].abs()).sort_values("abs_alpha")
        color, marker = _style("random0" if str(vector).startswith("random") else str(vector))
        ax.errorbar(g["abs_alpha"], g["behavior_rate"],
                    yerr=[g["behavior_rate"] - g["rate_lo"], g["rate_hi"] - g["behavior_rate"]],
                    color=color, marker=marker, capsize=2, label=vector)
    none = df[df["kind"] == "none"]
    if not none.empty:
        ax.axhline(none["behavior_rate"].iloc[0], color="black", linestyle=":", linewidth=0.8, label="unsteered")
    prompt = df[df["kind"] == "prompt"]
    if not prompt.empty:
        ax.axhline(prompt["behavior_rate"].iloc[0], color=_style("prompt")[0], linestyle="--", linewidth=0.8, label="prompt")
    ax.set(xlabel="|α| (fraction of the mean residual norm)", ylabel="behavior rate", ylim=(-0.03, 1.03),
           title=f"Steering curves — {direction}")
    ax.legend(fontsize=7)
    return _save(fig, out)


def plot_coherence(coh_df: pd.DataFrame, out: Path | None = None) -> Figure:
    """Detection vs steering effect per feature."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for ax, name, sign in ((axes[0], "induce", 1.0), (axes[1], "suppress", -1.0)):
        for is_control, g in coh_df.groupby("is_control"):
            ax.scatter(g["detection"], sign * g[f"{name}_effect"], color="#999999" if is_control else _style("sae_topk")[0],
                       marker="." if is_control else "D", s=45, label="random eligible" if is_control else "top sae_topk")
        rho = coh_df[f"spearman_{name}"].iloc[0] if len(coh_df) else np.nan
        lo, hi = (coh_df[f"spearman_{name}_lo"].iloc[0], coh_df[f"spearman_{name}_hi"].iloc[0]) if len(coh_df) else (np.nan, np.nan)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set(xlabel="detection: max(AUROC, 1 − AUROC) on test", ylabel=f"{name} effect (Δ behavior rate)",
               title=f"{name}: ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]")
        ax.legend(fontsize=7)
    return _save(fig, out)


def plot_shift(shift_df: pd.DataFrame, out: Path | None = None) -> Figure:
    """[TDK] ID vs OOD AUROC slope chart (each method at its best in-distribution layer)."""
    df = shift_df.dropna(subset=["ood_auroc"])
    df = df.sort_values("id_auroc", ascending=False).groupby(["method", "width"], sort=True).head(1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for _, r in df.iterrows():
        color, marker = _style(r["method"])
        ax.plot([0, 1], [r["id_auroc"], r["ood_auroc"]], color=color, marker=marker, label=_label(r["method"], r["width"]))
        ax.errorbar(1, r["ood_auroc"], yerr=[[r["ood_auroc"] - r["ood_lo"]], [r["ood_hi"] - r["ood_auroc"]]],
                    color=color, capsize=3)
    ax.set_xticks([0, 1], ["in-distribution (test)", "shifted"])
    ax.set_xlim(-0.3, 1.3)
    shift = df["shift"].iloc[0] if len(df) and "shift" in df.columns else ""
    ax.set(ylabel="AUROC", title=f"Distribution shift — {shift}")
    ax.legend(fontsize=7)
    return _save(fig, out)
