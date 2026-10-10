"""Render every figure whose inputs exist; missing inputs are skipped with a log line."""

import matplotlib.pyplot as plt
from _common import base_parser, setup

from src import plots
from src.eval import best_layers
from src.utils import get_logger, paths, read_parquet, save_run_info

log = get_logger("make_figures")


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg, behaviors = setup(args)

    for behavior in behaviors:
        p = paths(cfg, behavior)
        fig_dir = p.figures_dir
        tables = {
            "results": p.results_dir / "probe_results.parquet",
            "comparisons": p.results_dir / "comparisons.parquet",
            "k_curve": p.results_dir / "k_curve.parquet",
            "within_category": p.results_dir / "within_category.parquet",
            "sweep": p.steering_dir / "sweep.parquet",
            "tradeoff": p.steering_dir / "tradeoff.parquet",
            "coherence": p.steering_dir / "coherence.parquet",
        }
        tables.update({f"shift_{s}": p.results_dir / f"shift_{s}.parquet" for s in cfg.get("shifts", {})})
        data = {}
        for name, path in tables.items():
            if path.exists():
                df = read_parquet(path)
                if len(df):
                    data[name] = df

        def render(name: str, needs: tuple[str, ...], make) -> None:
            missing = [n for n in needs if n not in data]
            if missing:
                log.info("%s: skipping %s (missing or empty: %s)", behavior, name, ", ".join(missing))
                return
            plt.close(make(fig_dir / f"{behavior}_{name}"))
            log.info("%s: wrote %s", behavior, fig_dir / f"{behavior}_{name}.pdf")

        save_run_info(cfg, fig_dir, f"make_figures_{behavior}")
        res = data.get("results")
        has_sae = res is not None and res["method"].isin(["sae_topk", "sae_all", "sae_single"]).any()
        band = None
        if res is not None:
            logreg = best_layers(res).query("method == 'logreg'")
            band = (float(logreg["test_lo"].iloc[0]), float(logreg["test_hi"].iloc[0])) if len(logreg) else None

        render("layer_profile", ("results",), lambda out: plots.plot_layer_profile(res, "last", out))
        render("method_comparison", ("results", "comparisons"),
               lambda out: plots.plot_method_comparison(res, data["comparisons"], out))
        render("k_curve", ("k_curve",), lambda out: plots.plot_k_curve(data["k_curve"], out, band))
        if has_sae:
            render("width", ("results",), lambda out: plots.plot_width(res, out))
        else:
            log.info("%s: skipping width (no SAE probe results)", behavior)
        if band is not None:
            render("position_heatmap", ("results",), lambda out: plots.plot_position_heatmap(res, "logreg", out))
        else:
            log.info("%s: skipping position_heatmap (no logreg results)", behavior)
        render("within_category", ("within_category",),
               lambda out: plots.plot_within_category(data["within_category"], out))
        for direction in ("induce", "suppress"):
            render(f"tradeoff_{direction}", ("tradeoff",),
                   lambda out, d=direction: plots.plot_tradeoff(data["tradeoff"], d, out))
            render(f"steering_curves_{direction}", ("sweep",),
                   lambda out, d=direction: plots.plot_steering_curves(data["sweep"], d, out))
        render("coherence", ("coherence",), lambda out: plots.plot_coherence(data["coherence"], out))
        for shift in cfg.get("shifts", {}):
            render(f"shift_{shift}", (f"shift_{shift}",),
                   lambda out, s=shift: plots.plot_shift(data[f"shift_{s}"], out))


if __name__ == "__main__":
    main()
