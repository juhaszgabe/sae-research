"""Probe sweep (all methods × layers × positions × widths) and the follow-up analyses."""

from _common import base_parser, setup

from src import eval as ev
from src.utils import get_logger, paths, save_run_info

log = get_logger("run_probes")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--methods", nargs="+", default=None, help="default: config probes.methods")
    parser.add_argument("--widths", nargs="+", default=None, help="default: config sae.widths")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sweep-only", action="store_true", help="skip comparisons, k-curve and controls")
    args = parser.parse_args()
    cfg, behaviors = setup(args)

    for behavior in behaviors:
        save_run_info(cfg, paths(cfg, behavior).results_dir, "run_probes")
        results = ev.run_probe_sweep(cfg, behavior, methods=args.methods, widths=args.widths, overwrite=args.overwrite)
        best = ev.best_layers(results)
        log.info("%s best layers:\n%s", behavior,
                 best[["method", "width", "layer", "cv_auroc_mean", "test_auroc", "test_lo", "test_hi"]].to_string(index=False))
        if args.sweep_only:
            continue
        reference = cfg.get("eval.reference")
        if reference in set(best["method"]):
            ev.compare_methods(cfg, behavior, results, reference=reference)
        else:
            log.info("%s: no results for the reference %r yet, comparisons skipped", behavior, reference)
        ev.within_category(cfg, behavior, results)
        ev.k_curve(cfg, behavior, results)
        ev.run_controls(cfg, behavior, results)


if __name__ == "__main__":
    main()
