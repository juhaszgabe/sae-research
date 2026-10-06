"""Steering sweep, side effects (MMLU, NLL/KL), trade-off and detection–steering coherence."""

from _common import base_parser, setup

from src import sae as sae_lib
from src import steering
from src.activations import load_norms
from src.model import free_model, load_model
from src.utils import get_logger, paths, read_parquet, save_run_info

log = get_logger("run_steering")


def steering_width(cfg, results, layer: int) -> str | None:
    """SAE width used for steering: main_width if sae_topk was evaluated with it at (layer, last),
    else the widest evaluated one, else None (no SAE vectors)."""
    rows = results[(results["method"] == "sae_topk") & (results["layer"] == layer) & (results["position"] == "last")]
    widths = set(rows["width"])
    if not widths:
        return None
    main = cfg.get("sae.main_width")
    return main if main in widths else max(widths, key=lambda w: sae_lib.WIDTHS.get(w, 0))


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--skip-side-effects", action="store_true")
    args = parser.parse_args()
    cfg, behaviors = setup(args)

    bundle = load_model(cfg)
    try:
        for behavior in behaviors:
            p = paths(cfg, behavior)
            results_path = p.results_dir / "probe_results.parquet"
            if not results_path.exists():
                log.info("no probe results for %s, skipping (run run_probes.py first)", behavior)
                continue
            save_run_info(cfg, p.steering_dir, "run_steering")
            results = read_parquet(results_path)
            layer = steering.choose_layer(cfg, results)
            width = steering_width(cfg, results, layer)
            n_ref = load_norms(cfg, behavior)[layer]["last"]
            log.info("%s: steering at layer %d, SAE width %s, n_ref %.2f", behavior, layer, width, n_ref)

            sae = a_ref = top1 = None
            if width is not None:
                sae = sae_lib.load_sae(cfg, layer, width, device=str(bundle.device))
                index = read_parquet(p.acts_dir / "index.parquet")
                codes = sae_lib.load_codes(cfg, behavior, layer, "last", width)
                a_ref = sae_lib.feature_max_act(codes, (index["split"] == "train").to_numpy())
                selected, _ = steering.top_features(cfg, behavior, layer, width)
                top1 = selected[0] if selected else None
            try:
                vectors = steering.build_vectors(cfg, behavior, layer, results, sae)
                prompt_sets = steering.eval_prompt_sets(cfg, behavior)
                specs = steering.make_specs(cfg, behavior, layer, top1)
                sweep = steering.run_steering_sweep(bundle, cfg, behavior, specs, prompt_sets, vectors, n_ref, sae, a_ref)
                if not args.skip_side_effects:
                    side = steering.run_side_effects(bundle, cfg, behavior, specs, vectors, n_ref, sae, a_ref)
                    steering.tradeoff(sweep, side, out=p.steering_dir / "tradeoff.parquet")
                    steering.direction_coherence(sweep, side, out=p.steering_dir / "direction_coherence.parquet")
                if sae is not None and top1 is not None:
                    steering.feature_coherence(bundle, cfg, behavior, sae, results, prompt_sets, n_ref, a_ref)
            finally:
                if sae is not None:
                    sae_lib.unload(sae)
    finally:
        free_model(bundle)


if __name__ == "__main__":
    main()
