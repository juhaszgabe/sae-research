"""Generate the model's responses for a prompt pool and label them automatically."""

import pandas as pd
from _common import base_parser, setup

from src import data, labeling
from src.model import free_model, load_model
from src.utils import get_logger, paths, save_run_info, write_parquet

log = get_logger("generate_and_label")


def _label_and_save(behavior: str, pool: str, prompts: list[dict], generations: pd.DataFrame, out) -> None:
    rows = pd.DataFrame(prompts)[["prompt_id", "json_schema"]].merge(generations, on="prompt_id")
    labels = labeling.label_frame(behavior, rows)
    write_parquet(labels, out)
    log.info("%s/%s: %d generations, positive rate %.3f, ambiguous %.3f (labeler %s)", behavior, pool,
             len(labels), labels["label"].mean(), labels["ambiguous"].mean(), labeling.LABELER_VERSION)


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--pool", default="main", help="main or ood_{shift}")
    parser.add_argument("--relabel-only", action="store_true",
                        help="do not load the model or generate: re-apply the current labeler to the cached generations")
    args = parser.parse_args()
    cfg, behaviors = setup(args)

    if args.relabel_only:
        for behavior in behaviors:
            p = paths(cfg, behavior, args.pool)
            if not p.prompts.exists() or not p.generations.exists():
                log.info("no prompts or generations for %s/%s, skipping", behavior, args.pool)
                continue
            prompts = data.load_prompts(p.prompts)
            ids = {pr["prompt_id"] for pr in prompts}
            generations = data.load_generations(cfg, behavior, args.pool)
            generations = generations[generations["prompt_id"].isin(ids)]
            if len(generations) < len(ids):
                log.warning("%s/%s: %d of %d prompts have no cached generation (run without --relabel-only)",
                            behavior, args.pool, len(ids) - len(generations), len(ids))
            save_run_info(cfg, p.generations.parent, f"relabel_{args.pool}")
            _label_and_save(behavior, args.pool, prompts, generations, p.labels)
        return

    bundle = load_model(cfg)
    try:
        for behavior in behaviors:
            p = paths(cfg, behavior, args.pool)
            if not p.prompts.exists():
                log.info("no prompts at %s, skipping %s", p.prompts, behavior)
                continue
            save_run_info(cfg, p.generations.parent, f"generate_and_label_{args.pool}")
            prompts = data.load_prompts(p.prompts)
            generations = data.run_generation(bundle, cfg, behavior, args.pool, prompts)
            _label_and_save(behavior, args.pool, prompts, generations, p.labels)
    finally:
        free_model(bundle)


if __name__ == "__main__":
    main()
