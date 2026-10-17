"""Generate the model's responses for a prompt pool and label them automatically."""

import pandas as pd
from _common import base_parser, setup

from src import data, labeling
from src.model import free_model, load_model
from src.utils import get_logger, paths, save_run_info, write_parquet

log = get_logger("generate_and_label")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--pool", default="main", help="main or ood_{shift}")
    args = parser.parse_args()
    cfg, behaviors = setup(args)

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
            rows = pd.DataFrame(prompts)[["prompt_id", "json_schema"]].merge(generations, on="prompt_id")
            labels = labeling.label_frame(behavior, rows)
            write_parquet(labels, p.labels)
            log.info("%s/%s: %d generations, positive rate %.3f, ambiguous %.3f", behavior, args.pool,
                     len(labels), labels["label"].mean(), labels["ambiguous"].mean())
    finally:
        free_model(bundle)


if __name__ == "__main__":
    main()
