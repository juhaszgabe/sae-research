"""Assemble, balance and split the dataset (or an OOD pool); optionally export the validation sample."""

from _common import base_parser, setup

from src import data
from src.utils import get_logger, paths, read_parquet, save_run_info

log = get_logger("build_dataset")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--shift", default=None, help="build ood_{shift}.parquet instead of main.parquet")
    parser.add_argument("--export-validation", action="store_true",
                        help="also write validation/{model}/{behavior}/annotate.csv and key.csv")
    args = parser.parse_args()
    cfg, behaviors = setup(args)

    for behavior in behaviors:
        p = paths(cfg, behavior)
        if args.shift:
            if behavior not in cfg["shifts"][args.shift]["behaviors"]:
                log.info("shift %s is not defined for %s, skipping", args.shift, behavior)
                continue
            df = data.build_ood_dataset(cfg, behavior, args.shift)
            save_run_info(cfg, p.dataset_dir, f"build_dataset_ood_{args.shift}")
        else:
            df = data.build_dataset(cfg, behavior)
            save_run_info(cfg, p.dataset_dir, "build_dataset")
        log.info("%s: %d rows, positive rate %.3f", behavior, len(df), df["label"].mean())

        if args.export_validation and not args.shift:
            # The sample is drawn before ambiguous rows are excluded, so it covers them too.
            labeled = data.assemble(data.load_prompts(p.prompts), data.load_generations(cfg, behavior),
                                    read_parquet(p.labels))
            data.export_validation_sample(labeled, p.validation_dir, cfg.get("dataset.validation_n"), cfg["seed"])
            log.info("%s: validation sample written to %s", behavior, p.validation_dir)


if __name__ == "__main__":
    main()
