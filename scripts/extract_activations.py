"""Extract prompt-only activations at the fixed token positions for every dense layer."""

from _common import base_parser, setup

from src.activations import extract_dataset
from src.model import free_model, load_model
from src.utils import get_logger, paths

log = get_logger("extract_activations")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--pool", default="main", help="main or ood_{shift}")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg, behaviors = setup(args)

    bundle = load_model(cfg)
    try:
        for behavior in behaviors:
            if not paths(cfg, behavior, args.pool).pool_dataset.exists():
                log.info("no dataset for %s/%s, skipping", behavior, args.pool)
                continue
            out = extract_dataset(bundle, cfg, behavior, args.pool, overwrite=args.overwrite)  # writes its run info
            log.info("%s/%s: activations in %s", behavior, args.pool, out)
    finally:
        free_model(bundle)


if __name__ == "__main__":
    main()
