"""Encode the stored activations with the Gemma Scope 2 SAEs (one SAE in memory at a time)."""

from _common import base_parser, setup

from src.sae import encode_dataset
from src.utils import get_logger, paths

log = get_logger("encode_sae")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--widths", nargs="+", default=None, help="e.g. 16k 65k 262k; default: config sae.widths")
    parser.add_argument("--pool", default="main", help="main or ood_{shift}")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg, behaviors = setup(args)

    for behavior in behaviors:
        if not (paths(cfg, behavior, args.pool).acts_dir / "index.parquet").exists():
            log.info("no activations for %s/%s, skipping", behavior, args.pool)
            continue
        quality = encode_dataset(cfg, behavior, args.widths, args.pool, overwrite=args.overwrite)  # writes its run info
        log.info("%s/%s SAE quality:\n%s", behavior, args.pool, quality.to_string(index=False))


if __name__ == "__main__":
    main()
