"""Build the prompt pools: prompts/{behavior}/main.jsonl or ood_{shift}.jsonl."""

from _common import base_parser, setup

from src import data
from src.utils import get_logger, paths, save_run_info

log = get_logger("build_prompts")

BUILDERS = {
    "refusal": data.build_refusal_prompts,
    "format_break": data.build_format_prompts,
    "hedging": data.build_hedging_prompts,
}


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--pool", choices=["main", "ood"], default="main")
    parser.add_argument("--shift", default=None, help="shift name from the config (required for --pool ood)")
    args = parser.parse_args()
    if args.pool == "ood" and not args.shift:
        parser.error("--pool ood requires --shift NAME")
    cfg, behaviors = setup(args)

    for behavior in behaviors:
        if args.pool == "main":
            prompts, pool = BUILDERS[behavior](cfg), "main"
        else:
            if behavior not in cfg["shifts"][args.shift]["behaviors"]:
                log.info("shift %s is not defined for %s, skipping", args.shift, behavior)
                continue
            prompts, pool = data.build_ood_prompts(cfg, behavior, args.shift, None), f"ood_{args.shift}"
        out = paths(cfg, behavior, pool).prompts
        data.save_prompts(prompts, out)
        save_run_info(cfg, out.parent, f"build_prompts_{pool}")
        log.info("%s/%s: wrote %d prompts to %s", behavior, pool, len(prompts), out)


if __name__ == "__main__":
    main()
