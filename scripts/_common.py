"""Shared argparse plumbing for the CLI entry points (no research logic)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # lets `python scripts/x.py` work without `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "model_b.yaml"), help="experiment YAML")
    parser.add_argument("--behavior", nargs="+", default=None, help="behavior(s); default: all in the config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="dotted config override, value parsed as YAML (repeatable)")
    return parser


def parse_overrides(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def setup(args: argparse.Namespace) -> tuple[Config, list[str]]:
    """Returns (config with --set overrides applied, behaviors to process)."""
    cfg = load_config(args.config, **parse_overrides(args.overrides))
    return cfg, list(args.behavior) if args.behavior else list(cfg["behaviors"])
