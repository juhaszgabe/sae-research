"""Seeds, IO, a simple JSONL cache, logging and the fixed output paths (spec §4.3)."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import random
import re
import subprocess
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import Config


# --------------------------------------------------------------------------- seeds / logging

def derive_seed(seed: int, *names: str | int) -> int:
    """int(sha256("|".join(map(str, (seed, *names)))).hexdigest()[:8], 16). Every random op uses one."""
    key = "|".join(map(str, (seed, *names)))
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    """Logger "sbm.<name>" at INFO; the single handler lives on the "sbm" root."""
    root = logging.getLogger("sbm")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logging.getLogger(f"sbm.{name}")


_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"'})
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """NFKC; curly quotes → ASCII; collapse whitespace; strip; lowercase."""
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    return _WS.sub(" ", s).strip().lower()


# --------------------------------------------------------------------------- IO
# All writes are atomic: write *.tmp then os.replace; parents are created.

def _tmp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".tmp")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    tmp = _tmp(path)
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_dumps(row) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """For incremental caches: appends and flushes, one line per row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(_dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    tmp = _tmp(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    tmp = _tmp(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- paths (§4.3)

def layer_tag(layer: int) -> str:
    """Zero-padded layer index used in file names (7 → "07")."""
    return f"{int(layer):02d}"


def paths(cfg: "Config", behavior: str, pool: str = "main") -> SimpleNamespace:
    """Returns every path in §4.3 for this (model, behavior, pool): .prompts, .generations,
    .dataset, .unused, .ood(shift), .validation_dir, .acts_dir, .codes_dir(width, l0),
    .results_dir, .steering_dir, .figures_dir, .pretest_dir.

    Additionally: .labels (labeler output next to the generations), .pool_dataset (the dataset
    parquet of `pool`: main.parquet or ood_{shift}.parquet) and .side_effects_dir."""
    model = cfg.model.name
    data, art = Path(cfg.data_root), Path(cfg.artifact_root)
    gen_dir = data / "generations" / model / behavior
    ds_dir = data / "datasets" / model / behavior

    def ood(shift: str) -> Path:
        return ds_dir / f"ood_{shift}.parquet"

    def codes_dir(width: str, l0: str | None = None) -> Path:
        l0 = l0 or cfg.get("sae.l0")
        return art / "sae_codes" / model / behavior / pool / f"{width}_{l0}"

    return SimpleNamespace(
        prompts=data / "prompts" / behavior / f"{pool}.jsonl",
        generations=gen_dir / f"{pool}.jsonl",
        labels=gen_dir / f"{pool}_labels.parquet",
        dataset_dir=ds_dir,
        dataset=ds_dir / "main.parquet",
        unused=ds_dir / "unused.parquet",
        ood=ood,
        pool_dataset=ds_dir / f"{pool}.parquet",
        validation_dir=data / "validation" / model / behavior,
        side_effects_dir=data / "side_effects",
        acts_dir=art / "activations" / model / behavior / pool,
        codes_dir=codes_dir,
        results_dir=art / "results" / model / behavior,
        steering_dir=art / "steering" / model / behavior,
        figures_dir=art / "figures" / model,
        pretest_dir=art / "pretest" / model,
    )


# --------------------------------------------------------------------------- cache

class JsonlCache:
    """Tiny append-only cache: key → dict. Loads existing file on init.
    Used for generations (key = prompt_id + generation-params hash) and Neuronpedia responses."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a half-written last line after a crash
                    self._data[rec["key"]] = rec["value"]

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, value: dict) -> None:
        """Appends one line immediately (crash-safe)."""
        append_jsonl(self.path, [{"key": key, "value": value}])
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------- run info

def _version(module: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:
        return None


def env_info() -> dict:
    """python/torch/transformers/sae_lens/sklearn versions, GPU name, git commit."""
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "sae_lens": _version("sae-lens"),
        "sklearn": _version("scikit-learn"),
        "numpy": _version("numpy"),
        "gpu": gpu,
        "git_commit": commit,
    }


def save_run_info(cfg: "Config", out_dir: Path, name: str) -> None:
    """Writes {out_dir}/run_{name}.json = {config: cfg.raw, config_hash, env_info, utc timestamp}.
    Every script calls it; notebooks SHOULD too."""
    write_json(
        Path(out_dir) / f"run_{name}.json",
        {
            "config": cfg.raw,
            "config_hash": cfg.hash(),
            "env_info": env_info(),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
    )
