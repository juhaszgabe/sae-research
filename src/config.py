"""Load the experiment YAML into a small frozen config object."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

_MISSING = ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    auto_class: Literal["causal_lm", "image_text_to_text"]
    n_layers: int
    d_model: int
    layers_dense: tuple[int, ...]
    layers_sae: tuple[int, ...]
    sae_release: str
    sae_release_all: str


@dataclass(frozen=True)
class Config:
    raw: dict  # the merged YAML dict (after overrides)
    model: ModelSpec  # resolved from model.name + model_registry
    data_root: Path
    artifact_root: Path

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """cfg.get("probes.k_grid"). Raises KeyError if missing and no default."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"config key not found: {dotted!r}")
                return default
            node = node[part]
        return node

    def __getitem__(self, section: str) -> Any:
        return self.raw[section]

    def to_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False, allow_unicode=True)
        os.replace(tmp, path)

    def hash(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = d
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _resolve_model(raw: dict) -> ModelSpec:
    name = raw["model"]["name"]
    registry = raw.get("model_registry", {})
    if name not in registry:
        raise ValueError(f"model.name {name!r} not in model_registry (available: {sorted(registry)})")
    entry = registry[name]
    n_layers = int(entry["n_layers"])

    layers_dense = entry["layers_dense"]
    if layers_dense == "all":
        layers_dense = list(range(n_layers))
        entry["layers_dense"] = layers_dense
    layers_dense = tuple(int(x) for x in layers_dense)
    layers_sae = tuple(int(x) for x in entry["layers_sae"])

    bad = [x for x in (*layers_dense, *layers_sae) if not 0 <= x < n_layers]
    if bad:
        raise ValueError(f"{name}: layers {sorted(set(bad))} outside [0, {n_layers})")
    missing = sorted(set(layers_sae) - set(layers_dense))
    if missing:
        raise ValueError(f"{name}: layers_sae {missing} not in layers_dense")

    # R5: an IT model must be paired with IT SAEs (and a PT model with PT SAEs).
    is_it = name.endswith("-it")
    for key in ("sae_release", "sae_release_all"):
        if ("-it-" in entry[key]) != is_it:
            raise ValueError(
                f"{name}: {key}={entry[key]!r} does not match the model variant "
                f"({'IT' if is_it else 'PT'} model needs {'IT' if is_it else 'PT'} SAEs)"
            )
    if entry["auto_class"] not in ("causal_lm", "image_text_to_text"):
        raise ValueError(f"{name}: unknown auto_class {entry['auto_class']!r}")

    return ModelSpec(
        name=name,
        hf_id=entry["hf_id"],
        auto_class=entry["auto_class"],
        n_layers=n_layers,
        d_model=int(entry["d_model"]),
        layers_dense=layers_dense,
        layers_sae=layers_sae,
        sae_release=entry["sae_release"],
        sae_release_all=entry["sae_release_all"],
    )


def load_config(path: str | Path, **overrides: Any) -> Config:
    """Loads YAML, applies dotted overrides (e.g. {"model.name": "gemma-3-4b-it"}),
    resolves "layers_dense: all" to range(n_layers), applies env SBM_DATA_ROOT / SBM_ARTIFACT_ROOT.
    Validates: model.name in model_registry; layers_sae ⊆ layers_dense; all layers in [0, n_layers);
    '-it-' in sae_release iff model name ends with '-it' (R5)."""
    with open(path, encoding="utf-8") as f:
        raw = copy.deepcopy(yaml.safe_load(f))
    for dotted, value in overrides.items():
        _set_dotted(raw, dotted, value)

    paths = raw.setdefault("paths", {})
    paths["data_root"] = os.environ.get("SBM_DATA_ROOT") or paths.get("data_root", "data")
    paths["artifact_root"] = os.environ.get("SBM_ARTIFACT_ROOT") or paths.get("artifact_root", "artifacts")

    return Config(
        raw=raw,
        model=_resolve_model(raw),
        data_root=Path(paths["data_root"]),
        artifact_root=Path(paths["artifact_root"]),
    )
