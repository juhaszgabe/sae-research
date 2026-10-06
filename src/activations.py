"""Activation extraction at fixed token positions, storage, and hook sanity checks.

Model-agnostic: only uses ModelBundle, find_positions and capture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import Config
from .model import ModelBundle, capture, find_positions, format_prompt, generate, tokenize
from .utils import get_logger, layer_tag, paths, read_json, read_parquet, save_run_info, write_json, write_parquet

log = get_logger("activations")

POSITIONS = ("user_last", "eot_user", "turn_start", "model_tag", "last")
POOLED = ("mean_user",)

_INDEX_COLUMNS = ("prompt_id", "split", "fold", "label", "group_id", "prompt_category")


def _forward(bundle: ModelBundle, enc: dict[str, torch.Tensor], **kwargs):
    return bundle.model(
        input_ids=enc["input_ids"].to(bundle.device),
        attention_mask=enc["attention_mask"].to(bundle.device),
        position_ids=enc["position_ids"].to(bundle.device),
        use_cache=False,
        **kwargs,
    )


@torch.inference_mode()
def extract(bundle: ModelBundle, user_texts: list[str], layers: Sequence[int],
            positions: Sequence[str] = POSITIONS, pooled: Sequence[str] = POOLED,
            batch_size: int = 8, max_len: int = 1024) -> dict[int, dict[str, np.ndarray]]:
    """Prompt-only forward passes (use_cache=False, explicit position_ids) with `capture`.
    Returns {layer: {name: float32 [N, D]}} in input order. `mean_user` = mean over user_span.
    Raises on NaN/Inf."""
    unknown = [p for p in positions if p not in POSITIONS] + [p for p in pooled if p not in POOLED]
    if unknown:
        raise ValueError(f"unknown positions {unknown}; available: {POSITIONS + POOLED}")
    layers = [int(x) for x in layers]
    names = [*positions, *pooled]
    n = len(user_texts)
    d = bundle.spec.d_model
    out = {layer: {name: np.zeros((n, d), dtype=np.float32) for name in names} for layer in layers}

    formatted = [format_prompt(bundle.tokenizer, t) for t in user_texts]
    lengths = [len(x) for x in bundle.tokenizer(formatted, add_special_tokens=False)["input_ids"]]
    order = sorted(range(n), key=lambda i: -lengths[i])

    for start in tqdm(range(0, n, batch_size), desc="extract", disable=n <= batch_size):
        idx = order[start:start + batch_size]
        enc = tokenize(bundle, [formatted[i] for i in idx], max_len)
        pos = [find_positions(enc["input_ids"][b], enc["attention_mask"][b], bundle.ids) for b in range(len(idx))]
        with capture(bundle, layers) as store:
            _forward(bundle, enc)
        rows = torch.arange(len(idx), device=bundle.device)
        for layer in layers:
            h = store[layer][0]
            for name in positions:
                cols = torch.tensor([p[name] for p in pos], device=bundle.device)
                out[layer][name][idx] = h[rows, cols].float().cpu().numpy()
            if "mean_user" in pooled:
                means = torch.stack([h[b, a:e].float().mean(0) for b, (a, e) in enumerate(p["user_span"] for p in pos)])
                out[layer]["mean_user"][idx] = means.cpu().numpy()

    for layer in layers:
        for name, arr in out[layer].items():
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"NaN/Inf in activations at layer {layer}, position {name}")
    return out


def extract_dataset(bundle: ModelBundle, cfg: Config, behavior: str, pool: str = "main",
                    overwrite: bool = False) -> Path:
    """Reads the dataset parquet for `pool`, runs `extract` on layers_dense, writes
    acts_dir/layer_{LL}.npz (np.savez, one float32 array per position/pooled name),
    acts_dir/index.parquet (row, prompt_id, split, fold, label, group_id, prompt_category),
    acts_dir/norms.json ({layer: {position: mean L2 norm over TRAIN rows}}), run info.
    Skips if all files exist and index prompt_ids match the dataset (unless overwrite)."""
    p = paths(cfg, behavior, pool)
    df = read_parquet(p.pool_dataset).reset_index(drop=True)
    layers = list(cfg.model.layers_dense)
    acts_dir: Path = p.acts_dir
    files = [acts_dir / f"layer_{layer_tag(layer)}.npz" for layer in layers]
    index_path, norms_path = acts_dir / "index.parquet", acts_dir / "norms.json"

    if not overwrite and all(f.exists() for f in (*files, index_path, norms_path)):
        if read_parquet(index_path)["prompt_id"].tolist() == df["prompt_id"].tolist():
            log.info("activations for %s/%s already extracted, skipping", behavior, pool)
            return acts_dir
        log.warning("existing activation index does not match the dataset; re-extracting")

    acts = extract(
        bundle, df["text"].tolist(), layers,
        positions=cfg.get("extraction.positions"), pooled=cfg.get("extraction.pooled"),
        batch_size=cfg.get("extraction.batch_size"), max_len=cfg.get("extraction.max_prompt_tokens"),
    )

    # Norms over TRAIN rows; OOD pools have none, so they fall back to all rows.
    train = (df["split"] == "train").to_numpy()
    ref_rows = train if train.any() else np.ones(len(df), dtype=bool)
    norms: dict[str, dict[str, float]] = {}
    acts_dir.mkdir(parents=True, exist_ok=True)
    for layer, file in zip(layers, files):
        tmp = file.with_name(file.stem + ".tmp.npz")
        np.savez(tmp, **acts[layer])
        tmp.replace(file)
        norms[str(layer)] = {
            name: float(np.linalg.norm(arr[ref_rows], axis=1).mean()) for name, arr in acts[layer].items()
        }
    write_json(norms_path, norms)

    index = df[[c for c in _INDEX_COLUMNS if c in df.columns]].copy()
    index.insert(0, "row", np.arange(len(df)))
    write_parquet(index, index_path)
    save_run_info(cfg, acts_dir, "extract_activations")
    log.info("wrote activations for %d prompts × %d layers to %s", len(df), len(layers), acts_dir)
    return acts_dir


def load_acts(cfg: Config, behavior: str, layer: int, position: str,
              pool: str = "main") -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (X float32 [N, D], index DataFrame aligned with X)."""
    acts_dir = paths(cfg, behavior, pool).acts_dir
    with np.load(acts_dir / f"layer_{layer_tag(layer)}.npz") as z:
        if position not in z.files:
            raise KeyError(f"position {position!r} not stored for layer {layer} (available: {z.files})")
        X = z[position].astype(np.float32, copy=False)
    index = read_parquet(acts_dir / "index.parquet")
    if len(index) != len(X):
        raise ValueError(f"activation rows ({len(X)}) != index rows ({len(index)}) in {acts_dir}")
    return X, index


def load_norms(cfg: Config, behavior: str) -> dict[int, dict[str, float]]:
    raw = read_json(paths(cfg, behavior).acts_dir / "norms.json")
    return {int(layer): dict(v) for layer, v in raw.items()}


# --------------------------------------------------------------------------- sanity checks (scripts/pretest.py)

@torch.inference_mode()
def check_hooks_vs_hidden_states(bundle: ModelBundle, texts: list[str], atol: float = 1e-3) -> dict:
    """Compare captured layer L output with output_hidden_states[L+1] for all L < n_layers-1.
    Returns {"max_abs_diff": float, "ok": bool, "median_norm_per_layer": list[float]}."""
    n_layers = bundle.spec.n_layers
    formatted = [format_prompt(bundle.tokenizer, t) for t in texts]
    enc = tokenize(bundle, formatted, max_len=10**6)
    with capture(bundle, range(n_layers)) as store:
        out = _forward(bundle, enc, output_hidden_states=True)
    mask = enc["attention_mask"].bool().to(bundle.device)

    max_diff, norms = 0.0, []
    for layer in range(n_layers):
        h = store[layer][0].float()
        norms.append(float(h[mask].norm(dim=-1).median()))
        # hidden_states[-1] is post-final-norm, so the last layer is not comparable.
        if layer < n_layers - 1:
            ref = out.hidden_states[layer + 1].float()
            max_diff = max(max_diff, float((h - ref)[mask].abs().max()))
    return {"max_abs_diff": max_diff, "ok": bool(max_diff <= atol), "median_norm_per_layer": norms}


def check_padding_equivalence(bundle: ModelBundle, texts: list[str], layer: int) -> dict:
    """`last` vectors from one left-padded batch vs. batch_size=1 runs.
    ok if max|Δ| / mean|h| < 1e-2 (bf16) or < 1e-5 (float32)."""
    batched = extract(bundle, texts, [layer], positions=("last",), pooled=(), batch_size=len(texts), max_len=10**6)
    single = extract(bundle, texts, [layer], positions=("last",), pooled=(), batch_size=1, max_len=10**6)
    a, b = batched[layer]["last"], single[layer]["last"]
    rel = float(np.abs(a - b).max() / (np.abs(b).mean() + 1e-12))
    dtype = next(bundle.model.parameters()).dtype
    tol = 1e-5 if dtype in (torch.float32, torch.float64) else 1e-2
    return {"rel_max_diff": rel, "tol": tol, "ok": bool(rel < tol)}


def check_generation_equivalence(bundle: ModelBundle, texts: list[str], n_tokens: int = 20) -> dict:
    """Batched vs. one-by-one greedy generation; ok if ≥ 95 % of prompts share the first n_tokens.
    If not ok → recommend generation.batch_size = 1."""
    batched = generate(bundle, texts, n_tokens, batch_size=len(texts), show_progress=False)
    single = generate(bundle, texts, n_tokens, batch_size=1, show_progress=False)
    same = [a["response_token_ids"][:n_tokens] == b["response_token_ids"][:n_tokens] for a, b in zip(batched, single)]
    frac = float(np.mean(same)) if same else 1.0
    ok = frac >= 0.95
    return {
        "frac_identical": frac,
        "ok": bool(ok),
        "recommendation": None if ok else "set generation.batch_size = 1",
    }
