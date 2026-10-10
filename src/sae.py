"""Gemma Scope 2 SAEs: lookup/loading, JumpReLU encode/decode, codes storage, Neuronpedia."""

from __future__ import annotations

import difflib
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import Tensor

from .activations import POSITIONS, load_acts
from .config import Config
from .utils import JsonlCache, get_logger, layer_tag, paths, save_run_info

log = get_logger("sae")

WIDTHS = {"16k": 2**14, "65k": 2**16, "262k": 2**18, "1m": 2**20}

_NEURONPEDIA_API = "https://www.neuronpedia.org/api/feature"
_last_request = 0.0


@dataclass(frozen=True)
class SaeInfo:
    release: str
    sae_id: str
    repo_id: str
    folder: str
    layer: int
    width: str
    l0: str
    l0_target: float
    model_hf_id: str
    neuronpedia_id: str | None


class JumpReLUSAE(torch.nn.Module):
    """Gemma Scope 2 JumpReLU SAE. No input normalization, no b_dec subtraction (spec §5.2)."""

    W_enc: Tensor  # [D, W]
    b_enc: Tensor  # [W]
    W_dec: Tensor  # [W, D]
    b_dec: Tensor  # [D]
    threshold: Tensor  # [W]

    def __init__(self, W_enc: Tensor, b_enc: Tensor, W_dec: Tensor, b_dec: Tensor, threshold: Tensor,
                 info: SaeInfo | None = None):
        super().__init__()
        for name, t in (("W_enc", W_enc), ("b_enc", b_enc), ("W_dec", W_dec), ("b_dec", b_dec), ("threshold", threshold)):
            self.register_buffer(name, t.detach().to(torch.float32).clone())
        self.info = info

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[0]

    @property
    def width(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x: Tensor) -> Tensor:
        """[..., D] → [..., W]; casts x to float32."""
        pre = x.to(torch.float32) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, f: Tensor) -> Tensor:
        """[..., W] → [..., D]"""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def direction(self, feature: int) -> Tensor:
        """W_dec[feature] / ||W_dec[feature]||"""
        w = self.W_dec[feature]
        return w / w.norm()

    def pre_act(self, x: Tensor, feature: int) -> Tensor:
        """x·W_enc[:, feature] + b_enc[feature] (cheap single-feature path for clamping)."""
        return x.to(torch.float32) @ self.W_enc[:, feature] + self.b_enc[feature]


# --------------------------------------------------------------------------- lookup / loading

def lookup_sae(cfg: Config, layer: int, width: str, l0: str, suffix: str = "") -> SaeInfo:
    """Chooses cfg.model.sae_release if layer ∈ layers_sae else sae_release_all.
    sae_id = f"layer_{layer}_width_{width}_l0_{l0}{suffix}". Reads the SAELens registry (§5.2).
    Missing id → ValueError listing difflib close matches."""
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    release = cfg.model.sae_release if layer in cfg.model.layers_sae else cfg.model.sae_release_all
    sae_id = f"layer_{layer}_width_{width}_l0_{l0}{suffix}"
    directory = get_pretrained_saes_directory()
    if release not in directory:
        close = difflib.get_close_matches(release, list(directory), n=5)
        raise ValueError(f"SAE release {release!r} not in the SAELens registry (close matches: {close})")
    entry = directory[release]
    if sae_id not in entry.saes_map:
        close = difflib.get_close_matches(sae_id, list(entry.saes_map), n=8, cutoff=0.5)
        raise ValueError(f"SAE id {sae_id!r} not in release {release!r} (close matches: {close})")
    neuronpedia = (entry.neuronpedia_id or {}).get(sae_id)
    l0_target = (entry.expected_l0 or {}).get(sae_id)
    return SaeInfo(
        release=release, sae_id=sae_id, repo_id=entry.repo_id, folder=entry.saes_map[sae_id],
        layer=int(layer), width=width, l0=l0,
        l0_target=float(l0_target) if l0_target is not None else float("nan"),
        model_hf_id=entry.model, neuronpedia_id=neuronpedia,
    )


def _hook_point(info: SaeInfo) -> str | None:
    """hf_hook_point_in from the SAE folder's config.json, or None if there is no such file/key."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(info.repo_id, "config.json", subfolder=info.folder)
    except EntryNotFoundError:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("hf_hook_point_in")


def _validate(sae: JumpReLUSAE, info: SaeInfo, cfg: Config) -> None:
    # R5: IT model ↔ IT SAE. Compared on the repo name so a missing "google/" prefix is tolerated.
    if info.model_hf_id.split("/")[-1] != cfg.model.hf_id.split("/")[-1]:
        raise ValueError(
            f"SAE {info.release}/{info.sae_id} was trained on {info.model_hf_id!r}, "
            f"but the configured model is {cfg.model.hf_id!r} (PT SAE on an IT model is an error)"
        )
    d, w = cfg.model.d_model, WIDTHS.get(info.width)
    if w is None:
        raise ValueError(f"unknown SAE width {info.width!r} (known: {sorted(WIDTHS)})")
    expected = {"W_enc": (d, w), "b_enc": (w,), "W_dec": (w, d), "b_dec": (d,), "threshold": (w,)}
    for name, shape in expected.items():
        got = tuple(getattr(sae, name).shape)
        if got != shape:
            raise ValueError(f"{info.sae_id}: {name} has shape {got}, expected {shape}")
    if bool((sae.threshold < 0).any()):
        raise ValueError(f"{info.sae_id}: negative JumpReLU thresholds")
    hook = _hook_point(info)
    if hook is not None and hook != f"model.layers.{info.layer}.output":
        raise ValueError(f"{info.sae_id}: hook point {hook!r} != 'model.layers.{info.layer}.output'")


def load_sae(cfg: Config, layer: int, width: str, l0: str | None = None,
             device: str | None = None, backend: Literal["saelens", "raw"] = "saelens") -> JumpReLUSAE:
    """l0 defaults to cfg sae.l0. saelens: SAE.from_pretrained(...) then copy W_enc, b_enc, W_dec,
    b_dec, threshold as float32. raw: hf_hub_download(repo_id, "params.safetensors", subfolder=folder)
    with keys w_enc, w_dec, b_enc, b_dec, threshold.
    Validation (raise): info.model_hf_id == cfg.model.hf_id (R5, IT↔IT); shapes match d_model and
    width; threshold ≥ 0; if config.json exists: hf_hook_point_in == f"model.layers.{layer}.output"."""
    l0 = l0 or cfg.get("sae.l0")
    device = device or cfg.get("sae.device")
    info = lookup_sae(cfg, layer, width, l0)

    if backend == "saelens":
        from sae_lens import SAE

        loaded = SAE.from_pretrained(info.release, info.sae_id, device="cpu")
        if isinstance(loaded, tuple):  # SAELens < 6 returned (sae, cfg_dict, sparsity)
            loaded = loaded[0]
        sae = JumpReLUSAE(loaded.W_enc, loaded.b_enc, loaded.W_dec, loaded.b_dec, loaded.threshold, info)
        del loaded
    elif backend == "raw":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        params = load_file(hf_hub_download(info.repo_id, "params.safetensors", subfolder=info.folder))
        sae = JumpReLUSAE(params["w_enc"], params["b_enc"], params["w_dec"], params["b_dec"], params["threshold"], info)
        del params
    else:
        raise ValueError(f"unknown backend {backend!r}")

    _validate(sae, info, cfg)
    sae.to(device).eval()
    log.info("loaded SAE %s/%s on %s", info.release, info.sae_id, device)
    return sae


def unload(sae: JumpReLUSAE) -> None:
    for name in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold"):
        setattr(sae, name, torch.empty(0))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- encoding

@torch.inference_mode()
def encode_matrix(sae: JumpReLUSAE, X: np.ndarray, batch_size: int = 256) -> sp.csr_matrix:
    """float32 csr [N, W]; encodes in chunks on sae's device."""
    device = sae.W_enc.device
    rows, cols, vals = [], [], []
    for start in range(0, len(X), batch_size):
        f = sae.encode(torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).to(device))
        nz = f.nonzero(as_tuple=True)
        rows.append((nz[0] + start).cpu().numpy())
        cols.append(nz[1].cpu().numpy())
        vals.append(f[nz].cpu().numpy().astype(np.float32))
    shape = (len(X), sae.width)
    if not rows:
        return sp.csr_matrix(shape, dtype=np.float32)
    coo = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape, dtype=np.float32)
    return coo.tocsr()


@torch.inference_mode()
def sae_quality(sae: JumpReLUSAE, X: np.ndarray, batch_size: int = 256) -> dict:
    """{l0_mean, l0_median, fvu = Σ||x−x̂||² / Σ||x−x̄||², cos_mean, frac_features_active}.
    Interpretation: fvu > 1 → error (wrong hook/model/PT-vs-IT); fvu > 0.5 or l0_mean outside
    [0.25, 4]·l0_target → warning."""
    device = sae.W_enc.device
    mean = torch.from_numpy(X.astype(np.float32).mean(0)).to(device)
    l0s, coss = [], []
    sse = sst = 0.0
    active = torch.zeros(sae.width, dtype=torch.bool, device=device)
    for start in range(0, len(X), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).to(device).float()
        f = sae.encode(x)
        x_hat = sae.decode(f)
        l0s.append((f > 0).sum(-1).float().cpu())
        coss.append(torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).cpu())
        sse += float(((x - x_hat) ** 2).sum())
        sst += float(((x - mean) ** 2).sum())
        active |= (f > 0).any(0)
    l0 = torch.cat(l0s)
    return {
        "l0_mean": float(l0.mean()),
        "l0_median": float(l0.median()),
        "fvu": sse / max(sst, 1e-12),
        "cos_mean": float(torch.cat(coss).mean()),
        "frac_features_active": float(active.float().mean()),
    }


def quality_status(q: dict, l0_target: float) -> str:
    """"error" / "warning" / "ok" by the thresholds documented in sae_quality."""
    if q["fvu"] > 1:
        return "error"
    off_target = np.isfinite(l0_target) and not (0.25 * l0_target <= q["l0_mean"] <= 4 * l0_target)
    return "warning" if q["fvu"] > 0.5 or off_target else "ok"


def encode_dataset(cfg: Config, behavior: str, widths: Sequence[str] | None = None,
                   pool: str = "main", overwrite: bool = False) -> pd.DataFrame:
    """For each width (default cfg sae.widths) × layer in layers_sae: load SAE once, encode every
    fixed position in cfg eval.positions (skip pooled), save codes_dir/layer_{LL}_{position}.npz
    (scipy.sparse.save_npz), unload. Returns a quality table (width, layer, position, metrics on
    TRAIN rows), also saved as codes_dir/quality.csv."""
    widths = list(widths) if widths is not None else list(cfg.get("sae.widths"))
    positions = [p for p in cfg.get("eval.positions") if p in POSITIONS]
    l0 = cfg.get("sae.l0")
    p = paths(cfg, behavior, pool)
    tables = []

    for width in widths:
        codes_dir: Path = p.codes_dir(width, l0)
        codes_dir.mkdir(parents=True, exist_ok=True)
        quality_path = codes_dir / "quality.csv"
        old = pd.read_csv(quality_path, dtype={"width": str}) if quality_path.exists() else None
        rows = []
        for layer in cfg.model.layers_sae:
            files = {pos: codes_dir / f"layer_{layer_tag(layer)}_{pos}.npz" for pos in positions}
            if not overwrite and old is not None and all(f.exists() for f in files.values()) \
                    and set(positions) <= set(old.loc[old["layer"] == layer, "position"]):
                rows.extend(old[(old["layer"] == layer) & old["position"].isin(positions)].to_dict("records"))
                log.info("codes for %s layer %d (%s) exist, skipping", width, layer, pool)
                continue
            sae = load_sae(cfg, layer, width, l0)
            try:
                for pos, file in files.items():
                    X, index = load_acts(cfg, behavior, layer, pos, pool)
                    codes = encode_matrix(sae, X, cfg.get("sae.encode_batch_size"))
                    tmp = file.with_name(file.stem + ".tmp.npz")
                    sp.save_npz(tmp, codes)
                    tmp.replace(file)
                    train = (index["split"] == "train").to_numpy()
                    q = sae_quality(sae, X[train] if train.any() else X)
                    status = quality_status(q, sae.info.l0_target)
                    if status != "ok":
                        log.log(40 if status == "error" else 30, "SAE quality %s at %s layer %d %s: %s",
                                status.upper(), width, layer, pos, q)
                    rows.append({"width": width, "layer": layer, "position": pos, **q,
                                 "l0_target": sae.info.l0_target, "status": status})
            finally:
                unload(sae)
                del sae
        table = pd.DataFrame(rows)
        table.to_csv(quality_path, index=False)
        save_run_info(cfg, codes_dir, "encode_sae")
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def load_codes(cfg: Config, behavior: str, layer: int, position: str, width: str,
               l0: str | None = None, pool: str = "main") -> sp.csr_matrix:
    codes_dir = paths(cfg, behavior, pool).codes_dir(width, l0)
    return sp.load_npz(codes_dir / f"layer_{layer_tag(layer)}_{position}.npz").tocsr().astype(np.float32)


def feature_max_act(codes: sp.csr_matrix, rows: np.ndarray | None = None) -> np.ndarray:
    """Column-wise max over given rows (train rows for steering reference) → float32 [W]."""
    sub = codes if rows is None else codes[rows]
    return np.asarray(sub.max(axis=0).todense(), dtype=np.float32).ravel()


# --------------------------------------------------------------------------- Neuronpedia / interpretation

def neuronpedia_url(info: SaeInfo, feature: int) -> str | None:
    """f"https://www.neuronpedia.org/{info.neuronpedia_id}/{feature}" or None."""
    if not info.neuronpedia_id:
        return None
    return f"https://www.neuronpedia.org/{info.neuronpedia_id}/{int(feature)}"


def _snippet(act: dict, window: int = 12) -> str | None:
    tokens, values = act.get("tokens"), act.get("values")
    if not isinstance(tokens, list) or not tokens:
        return None
    peak = int(np.argmax(values)) if isinstance(values, list) and len(values) == len(tokens) else len(tokens) // 2
    lo, hi = max(0, peak - window), min(len(tokens), peak + window + 1)
    return "".join(str(t) for t in tokens[lo:peak]) + "[[" + str(tokens[peak]) + "]]" + "".join(str(t) for t in tokens[peak + 1:hi])


def fetch_neuronpedia(info: SaeInfo, feature: int, cache: JsonlCache | None = None) -> dict | None:
    """GET https://www.neuronpedia.org/api/feature/{modelId}/{source}/{feature}
    (neuronpedia_id = modelId/source); 1 s between requests, 3 retries. Returns
    {"explanations": [str], "top_snippets": [str]} parsed defensively, or None on failure."""
    import requests

    global _last_request
    if not info.neuronpedia_id:
        return None
    key = f"{info.neuronpedia_id}/{int(feature)}"
    if cache is not None and key in cache:
        return cache.get(key)

    data = None
    for attempt in range(3):
        wait = 1.0 - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            resp = requests.get(f"{_NEURONPEDIA_API}/{key}", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                break
            log.warning("Neuronpedia %s → HTTP %d (attempt %d/3)", key, resp.status_code, attempt + 1)
        except (requests.RequestException, ValueError) as e:
            log.warning("Neuronpedia %s failed (attempt %d/3): %s", key, attempt + 1, e)
    if not isinstance(data, dict):
        return None

    explanations = [
        e["description"] for e in data.get("explanations") or []
        if isinstance(e, dict) and isinstance(e.get("description"), str)
    ]
    snippets = [s for s in (_snippet(a) for a in (data.get("activations") or [])[:10] if isinstance(a, dict)) if s]
    result = {"explanations": explanations, "top_snippets": snippets}
    if cache is not None:
        cache.put(key, result)
    return result


def describe_features(cfg: Config, info: SaeInfo, features: Sequence[int],
                      codes: sp.csr_matrix, df: pd.DataFrame, n_examples: int = 10) -> pd.DataFrame:
    """One row per feature: url, explanations, top snippets, firing rate by label,
    top-n dataset prompts by activation (text[:200], label, split). Used for the week-9 interpretation."""
    if codes.shape[0] != len(df):
        raise ValueError(f"codes rows ({codes.shape[0]}) != dataset rows ({len(df)})")
    cache = JsonlCache(Path(cfg.artifact_root) / "neuronpedia" / "cache.jsonl")
    y = df["label"].to_numpy()
    csc = codes.tocsc()
    rows = []
    for feature in features:
        col = np.asarray(csc[:, int(feature)].todense()).ravel()
        fires = col > 0
        top = [i for i in np.argsort(-col, kind="stable")[:n_examples] if col[i] > 0]
        np_data = fetch_neuronpedia(info, int(feature), cache) or {}
        rows.append({
            "feature": int(feature),
            "url": neuronpedia_url(info, int(feature)),
            "explanations": np_data.get("explanations", []),
            "top_snippets": np_data.get("top_snippets", []),
            "firing_rate_pos": float(fires[y == 1].mean()) if (y == 1).any() else float("nan"),
            "firing_rate_neg": float(fires[y == 0].mean()) if (y == 0).any() else float("nan"),
            "max_act": float(col.max()) if len(col) else 0.0,
            "top_examples": [
                {"text": str(df["text"].iloc[i])[:200], "label": int(y[i]), "split": str(df["split"].iloc[i]),
                 "activation": float(col[i])}
                for i in top
            ],
        })
    return pd.DataFrame(rows)
