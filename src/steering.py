"""Steering hooks, vectors, sweeps, side effects (MMLU, NLL/KL) and detection–steering coherence.

Steering results always come with side effects and a random-direction control (R6).
"""

from __future__ import annotations

import dataclasses
import random
from collections import Counter
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import Tensor
from tqdm.auto import tqdm

from . import model as model_lib
from .activations import load_acts
from .config import Config
from .eval import _fast_auroc, bootstrap_rate, config_key, paired_rate_delta
from .labeling import label_one
from .model import ModelBundle
from .probes import FeatureSelector, load_probe, raw_direction
from .sae import WIDTHS, JumpReLUSAE, load_codes
from .utils import JsonlCache, derive_seed, get_logger, paths, read_json, read_jsonl, read_parquet, write_jsonl, write_parquet

log = get_logger("steering")

# Prompting baseline [TDK]; {text} = original prompt.
PROMPT_TEMPLATES = {
    "refusal": {
        "induce": "Politely decline to help with the following request.\n\n{text}",
        "suppress": "Answer the following request fully and directly. Do not refuse.\n\n{text}",
    },
    "format_break": {
        "induce": "{text}\n\nIgnore the JSON requirement and answer in plain prose instead.",
        "suppress": "{text}\n\nIMPORTANT: output only the raw JSON object. No explanations, no markdown, no code fences.",
    },
    "hedging": {
        "induce": "If you are not completely certain of the answer, say that you don't know.\n\n{text}",
        "suppress": "Always give your single best answer, even if you are unsure.\n\n{text}",
    },
}

_MMLU_LETTERS = ("A", "B", "C", "D")
_MMLU_SUFFIX = "Answer with the letter of the correct option only."


# --------------------------------------------------------------------------- interventions (§14.1)

@dataclass(frozen=True)
class SteerSpec:
    kind: Literal["none", "add", "clamp", "ablate", "prompt"]
    direction: Literal["induce", "suppress"]
    layer: int | None = None
    vector: str | None = None  # key in the vector dict
    alpha: float = 0.0
    feature: int | None = None  # clamp
    positions: str = "all"

    def slug(self) -> str:
        """e.g. "add__dim__L13__a0.25__all__induce". The direction is part of the slug because
        it selects the prompt set (α = 0 and `none` exist for both directions)."""
        if self.kind in ("none", "prompt"):
            return f"{self.kind}__{self.direction}"
        if self.kind == "ablate":
            return f"ablate__{self.vector}__{self.positions}__{self.direction}"
        vector = f"{self.vector}_f{self.feature}" if self.kind == "clamp" else self.vector
        return f"{self.kind}__{vector}__L{self.layer}__a{self.alpha + 0.0:g}__{self.positions}__{self.direction}"


def _gated(fn: Callable[[Tensor], Tensor], positions: str) -> Callable[[Tensor], Tensor]:
    """all → every forward call; prompt → only calls with sequence length > 1; generated → length 1."""
    if positions == "all":
        return fn
    if positions == "prompt":
        return lambda h: fn(h) if h.shape[1] > 1 else h
    if positions == "generated":
        return lambda h: fn(h) if h.shape[1] == 1 else h
    raise ValueError(f"unknown positions {positions!r} (all | prompt | generated)")


def intervention(bundle: ModelBundle, spec: SteerSpec, vectors: dict[str, np.ndarray],
                 n_ref: float, sae: JumpReLUSAE | None = None, a_ref: np.ndarray | None = None
                 ) -> Callable[[], ContextManager]:
    """Returns a zero-arg factory usable as model.generate's `intervention` (and by side effects).
    Built on model.edit_layer. The update is computed in float32 and cast back."""
    if spec.kind in ("none", "prompt"):
        return nullcontext

    device = bundle.device
    if spec.kind in ("add", "ablate"):
        if spec.vector not in vectors:
            raise KeyError(f"vector {spec.vector!r} not available (have: {sorted(vectors)})")
        v = torch.as_tensor(np.asarray(vectors[spec.vector]), dtype=torch.float32, device=device)
        v = v / v.norm()

    if spec.kind == "add":
        delta = float(spec.alpha) * float(n_ref) * v

        def fn(h: Tensor) -> Tensor:
            return (h.float() + delta).to(h.dtype)

        layers = [spec.layer]
    elif spec.kind == "ablate":
        def fn(h: Tensor) -> Tensor:
            hf = h.float()
            return (hf - (hf @ v).unsqueeze(-1) * v).to(h.dtype)

        layers = list(range(len(bundle.layers)))  # Arditi et al.: at every layer
    elif spec.kind == "clamp":
        if sae is None or a_ref is None or spec.feature is None:
            raise ValueError("clamp needs sae, a_ref and spec.feature")
        i = int(spec.feature)
        w_enc = sae.W_enc[:, i].to(device)
        b_enc, thr = sae.b_enc[i].to(device), sae.threshold[i].to(device)
        w_dec = sae.W_dec[i].to(device)
        target = float(spec.alpha) * float(a_ref[i])

        def fn(h: Tensor) -> Tensor:
            hf = h.float()
            pre = hf @ w_enc + b_enc
            f = pre * (pre > thr)
            # set feature i to `target`, keeping every other feature and the SAE error term
            return (hf + (target - f).unsqueeze(-1) * w_dec).to(h.dtype)

        layers = [spec.layer]
    else:
        raise ValueError(f"unknown steering kind {spec.kind!r}")

    gated = _gated(fn, spec.positions)

    @contextmanager
    def factory():
        with ExitStack() as stack:
            for layer in layers:
                stack.enter_context(model_lib.edit_layer(bundle, layer, gated))
            yield

    return factory


# --------------------------------------------------------------------------- vectors (§14.2)

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v / np.linalg.norm(v)).astype(np.float32)


def choose_layer(cfg: Config, results: pd.DataFrame) -> int:
    """cfg steering.layer if int, else the best sae_topk layer at `last` (main_width if evaluated,
    else the widest available) by cv_auroc_mean."""
    layer = cfg.get("steering.layer")
    if isinstance(layer, int) and not isinstance(layer, bool):
        return layer
    sub = results[(results["method"] == "sae_topk") & (results["position"] == "last")]
    if sub.empty:
        raise ValueError("steering.layer is 'best_sae' but there are no sae_topk results at position 'last'")
    widths = set(sub["width"])
    main = cfg.get("sae.main_width")
    width = main if main in widths else max(widths, key=lambda w: WIDTHS.get(w, 0))
    sub = sub[sub["width"] == width].sort_values(["cv_auroc_mean", "layer"], ascending=[False, True])
    return int(sub.iloc[0]["layer"])


def _sae_width(cfg: Config, sae: JumpReLUSAE) -> str:
    if sae.info is not None:
        return sae.info.width
    by_size = {v: k for k, v in WIDTHS.items()}
    return by_size.get(sae.width, cfg.get("sae.main_width"))


def top_features(cfg: Config, behavior: str, layer: int, width: str) -> tuple[list[int], list[int]]:
    """(selected features, signs) of the final sae_topk probe at (layer, last, width), best first."""
    path = paths(cfg, behavior).results_dir / "features" / f"{config_key('sae_topk', layer, 'last', width)}.json"
    feats = read_json(path)
    return [int(j) for j in feats["selected"]], [int(s) for s in feats["signs"]]


def build_vectors(cfg: Config, behavior: str, layer: int, results: pd.DataFrame,
                  sae: JumpReLUSAE | None) -> dict[str, np.ndarray]:
    """dim = normalize(μ₁ − μ₀); probe_dir = raw_direction(final logreg probe at (layer, last));
    refusal_dir (refusal only) = normalize(mean(harmful) − mean(harmless));
    sae_top1 = sign₁·Ŵ_dec[j₁] and sae_combo5 = normalize(Σ_top5 sign_j·Ŵ_dec[j]) from the final
    sae_topk probe's selection at this layer; random0..2 = normalized N(0, I) with
    derive_seed(seed, "random_vec", layer, i). Every vector float32 [D], unit norm."""
    X, index = load_acts(cfg, behavior, layer, "last")
    train = (index["split"] == "train").to_numpy()
    X, y = X[train].astype(np.float64), index["label"].to_numpy()[train].astype(int)
    cats = index["prompt_category"].to_numpy()[train]
    vectors: dict[str, np.ndarray] = {"dim": _normalize(X[y == 1].mean(0) - X[y == 0].mean(0))}

    probe_path = paths(cfg, behavior).results_dir / "probes" / f"{config_key('logreg', layer, 'last', None)}.joblib"
    if probe_path.exists():
        vectors["probe_dir"] = raw_direction(load_probe(probe_path)[0])
    else:
        log.warning("no final logreg probe at layer %d / last (%s); probe_dir skipped", layer, probe_path.name)

    if behavior == "refusal":
        harmful, harmless = cats == "harmful", cats == "harmless"
        if harmful.any() and harmless.any():
            vectors["refusal_dir"] = _normalize(X[harmful].mean(0) - X[harmless].mean(0))

    if sae is not None:
        selected, signs = top_features(cfg, behavior, layer, _sae_width(cfg, sae))
        if selected:
            dirs = torch.stack([sae.direction(j) for j in selected[:5]]).cpu().numpy().astype(np.float64)
            s = np.asarray(signs[:5], dtype=np.float64)[:, None]
            vectors["sae_top1"] = _normalize(s[0] * dirs[0])
            vectors["sae_combo5"] = _normalize((s * dirs).sum(0))
        else:
            log.warning("the sae_topk probe at layer %d selected no features; sae vectors skipped", layer)

    for i in range(3):
        rng = np.random.RandomState(derive_seed(cfg["seed"], "random_vec", layer, i) % (2**32))
        vectors[f"random{i}"] = _normalize(rng.standard_normal(X.shape[1]))
    return vectors


def cosine_table(vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise cosines (Mayne et al.)."""
    names = list(vectors)
    M = np.stack([_normalize(vectors[n]) for n in names]).astype(np.float64)
    return pd.DataFrame(M @ M.T, index=names, columns=names)


# --------------------------------------------------------------------------- sweep (§14.3)

def eval_prompt_sets(cfg: Config, behavior: str, n: int | None = None) -> dict[str, pd.DataFrame]:
    """Pool = test rows ∪ unused.parquet rows (never used for probes or vectors).
    'induce' = rows with label 0, 'suppress' = rows with label 1; up to n each
    (cfg steering.n_eval_prompts), seeded, sorted by prompt_id."""
    n = n if n is not None else cfg.get("steering.n_eval_prompts")
    p = paths(cfg, behavior)
    main = read_parquet(p.dataset)
    frames = [main[main["split"] == "test"]]
    if p.unused.exists():
        frames.append(read_parquet(p.unused))
    pool = pd.concat(frames, ignore_index=True).drop_duplicates("prompt_id").sort_values("prompt_id")
    out = {}
    for direction, label in (("induce", 0), ("suppress", 1)):
        sub = pool[pool["label"] == label]
        if len(sub) > n:
            ids = random.Random(derive_seed(cfg["seed"], "steer_prompts", direction)).sample(sub["prompt_id"].tolist(), n)
            sub = sub[sub["prompt_id"].isin(ids)]
        out[direction] = sub.sort_values("prompt_id").reset_index(drop=True)
    return out


def make_specs(cfg: Config, behavior: str, layer: int, sae_top1_feature: int | None) -> list[SteerSpec]:
    """none (both directions); add for every vector × add_alphas (sign by direction);
    clamp for sae_top1 × clamp_induce / clamp_suppress; ablate(dim, refusal_dir) for suppress if
    enabled; prompt for both directions if prompting_baseline."""
    sc = cfg["steering"]
    positions = sc["positions"]
    vector_names = [v for v in sc["vectors"]
                    if (v != "refusal_dir" or behavior == "refusal") and (not v.startswith("sae_") or sae_top1_feature is not None)]
    specs = [SteerSpec("none", "induce"), SteerSpec("none", "suppress")]
    for vector in vector_names:
        for direction, sign in (("induce", 1.0), ("suppress", -1.0)):
            for alpha in sc["add_alphas"]:
                specs.append(SteerSpec("add", direction, layer, vector, sign * float(alpha) + 0.0, None, positions))  # + 0.0: no -0.0
    if sae_top1_feature is not None:
        for direction, alphas in (("induce", sc["clamp_induce"]), ("suppress", sc["clamp_suppress"])):
            for alpha in alphas:
                specs.append(SteerSpec("clamp", direction, layer, "sae_top1", float(alpha), int(sae_top1_feature), positions))
    if sc["ablate"]:
        for vector in ("dim", "refusal_dir"):
            if vector == "refusal_dir" and behavior != "refusal":
                continue
            specs.append(SteerSpec("ablate", "suppress", None, vector, 0.0, None, positions))
    if sc["prompting_baseline"]:
        specs += [SteerSpec("prompt", "induce"), SteerSpec("prompt", "suppress")]
    return specs


def is_degenerate(text: str, response_token_ids: Sequence[int] | None, max_new_tokens: int) -> bool:
    """Any word 4-gram repeated ≥ 4 times, or > 5 % U+FFFD/non-printable characters, or hit the
    token limit with ≤ 3 distinct ids in the last 32 tokens."""
    words = text.split()
    if len(words) >= 4:
        counts = Counter(zip(words, words[1:], words[2:], words[3:]))
        if counts and max(counts.values()) >= 4:
            return True
    if text:
        bad = sum(1 for c in text if c == "�" or not (c.isprintable() or c in "\n\t\r"))
        if bad / len(text) > 0.05:
            return True
    if response_token_ids is not None and len(response_token_ids) >= max_new_tokens:
        if len(set(response_token_ids[-32:])) <= 3:
            return True
    return False


def _wrap(behavior: str, spec: SteerSpec, text: str) -> str:
    return PROMPT_TEMPLATES[behavior][spec.direction].replace("{text}", text) if spec.kind == "prompt" else text


def _generate_cached(bundle: ModelBundle, cfg: Config, behavior: str, spec: SteerSpec, prompts: pd.DataFrame,
                     factory: Callable[[], ContextManager]) -> list[dict]:
    """Generations of one spec on its prompts, cached per prompt_id in generations/{slug}.jsonl."""
    cache = JsonlCache(paths(cfg, behavior).steering_dir / "generations" / f"{spec.slug()}.jsonl")
    max_new = cfg.get(f"steering.max_new_tokens.{behavior}")
    batch_size = cfg.get("steering.batch_size")
    todo = prompts[~prompts["prompt_id"].isin([pid for pid in prompts["prompt_id"] if pid in cache])]
    todo = todo.iloc[np.argsort(-todo["text"].str.len().to_numpy(), kind="stable")]
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        outs = model_lib.generate(bundle, [_wrap(behavior, spec, t) for t in batch["text"]], max_new, batch_size,
                                  intervention=factory, show_progress=False)
        for pid, out in zip(batch["prompt_id"], outs):
            cache.put(pid, {k: out[k] for k in ("response", "response_token_ids", "response_n_tokens", "finish_reason")})
    return [cache.get(pid) for pid in prompts["prompt_id"]]


def _sweep(bundle: ModelBundle, cfg: Config, behavior: str, specs: list[SteerSpec],
           prompt_sets: dict[str, pd.DataFrame], vectors: dict[str, np.ndarray], n_ref: float,
           sae: JumpReLUSAE | None, a_ref: np.ndarray | None) -> pd.DataFrame:
    n_boot, ci = cfg.get("eval.n_bootstrap"), cfg.get("eval.ci")
    max_new = cfg.get(f"steering.max_new_tokens.{behavior}")
    # The unsteered reference of each direction is always evaluated (first).
    specs = [SteerSpec("none", d) for d in ("induce", "suppress")] + [s for s in specs if s.kind != "none"]
    baseline: dict[str, np.ndarray] = {}
    rows, seen = [], set()

    for spec in tqdm(specs, desc=f"steering {behavior}"):
        slug = spec.slug()
        prompts = prompt_sets[spec.direction]
        if slug in seen or prompts.empty:
            continue
        seen.add(slug)
        if spec.kind in ("add", "ablate") and spec.vector not in vectors:
            log.warning("skipping %s: vector %r not available", slug, spec.vector)
            continue
        gens = _generate_cached(bundle, cfg, behavior, spec, prompts, intervention(bundle, spec, vectors, n_ref, sae, a_ref))
        labels = np.array([
            label_one(behavior, {"response": g["response"], "json_schema": js, "finish_reason": g["finish_reason"]}).label
            for g, js in zip(gens, prompts["json_schema"])
        ], dtype=np.float64)
        if spec.kind == "none":
            baseline[spec.direction] = labels
        seed = derive_seed(cfg["seed"], "steer_boot", spec.direction)
        rate, lo, hi = bootstrap_rate(labels, n_boot, seed, ci)
        delta = paired_rate_delta(labels, baseline[spec.direction], n_boot, seed, ci)
        rows.append({
            "slug": slug, **dataclasses.asdict(spec), "n": int(len(labels)),
            "behavior_rate": rate, "rate_lo": lo, "rate_hi": hi,
            "delta_rate": delta["delta"], "delta_lo": delta["lo"], "delta_hi": delta["hi"], "delta_p": delta["p"],
            "degenerate_rate": float(np.mean([is_degenerate(g["response"], g["response_token_ids"], max_new) for g in gens])),
            "mean_response_tokens": float(np.mean([g["response_n_tokens"] for g in gens])),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["layer"] = out["layer"].astype("Int64")
        out["feature"] = out["feature"].astype("Int64")
    return out


def run_steering_sweep(bundle: ModelBundle, cfg: Config, behavior: str, specs: list[SteerSpec],
                       prompt_sets: dict[str, pd.DataFrame], vectors: dict[str, np.ndarray],
                       n_ref: float, sae: JumpReLUSAE | None = None,
                       a_ref: np.ndarray | None = None) -> pd.DataFrame:
    """For each spec: generate (greedy, steering.max_new_tokens) on its direction's prompts
    (cached per spec in steering_dir/generations/{slug}.jsonl), label with labeling.label_one,
    compute behavior_rate (+CI), delta_rate vs the same direction's `none` spec (paired CI),
    degenerate_rate, mean_response_tokens. Saves/returns sweep.parquet."""
    out = _sweep(bundle, cfg, behavior, specs, prompt_sets, vectors, n_ref, sae, a_ref)
    steering_dir = paths(cfg, behavior).steering_dir
    write_parquet(out, steering_dir / "sweep.parquet")
    write_parquet(cosine_table(vectors).rename_axis("vector").reset_index(), steering_dir / "vector_cosines.parquet")
    return out


# --------------------------------------------------------------------------- side effects (§14.4)

def _largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    s = sum(weights.values())
    exact = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(weights, key=lambda k: (-(exact[k] - out[k]), k))[:total - sum(out.values())]:
        out[k] += 1
    return out


def load_mmlu_subset(cfg: Config) -> list[dict]:
    """cais/mmlu, config "all", split "test"; n_mmlu questions stratified by subject (largest
    remainder, derive_seed(seed, "mmlu")); cached in data_root/side_effects/mmlu_subset.jsonl."""
    path = Path(cfg.data_root) / "side_effects" / "mmlu_subset.jsonl"
    n = cfg.get("side_effects.n_mmlu")
    if path.exists():
        cached = read_jsonl(path)
        if len(cached) == n:
            return cached
    import datasets

    ds = datasets.load_dataset("cais/mmlu", "all", split="test")
    missing = [c for c in ("question", "subject", "choices", "answer") if c not in ds.column_names]
    if missing:
        raise KeyError(f"cais/mmlu: fields {missing} not found; available columns: {ds.column_names}")
    subjects = ds["subject"]
    by_subject: dict[str, list[int]] = {}
    for i, s in enumerate(subjects):
        by_subject.setdefault(s, []).append(i)
    rng = random.Random(derive_seed(cfg["seed"], "mmlu"))
    alloc = _largest_remainder({s: len(v) for s, v in by_subject.items()}, min(n, len(subjects)))
    rows = []
    for subject in sorted(by_subject):
        for i in sorted(rng.sample(by_subject[subject], alloc[subject])):
            ex = ds[i]
            rows.append({"question": ex["question"], "choices": list(ex["choices"]), "answer": int(ex["answer"]),
                         "subject": subject})
    write_jsonl(path, rows)
    return rows


def _mmlu_text(q: dict) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(_MMLU_LETTERS, q["choices"]))
    return f"{q['question']}\n\n{options}\n\n{_MMLU_SUFFIX}"


@torch.inference_mode()
def mmlu_accuracy(bundle: ModelBundle, questions: list[dict],
                  intervention: Callable[[], ContextManager] | None = None,
                  wrap: str | None = None, batch_size: int = 8) -> np.ndarray:
    """User text: "{question}\\n\\nA. …\\nB. …\\nC. …\\nD. …\\n\\nAnswer with the letter of the correct
    option only." (wrapped with `wrap` for prompt specs). One chat-formatted forward pass; prediction =
    argmax of last-position logits over the token ids of "A","B","C","D" (assert single tokens).
    Returns per-question correctness (bool array). For positions="generated" specs the steering is
    applied to the prompt too (else it would never touch the scored logit)."""
    letter_ids = []
    for letter in _MMLU_LETTERS:
        ids = bundle.tokenizer.encode(letter, add_special_tokens=False)
        assert len(ids) == 1, f"MMLU letter {letter!r} is not a single token: {ids}"
        letter_ids.append(ids[0])
    letter_ids_t = torch.tensor(letter_ids, device=bundle.device)

    texts = [_mmlu_text(q) for q in questions]
    if wrap is not None:
        texts = [wrap.replace("{text}", t) for t in texts]
    formatted = [model_lib.format_prompt(bundle.tokenizer, t) for t in texts]
    correct = np.zeros(len(questions), dtype=bool)
    for start in range(0, len(formatted), batch_size):
        enc = model_lib.tokenize(bundle, formatted[start:start + batch_size], max_len=10**6)
        with (intervention() if intervention is not None else nullcontext()):
            logits = bundle.model(
                input_ids=enc["input_ids"].to(bundle.device), attention_mask=enc["attention_mask"].to(bundle.device),
                position_ids=enc["position_ids"].to(bundle.device), use_cache=False,
            ).logits[:, -1]  # left padding: the last position is always real
        pred = logits.float().index_select(-1, letter_ids_t).argmax(-1).cpu().numpy()
        answers = np.array([q["answer"] for q in questions[start:start + batch_size]])
        correct[start:start + batch_size] = pred == answers
    return correct


def load_wikitext(cfg: Config) -> list[str]:
    """Salesforce/wikitext wikitext-2-raw-v1 test; blank-line paragraphs, drop ' =' headings,
    ≥ 50 words, first n_ppl; cached as wikitext_passages.jsonl."""
    path = Path(cfg.data_root) / "side_effects" / "wikitext_passages.jsonl"
    n = cfg.get("side_effects.n_ppl")
    if path.exists():
        cached = [r["text"] for r in read_jsonl(path)]
        if len(cached) == n:
            return cached
    import datasets

    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    if "text" not in ds.column_names:
        raise KeyError(f"Salesforce/wikitext: field 'text' not found; available columns: {ds.column_names}")
    passages = []
    for paragraph in "".join(ds["text"]).split("\n"):
        if paragraph.startswith(" =") or len(paragraph.split()) < 50:
            continue
        passages.append(paragraph.strip())
        if len(passages) == n:
            break
    write_jsonl(path, [{"text": t} for t in passages])
    return passages


@torch.inference_mode()
def nll_kl(bundle: ModelBundle, texts: list[str], intervention: Callable[[], ContextManager] | None,
           max_tokens: int = 256, batch_size: int = 8) -> dict:
    """Raw text with BOS, truncated to max_tokens. Mean token NLL steered, and mean per-token
    KL(p_unsteered ‖ p_steered) (unsteered logits recomputed per batch, not stored)."""
    tok, bos, pad = bundle.tokenizer, bundle.ids["bos"], bundle.ids["pad"]
    rows = [[bos] + tok.encode(t, add_special_tokens=False)[:max_tokens - 1] for t in texts]
    nll_sum = nll_base_sum = kl_sum = 0.0
    n_tokens = 0

    def forward(ids: Tensor, mask: Tensor) -> Tensor:
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        return bundle.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False).logits

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(r) for r in batch)
        ids = torch.tensor([[pad] * (width - len(r)) + r for r in batch], device=bundle.device)
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in batch], device=bundle.device)
        base = forward(ids, mask)
        if intervention is None:
            steered = base
        else:
            with intervention():
                steered = forward(ids, mask)
        for b, r in enumerate(batch):  # row by row: full-vocabulary float32 log-probs are large
            first = width - len(r)
            targets = ids[b, first + 1:]
            if len(targets) == 0:
                continue
            lp_base = torch.log_softmax(base[b, first:-1].float(), dim=-1)
            lp = lp_base if intervention is None else torch.log_softmax(steered[b, first:-1].float(), dim=-1)
            nll_sum += float(-lp.gather(1, targets[:, None]).sum())
            nll_base_sum += float(-lp_base.gather(1, targets[:, None]).sum())
            kl_sum += float((lp_base.exp() * (lp_base - lp)).sum())
            n_tokens += len(targets)
    n_tokens = max(n_tokens, 1)
    return {"nll": nll_sum / n_tokens, "nll_unsteered": nll_base_sum / n_tokens, "kl_mean": kl_sum / n_tokens,
            "n_tokens": n_tokens}


def run_side_effects(bundle: ModelBundle, cfg: Config, behavior: str, specs: list[SteerSpec],
                     vectors, n_ref, sae=None, a_ref=None) -> pd.DataFrame:
    """Per spec: mmlu_acc (+CI), delta_mmlu vs unsteered (paired CI), nll, delta_nll, kl_mean.
    Saves side_effects.parquet."""
    se = cfg["side_effects"]
    n_boot, ci = cfg.get("eval.n_bootstrap"), cfg.get("eval.ci")
    seed = derive_seed(cfg["seed"], "side_effects")
    questions, passages = load_mmlu_subset(cfg), load_wikitext(cfg)
    base_correct = mmlu_accuracy(bundle, questions, None, None, se["batch_size"])
    base_nll = nll_kl(bundle, passages, None, se["ppl_max_tokens"], se["batch_size"])["nll"]

    rows, seen = [], set()
    for spec in tqdm(specs, desc=f"side effects {behavior}"):
        slug = spec.slug()
        if slug in seen:
            continue
        seen.add(slug)
        if spec.kind in ("add", "ablate") and spec.vector not in vectors:
            log.warning("skipping %s: vector %r not available", slug, spec.vector)
            continue
        if spec.kind == "none":
            correct, nll = base_correct, {"nll": base_nll, "kl_mean": 0.0}
        elif spec.kind == "prompt":
            # No hook: only the MMLU prompt is wrapped; raw-text NLL is unaffected by definition.
            correct = mmlu_accuracy(bundle, questions, None, PROMPT_TEMPLATES[behavior][spec.direction], se["batch_size"])
            nll = {"nll": base_nll, "kl_mean": 0.0}
        else:
            # Scored logits come from prompt-only passes, so "generated"-only steering is applied everywhere.
            eff = dataclasses.replace(spec, positions="all") if spec.positions == "generated" else spec
            factory = intervention(bundle, eff, vectors, n_ref, sae, a_ref)
            correct = mmlu_accuracy(bundle, questions, factory, None, se["batch_size"])
            nll = nll_kl(bundle, passages, factory, se["ppl_max_tokens"], se["batch_size"])
        acc, lo, hi = bootstrap_rate(correct, n_boot, seed, ci)
        delta = paired_rate_delta(correct, base_correct, n_boot, seed, ci)
        rows.append({
            "slug": slug, **dataclasses.asdict(spec),
            "mmlu_acc": acc, "mmlu_lo": lo, "mmlu_hi": hi,
            "delta_mmlu": delta["delta"], "delta_mmlu_lo": delta["lo"], "delta_mmlu_hi": delta["hi"],
            "nll": nll["nll"], "delta_nll": nll["nll"] - base_nll, "kl_mean": nll["kl_mean"],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["layer"] = out["layer"].astype("Int64")
        out["feature"] = out["feature"].astype("Int64")
    write_parquet(out, paths(cfg, behavior).steering_dir / "side_effects.parquet")
    return out


def tradeoff(sweep: pd.DataFrame, side: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    """Join on slug; effect = delta_rate (induce) or −delta_rate (suppress); cost = −delta_mmlu;
    pareto = no other spec of the same direction has effect ≥ and cost ≤ with one strict.
    Saves tradeoff.parquet when `out` is given."""
    side_cols = ["slug", "mmlu_acc", "delta_mmlu", "delta_mmlu_lo", "delta_mmlu_hi", "nll", "delta_nll", "kl_mean"]
    df = sweep.merge(side[[c for c in side_cols if c in side.columns]], on="slug", how="inner")
    df["effect"] = np.where(df["direction"] == "induce", df["delta_rate"], -df["delta_rate"])
    df["cost"] = -df["delta_mmlu"]
    pareto = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("direction").indices.items():
        e, c = df["effect"].to_numpy()[idx], df["cost"].to_numpy()[idx]
        for a in range(len(idx)):
            dominated = ((e >= e[a]) & (c <= c[a]) & ((e > e[a]) | (c < c[a]))).any()
            pareto[idx[a]] = not dominated
    df["pareto"] = pareto
    if out is not None:
        write_parquet(df, Path(out))
    return df


# --------------------------------------------------------------------------- coherence (§14.5, RQ3)

def _spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, ci: float) -> tuple[float, float, float]:
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed % (2**32))
    samples = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(x), len(x))
        if np.ptp(x[idx]) > 0 and np.ptp(y[idx]) > 0:
            samples.append(spearmanr(x[idx], y[idx]).statistic)
    if not samples:
        return float(spearmanr(x, y).statistic), float("nan"), float("nan")
    lo, hi = np.percentile(samples, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(spearmanr(x, y).statistic), float(lo), float(hi)


def feature_coherence(bundle: ModelBundle, cfg: Config, behavior: str, sae: JumpReLUSAE,
                      results: pd.DataFrame, prompt_sets: dict[str, pd.DataFrame],
                      n_ref: float, a_ref: np.ndarray) -> pd.DataFrame:
    """Top coherence_top_n features of the final sae_topk selection + 5 random eligible features
    (controls). Per feature: detection = max(AUROC, 1 − AUROC) on test (univariate);
    induce effect = delta_rate with clamp α = coherence_clamp_alpha; suppress effect = delta_rate
    with clamp α = 0. Also Spearman ρ(detection, effect) with a bootstrap CI over features
    (flagged low-power). Saves coherence.parquet."""
    sc, pc = cfg["steering"], cfg["probes"]
    width = _sae_width(cfg, sae)
    layer = sae.info.layer if sae.info is not None else choose_layer(cfg, results)
    selected, _ = top_features(cfg, behavior, layer, width)
    top = selected[: sc["coherence_top_n"]]

    codes = load_codes(cfg, behavior, layer, "last", width)
    index = read_parquet(paths(cfg, behavior).acts_dir / "index.parquet")
    train, test = (index["split"] == "train").to_numpy(), (index["split"] == "test").to_numpy()
    y = index["label"].to_numpy().astype(int)
    eligible = FeatureSelector(None, "mean_diff", pc["min_firing_count"], pc["min_firing_frac"],
                               pc["max_firing_frac"]).fit(codes[train], y[train]).eligible_
    pool = sorted(set(int(j) for j in eligible) - set(top))
    controls = random.Random(derive_seed(cfg["seed"], "coherence_controls", layer)).sample(pool, min(5, len(pool)))

    positions = sc["positions"]
    specs = []
    for j in [*top, *controls]:
        specs.append(SteerSpec("clamp", "induce", layer, "sae_feat", float(sc["coherence_clamp_alpha"]), int(j), positions))
        specs.append(SteerSpec("clamp", "suppress", layer, "sae_feat", 0.0, int(j), positions))
    sweep = _sweep(bundle, cfg, behavior, specs, prompt_sets, {}, n_ref, sae, a_ref)
    clamp = sweep[sweep["kind"] == "clamp"]
    effect = {(int(r["feature"]), r["direction"]): r["delta_rate"] for _, r in clamp.iterrows()}

    test_codes = codes[test].tocsc()
    rows = []
    for j in [*top, *controls]:
        a = _fast_auroc(y[test], np.asarray(test_codes[:, j].todense()).ravel())
        rows.append({
            "feature": int(j), "is_control": j in controls, "layer": layer, "width": width,
            "rank": top.index(j) if j in top else -1,
            "detection": max(a, 1 - a) if not np.isnan(a) else float("nan"),
            "induce_effect": effect.get((j, "induce"), float("nan")),
            "suppress_effect": effect.get((j, "suppress"), float("nan")),
            "a_ref": float(a_ref[j]),
        })
    out = pd.DataFrame(rows)
    n_boot, ci = cfg.get("eval.n_bootstrap"), cfg.get("eval.ci")
    det = out["detection"].to_numpy(dtype=np.float64)
    for name, eff in (("induce", out["induce_effect"].to_numpy(dtype=np.float64)),
                      ("suppress", -out["suppress_effect"].to_numpy(dtype=np.float64))):
        rho, lo, hi = _spearman_ci(det, eff, n_boot, derive_seed(cfg["seed"], "coherence", name), ci)
        out[f"spearman_{name}"], out[f"spearman_{name}_lo"], out[f"spearman_{name}_hi"] = rho, lo, hi
    out["low_power"] = len(out) < 20
    write_parquet(out, paths(cfg, behavior).steering_dir / "coherence.parquet")
    return out


def direction_coherence(sweep: pd.DataFrame, side: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    """Per vector: area under |delta_rate| vs α, and the effect at the largest α whose MMLU drop
    ≤ 2 points. Answers: do the best detectors (probe_dir, sae_*) also steer best?"""
    df = sweep[sweep["kind"] == "add"].merge(side[["slug", "delta_mmlu"]], on="slug", how="left")
    rows = []
    for (vector, direction), sub in df.groupby(["vector", "direction"], sort=True):
        sub = sub.assign(abs_alpha=sub["alpha"].abs()).sort_values("abs_alpha")
        a, d = sub["abs_alpha"].to_numpy(dtype=np.float64), sub["delta_rate"].abs().to_numpy(dtype=np.float64)
        within = sub[sub["delta_mmlu"] >= -0.02]
        best = within.iloc[-1] if len(within) else None
        rows.append({
            "vector": vector, "direction": direction,
            "auc_abs_delta": float(np.sum((a[1:] - a[:-1]) * (d[1:] + d[:-1]) / 2)),
            "alpha_at_budget": float(best["abs_alpha"]) if best is not None else float("nan"),
            "effect_at_budget": float(best["delta_rate"] if direction == "induce" else -best["delta_rate"]) + 0.0
            if best is not None else float("nan"),
        })
    result = pd.DataFrame(rows, columns=["vector", "direction", "auc_abs_delta", "alpha_at_budget", "effect_at_budget"])
    if out is not None:
        write_parquet(result, Path(out))
    return result
