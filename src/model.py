"""Model/tokenizer loading, chat formatting, token positions, hooks and generation."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterator, Sequence

import torch
from torch import Tensor
from tqdm.auto import tqdm

from .config import Config, ModelSpec
from .utils import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

log = get_logger("model")

# Verified special ids of the Gemma 3 tokenizer (spec §5.1); load_model asserts them.
EXPECTED_IDS = {"pad": 0, "eos": 1, "bos": 2, "turn_start": 105, "turn_end": 106}

_DECODER_LAYER_PATHS = (
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "model.layers",
)


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: "PreTrainedTokenizerBase"
    layers: torch.nn.ModuleList  # decoder layers of the text model
    spec: ModelSpec
    revision: str  # resolved HF commit sha (recorded in outputs)
    ids: dict[str, int]  # pad, eos, bos, turn_start, turn_end
    stop_ids: list[int]
    device: torch.device


# --------------------------------------------------------------------------- loading

def get_decoder_layers(model: torch.nn.Module, n_layers: int) -> torch.nn.ModuleList:
    """Try, in order: model.model.language_model.layers, model.language_model.model.layers,
    model.language_model.layers, model.model.layers. Return the first ModuleList of length
    n_layers; else raise listing all ModuleLists and their lengths."""
    for path in _DECODER_LAYER_PATHS:
        node: Any = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, torch.nn.ModuleList) and len(node) == n_layers:
            return node
    found = {name: len(m) for name, m in model.named_modules() if isinstance(m, torch.nn.ModuleList)}
    raise ValueError(f"no decoder ModuleList of length {n_layers} found; ModuleLists (name: length): {found}")


def _cached_revision(hf_id: str) -> str | None:
    """Commit sha of the snapshot the model was just loaded from (the cache path is
    .../snapshots/<sha>/config.json). transformers 5 no longer keeps it on the config."""
    from pathlib import Path

    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(hf_id, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def load_model(cfg: Config) -> ModelBundle:
    """causal_lm → AutoModelForCausalLM; image_text_to_text → Gemma3ForConditionalGeneration.
    torch_dtype from cfg, attn_implementation from cfg, eval(), tokenizer.padding_side='left'.
    Asserts: resolved layer count == n_layers, hidden size == d_model, special ids as in §5.1."""
    import transformers

    spec = cfg.model
    dtype = getattr(torch, cfg.get("model.dtype"))
    device = torch.device(cfg.get("model.device"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model.device is 'cuda' but CUDA is not available (override model.device)")

    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.hf_id)
    tokenizer.padding_side = "left"

    cls = (
        transformers.AutoModelForCausalLM
        if spec.auto_class == "causal_lm"
        else transformers.Gemma3ForConditionalGeneration
    )
    model = cls.from_pretrained(
        spec.hf_id, torch_dtype=dtype, attn_implementation=cfg.get("model.attn_implementation")
    )
    model.to(device).eval()

    layers = get_decoder_layers(model, spec.n_layers)
    d_model = model.get_input_embeddings().embedding_dim
    if d_model != spec.d_model:
        raise AssertionError(f"{spec.name}: hidden size {d_model} != configured d_model {spec.d_model}")

    ids = {
        "pad": tokenizer.pad_token_id,
        "eos": tokenizer.eos_token_id,
        "bos": tokenizer.bos_token_id,
        "turn_start": tokenizer.convert_tokens_to_ids(cfg.get("chat.turn_start_token")),
        "turn_end": tokenizer.convert_tokens_to_ids(cfg.get("chat.turn_end_token")),
    }
    if ids != EXPECTED_IDS:
        raise AssertionError(f"{spec.name}: special token ids {ids} != expected {EXPECTED_IDS}")
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in cfg.get("chat.stop_tokens")]
    if any(i is None or i == tokenizer.unk_token_id for i in stop_ids):
        raise AssertionError(f"stop tokens {cfg.get('chat.stop_tokens')} do not all resolve to ids: {stop_ids}")

    revision = getattr(model.config, "_commit_hash", None) or _cached_revision(spec.hf_id) or "unknown"
    log.info("loaded %s (revision %s, %d layers, d_model %d) on %s", spec.hf_id, revision, len(layers), d_model, device)
    return ModelBundle(model, tokenizer, layers, spec, revision, ids, stop_ids, device)


def free_model(bundle: ModelBundle) -> None:
    bundle.model = None  # type: ignore[assignment]
    bundle.layers = None  # type: ignore[assignment]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- formatting / tokens

def format_prompt(tokenizer: "PreTrainedTokenizerBase", user_text: str) -> str:
    """apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )


def tokenize(bundle: ModelBundle, formatted: list[str], max_len: int) -> dict[str, Tensor]:
    """add_special_tokens=False, left padding. Returns input_ids, attention_mask, position_ids
    (= (mask.cumsum(-1) - 1).clamp(min=0)). Raises if any row > max_len (no silent truncation)
    or if a row does not contain exactly one BOS as its first real token."""
    tok = bundle.tokenizer
    if tok.padding_side != "left":
        raise ValueError("tokenizer.padding_side must be 'left'")
    enc = tok(formatted, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids, mask = enc["input_ids"], enc["attention_mask"]

    lengths = mask.sum(-1)
    too_long = (lengths > max_len).nonzero().flatten().tolist()
    if too_long:
        raise ValueError(
            f"{len(too_long)} prompt(s) exceed max_len={max_len} tokens "
            f"(rows {too_long[:10]}, lengths {lengths[too_long[:10]].tolist()}); no silent truncation"
        )
    bos = bundle.ids["bos"]
    first_real = input_ids.shape[1] - lengths
    n_bos = ((input_ids == bos) & mask.bool()).sum(-1)
    first_tok = input_ids.gather(1, first_real.clamp(max=input_ids.shape[1] - 1).unsqueeze(1)).squeeze(1)
    bad = ((n_bos != 1) | (first_tok != bos)).nonzero().flatten().tolist()
    if bad:
        raise ValueError(
            f"rows {bad[:10]} do not contain exactly one BOS as their first real token "
            "(format with the chat template, which already adds <bos>)"
        )
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": input_ids, "attention_mask": mask, "position_ids": position_ids}


def find_positions(input_ids: Tensor, attention_mask: Tensor, ids: dict[str, int]) -> dict[str, Any]:
    """For ONE (padded) row. ts = index of the last turn_start token.
       turn_start = ts; model_tag = ts+1; last = last real index (assert == ts+2);
       eot_user = last turn_end index < ts; user_last = eot_user-1;
       user_span = (first user-content index, eot_user) — content starts 3 tokens after the
       first turn_start ("<start_of_turn>", "user", "\\n"); BOS never included.
    Raises ValueError with the token ids if any invariant fails."""
    row = input_ids.tolist()
    mask = attention_mask.tolist()

    def fail(msg: str) -> ValueError:
        real = [t for t, m in zip(row, mask) if m]
        return ValueError(f"find_positions: {msg}; real token ids: {real}")

    real_idx = [i for i, m in enumerate(mask) if m]
    if not real_idx:
        raise fail("row has no real tokens")
    last = real_idx[-1]
    starts = [i for i in real_idx if row[i] == ids["turn_start"]]
    if len(starts) < 2:
        raise fail(f"expected at least 2 turn_start tokens, found {len(starts)}")
    ts = starts[-1]
    if last != ts + 2:
        raise fail(f"last real index {last} != last turn_start + 2 ({ts + 2})")
    ends = [i for i in real_idx if row[i] == ids["turn_end"] and i < ts]
    if not ends:
        raise fail("no turn_end token before the last turn_start")
    eot_user = ends[-1]
    span_start = starts[0] + 3
    if span_start >= eot_user:
        raise fail(f"empty user span ({span_start}, {eot_user})")
    if ids["bos"] in row[span_start:eot_user]:
        raise fail("BOS inside the user span")
    return {
        "turn_start": ts,
        "model_tag": ts + 1,
        "last": last,
        "eot_user": eot_user,
        "user_last": eot_user - 1,
        "user_span": (span_start, eot_user),
    }


# --------------------------------------------------------------------------- hooks

def _hidden(output: Any) -> Tensor:
    # Decoder-layer outputs are a tuple in transformers 4.x and a tensor in 5.x.
    return output[0] if isinstance(output, (tuple, list)) else output


@contextmanager
def capture(bundle: ModelBundle, layers: Sequence[int]) -> Iterator[dict[int, list[Tensor]]]:
    """Forward hooks on bundle.layers[L]; each call appends output (output[0] if tuple) .detach().
    Hooks always removed on exit."""
    store: dict[int, list[Tensor]] = {int(layer): [] for layer in layers}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            store[layer].append(_hidden(output).detach())

        return hook

    try:
        for layer in store:
            handles.append(bundle.layers[layer].register_forward_hook(make_hook(layer)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def edit_layer(bundle: ModelBundle, layer: int, fn: Callable[[Tensor], Tensor]) -> Iterator[None]:
    """Forward hook replacing the layer's hidden-state output with fn(hidden); keeps tuple structure.
    Used by steering.py."""

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (fn(output[0]),) + tuple(output[1:])
        if isinstance(output, list):
            return [fn(output[0]), *output[1:]]
        return fn(output)

    handle = bundle.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- generation

@torch.inference_mode()
def generate(bundle: ModelBundle, user_texts: list[str], max_new_tokens: int, batch_size: int,
             intervention: Callable[[], ContextManager] | None = None,
             show_progress: bool = True) -> list[dict]:
    """Greedy (do_sample=False), eos_token_id=bundle.stop_ids, pad_token_id=pad.
    Sort by length (desc) for batching, restore order. `intervention()` (if given) is entered
    around every model.generate call. Returns per text:
    {formatted_prompt, prompt_n_tokens, response (decoded, skip_special_tokens, stripped),
     response_token_ids (list[int]), response_n_tokens, finish_reason: "eos"|"length"}."""
    tok = bundle.tokenizer
    formatted = [format_prompt(tok, t) for t in user_texts]
    lengths = [len(x) for x in tok(formatted, add_special_tokens=False)["input_ids"]]
    order = sorted(range(len(formatted)), key=lambda i: -lengths[i])
    stop = set(bundle.stop_ids)
    results: list[dict | None] = [None] * len(formatted)

    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    for idx in tqdm(batches, desc="generate", disable=not show_progress):
        enc = tokenize(bundle, [formatted[i] for i in idx], max_len=max(lengths[i] for i in idx))
        input_ids = enc["input_ids"].to(bundle.device)
        mask = enc["attention_mask"].to(bundle.device)
        kwargs = dict(
            input_ids=input_ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new_tokens,
            eos_token_id=bundle.stop_ids, pad_token_id=bundle.ids["pad"],
        )
        if intervention is None:
            out = bundle.model.generate(**kwargs)
        else:
            with intervention():
                out = bundle.model.generate(**kwargs)
        new_tokens = out[:, input_ids.shape[1]:].tolist()

        for i, toks in zip(idx, new_tokens):
            cut = next((j for j, t in enumerate(toks) if t in stop), None)
            finish = "length" if cut is None else "eos"
            toks = toks if cut is None else toks[:cut]
            results[i] = {
                "formatted_prompt": formatted[i],
                "prompt_n_tokens": lengths[i],
                "response": tok.decode(toks, skip_special_tokens=True).strip(),
                "response_token_ids": toks,
                "response_n_tokens": len(toks),
                "finish_reason": finish,
            }
    return results  # type: ignore[return-value]
