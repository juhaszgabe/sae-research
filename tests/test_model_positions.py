"""model.py: chat formatting, tokenization checks, token positions, layer resolution, hooks."""

from __future__ import annotations

import pytest
import torch
from conftest import CONFIG_PATH, random_texts

from src import activations
from src import model as M
from src.config import load_config


def _hook_counts(bundle) -> list[int]:
    # transformers may keep hooks of its own on the layers, so tests compare before/after counts
    return [len(layer._forward_hooks) for layer in bundle.layers]


def _forward(bundle, texts):
    enc = M.tokenize(bundle, [M.format_prompt(bundle.tokenizer, t) for t in texts], max_len=64)
    with torch.inference_mode():
        return bundle.model(**enc, use_cache=False)


# --------------------------------------------------------------------------- formatting / tokenization

def test_format_prompt_matches_gemma_template(fake_tokenizer):
    assert M.format_prompt(fake_tokenizer, "w1 w2") == "<bos><start_of_turn>user\nw1 w2<end_of_turn>\n<start_of_turn>model\n"


def test_tokenize_left_pads_and_builds_position_ids(bundle):
    formatted = [M.format_prompt(bundle.tokenizer, t) for t in ("w1 w2 w3", "w1 w2 w3 w4 w5 w6")]
    enc = M.tokenize(bundle, formatted, max_len=64)
    assert enc["input_ids"].shape == (2, 15)
    assert enc["attention_mask"][0].tolist() == [0, 0, 0] + [1] * 12
    assert enc["input_ids"][0, :3].tolist() == [0, 0, 0]
    assert enc["position_ids"][0].tolist() == [0, 0, 0] + list(range(12))
    assert enc["position_ids"][1].tolist() == list(range(15))


def test_tokenize_raises_on_too_long(bundle):
    formatted = [M.format_prompt(bundle.tokenizer, "w1 w2 w3 w4 w5 w6")]
    with pytest.raises(ValueError, match="max_len"):
        M.tokenize(bundle, formatted, max_len=10)


def test_tokenize_requires_exactly_one_leading_bos(bundle):
    good = M.format_prompt(bundle.tokenizer, "w1 w2 w3")
    with pytest.raises(ValueError, match="BOS"):
        M.tokenize(bundle, ["<bos>" + good], max_len=64)  # two BOS tokens
    with pytest.raises(ValueError, match="BOS"):
        M.tokenize(bundle, [good.removeprefix("<bos>")], max_len=64)  # no BOS


# --------------------------------------------------------------------------- positions

def test_find_positions_without_padding(bundle):
    # <bos> <sot> user \n w1 w2 w3 <eot> \n <sot> model \n
    enc = M.tokenize(bundle, [M.format_prompt(bundle.tokenizer, "w1 w2 w3")], max_len=64)
    pos = M.find_positions(enc["input_ids"][0], enc["attention_mask"][0], bundle.ids)
    assert pos == {"turn_start": 9, "model_tag": 10, "last": 11, "eot_user": 7, "user_last": 6, "user_span": (4, 7)}


def test_find_positions_with_left_padding(bundle):
    formatted = [M.format_prompt(bundle.tokenizer, t) for t in ("w1 w2 w3", "w1 w2 w3 w4 w5 w6")]
    enc = M.tokenize(bundle, formatted, max_len=64)
    short = M.find_positions(enc["input_ids"][0], enc["attention_mask"][0], bundle.ids)
    assert short == {"turn_start": 12, "model_tag": 13, "last": 14, "eot_user": 10, "user_last": 9, "user_span": (7, 10)}
    long = M.find_positions(enc["input_ids"][1], enc["attention_mask"][1], bundle.ids)
    assert long == {"turn_start": 12, "model_tag": 13, "last": 14, "eot_user": 10, "user_last": 9, "user_span": (4, 10)}


def test_find_positions_tokens_are_the_expected_ones(bundle):
    enc = M.tokenize(bundle, [M.format_prompt(bundle.tokenizer, "w1 w2 w3")], max_len=64)
    ids = enc["input_ids"][0]
    pos = M.find_positions(ids, enc["attention_mask"][0], bundle.ids)
    assert ids[pos["turn_start"]] == bundle.ids["turn_start"]
    assert ids[pos["eot_user"]] == bundle.ids["turn_end"]
    assert bundle.tokenizer.decode([int(ids[pos["model_tag"]])]) == "model"
    assert bundle.tokenizer.decode([int(ids[pos["user_last"]])]) == "w3"
    start, end = pos["user_span"]
    assert bundle.tokenizer.convert_ids_to_tokens(ids[start:end].tolist()) == ["w1", "w2", "w3"]
    assert bundle.ids["bos"] not in ids[start:end].tolist()


def test_find_positions_rejects_malformed_rows(bundle):
    ids = bundle.ids
    mask = torch.ones(6, dtype=torch.long)
    with pytest.raises(ValueError):  # no generation prompt: only one turn_start
        M.find_positions(torch.tensor([ids["bos"], ids["turn_start"], 4, 6, 20, ids["turn_end"]]), mask, ids)
    with pytest.raises(ValueError):  # last token is not two after the final turn_start
        M.find_positions(torch.tensor([ids["bos"], ids["turn_start"], 4, 6, 20, ids["turn_end"], 6,
                                       ids["turn_start"], 5, 6, 21]), torch.ones(11, dtype=torch.long), ids)
    with pytest.raises(ValueError):  # empty user turn
        M.find_positions(torch.tensor([ids["bos"], ids["turn_start"], 4, 6, ids["turn_end"], 6,
                                       ids["turn_start"], 5, 6]), torch.ones(9, dtype=torch.long), ids)


# --------------------------------------------------------------------------- decoder layers

def _layers(n: int) -> torch.nn.ModuleList:
    return torch.nn.ModuleList([torch.nn.Identity() for _ in range(n)])


def _shim(path: str, n: int) -> torch.nn.Module:
    root = torch.nn.Module()
    node = root
    parts = path.split(".")
    for part in parts[:-1]:
        child = torch.nn.Module()
        setattr(node, part, child)
        node = child
    setattr(node, parts[-1], _layers(n))
    return root


@pytest.mark.parametrize("path", [
    "model.language_model.layers", "language_model.model.layers", "language_model.layers", "model.layers",
])
def test_get_decoder_layers_on_shimmed_layouts(path):
    root = _shim(path, 6)
    layers = M.get_decoder_layers(root, 6)
    assert isinstance(layers, torch.nn.ModuleList) and len(layers) == 6


def test_get_decoder_layers_skips_wrong_length_and_reports(tiny_model):
    root = _shim("model.layers", 6)
    root.model.vision = _layers(3)
    with pytest.raises(ValueError, match="model.layers"):
        M.get_decoder_layers(root, 5)
    assert len(M.get_decoder_layers(tiny_model, 4)) == 4


# --------------------------------------------------------------------------- hooks

def test_capture_matches_hidden_states(bundle):
    res = activations.check_hooks_vs_hidden_states(bundle, random_texts(6))
    assert res["ok"], res
    assert len(res["median_norm_per_layer"]) == 4


def test_capture_shape_and_removal(bundle):
    before = _hook_counts(bundle)
    with M.capture(bundle, [0, 2]) as store:
        _forward(bundle, ["w1 w2 w3", "w4 w5"])
    assert set(store) == {0, 2}
    assert store[2][0].shape == (2, 12, 64)
    assert _hook_counts(bundle) == before


def test_hooks_removed_after_exceptions(bundle):
    before = _hook_counts(bundle)
    with pytest.raises(RuntimeError, match="boom"):
        with M.capture(bundle, [0, 1, 2, 3]):
            raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        with M.edit_layer(bundle, 1, lambda h: h):
            raise RuntimeError("boom")

    def explode(h):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        with M.edit_layer(bundle, 1, explode):
            _forward(bundle, ["w1 w2 w3"])
    assert _hook_counts(bundle) == before


def test_edit_layer_replaces_the_hidden_state(bundle):
    with M.capture(bundle, [1, 2]) as clean:
        _forward(bundle, ["w1 w2 w3"])
    # the edit hook is registered first, so the capture hook sees the edited output
    with M.edit_layer(bundle, 1, lambda h: torch.zeros_like(h)), M.capture(bundle, [1, 2]) as edited:
        _forward(bundle, ["w1 w2 w3"])
    assert clean[1][0].abs().sum() > 0
    assert edited[1][0].abs().sum() == 0
    assert not torch.allclose(clean[2][0], edited[2][0])


# --------------------------------------------------------------------------- batching equivalence / generation

def test_padding_equivalence_on_tiny_model(bundle):
    res = activations.check_padding_equivalence(bundle, random_texts(8), layer=2)
    assert res["ok"], res


def test_extract_returns_all_positions_in_input_order(bundle):
    texts = random_texts(7, seed=3)
    out = activations.extract(bundle, texts, [1, 3], batch_size=3)
    assert set(out) == {1, 3}
    assert set(out[1]) == set(activations.POSITIONS) | set(activations.POOLED)
    assert out[3]["last"].shape == (7, 64) and out[3]["last"].dtype.name == "float32"
    single = activations.extract(bundle, [texts[4]], [3], batch_size=1)
    assert torch.allclose(torch.from_numpy(out[3]["mean_user"][4]), torch.from_numpy(single[3]["mean_user"][0]), atol=1e-3)


def test_generate_output_fields_and_order(bundle):
    texts = random_texts(5, seed=1)
    outs = M.generate(bundle, texts, max_new_tokens=4, batch_size=2, show_progress=False)
    assert len(outs) == 5
    for text, out in zip(texts, outs):
        assert set(out) == {"formatted_prompt", "prompt_n_tokens", "response", "response_token_ids",
                            "response_n_tokens", "finish_reason"}
        assert out["formatted_prompt"] == M.format_prompt(bundle.tokenizer, text)
        assert out["finish_reason"] in ("eos", "length")
        assert out["response_n_tokens"] == len(out["response_token_ids"]) <= 4
        assert not set(out["response_token_ids"]) & set(bundle.stop_ids)
    assert activations.check_generation_equivalence(bundle, texts, n_tokens=4)["ok"]


def test_generate_enters_the_intervention(bundle):
    from contextlib import contextmanager

    calls = []

    @contextmanager
    def intervention():
        calls.append("enter")
        yield
        calls.append("exit")

    M.generate(bundle, random_texts(3), max_new_tokens=2, batch_size=2, intervention=intervention, show_progress=False)
    assert calls == ["enter", "exit"] * 2  # one model.generate call per batch


# --------------------------------------------------------------------------- real model

@pytest.mark.hf
def test_real_270m_positions_and_hooks():
    cfg = load_config(CONFIG_PATH, **{"model.name": "gemma-3-270m-it", "model.device": "cpu", "model.dtype": "float32"})
    bundle = M.load_model(cfg)
    try:
        texts = ["What is the capital of France?", "Write a haiku about autumn leaves."]
        enc = M.tokenize(bundle, [M.format_prompt(bundle.tokenizer, t) for t in texts], max_len=128)
        for i in range(len(texts)):
            pos = M.find_positions(enc["input_ids"][i], enc["attention_mask"][i], bundle.ids)
            assert bundle.tokenizer.decode([int(enc["input_ids"][i][pos["model_tag"]])]).strip() == "model"
        assert activations.check_hooks_vs_hidden_states(bundle, texts)["ok"]
        assert activations.check_padding_equivalence(bundle, texts, layer=9)["ok"]
    finally:
        M.free_model(bundle)
