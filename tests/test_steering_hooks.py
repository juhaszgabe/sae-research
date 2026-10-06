"""steering.py: intervention hooks (add / clamp / ablate / position gating), specs, degeneracy, trade-off."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import random_texts

from src import model as M
from src import steering as st

LAYER = 2
N_REF = 10.0


@pytest.fixture
def vectors() -> dict[str, np.ndarray]:
    rng = np.random.RandomState(0)
    out = {}
    for name in ("dim", "probe_dir", "random0"):
        v = rng.randn(64)
        out[name] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


@pytest.fixture
def hidden() -> torch.Tensor:
    return torch.randn(2, 5, 64, generator=torch.Generator().manual_seed(0)) * 3


def _hook_counts(bundle) -> list[int]:
    return [len(layer._forward_hooks) for layer in bundle.layers]


def apply(bundle, spec, h, vectors, sae=None, a_ref=None, layer=LAYER) -> torch.Tensor:
    """Runs the hook that `spec` installs on `layer` directly on a hidden-state tensor."""
    with st.intervention(bundle, spec, vectors, N_REF, sae, a_ref)():
        hook = list(bundle.layers[layer]._forward_hooks.values())[-1]  # the most recently registered hook
        return hook(bundle.layers[layer], None, (h.clone(), None))[0]


def layer_output(bundle, texts, factory=None, layer=LAYER) -> torch.Tensor:
    """Output of `layer` in a real forward pass, with an optional intervention active."""
    enc = M.tokenize(bundle, [M.format_prompt(bundle.tokenizer, t) for t in texts], max_len=64)
    with torch.inference_mode():
        if factory is None:
            with M.capture(bundle, [layer]) as store:
                bundle.model(**enc, use_cache=False)
        else:
            # the steering hook is registered first, so the capture hook sees the steered output
            with factory(), M.capture(bundle, [layer]) as store:
                bundle.model(**enc, use_cache=False)
    return store[layer][0]


# --------------------------------------------------------------------------- add

def test_add_alpha_zero_equals_none(bundle, vectors, hidden):
    spec = st.SteerSpec("add", "induce", LAYER, "dim", 0.0)
    assert torch.equal(apply(bundle, spec, hidden, vectors), hidden)

    texts = random_texts(4)
    clean = layer_output(bundle, texts)
    none = layer_output(bundle, texts, st.intervention(bundle, st.SteerSpec("none", "induce"), vectors, N_REF))
    zero = layer_output(bundle, texts, st.intervention(bundle, spec, vectors, N_REF))
    assert torch.equal(clean, none) and torch.equal(clean, zero)


def test_add_shifts_by_alpha_times_reference_norm(bundle, vectors, hidden):
    v = torch.from_numpy(vectors["dim"])
    out = apply(bundle, st.SteerSpec("add", "induce", LAYER, "dim", 0.25), hidden, vectors)
    assert torch.allclose(out - hidden, 0.25 * N_REF * v.expand_as(hidden), atol=1e-5)
    back = apply(bundle, st.SteerSpec("add", "suppress", LAYER, "dim", -0.25), out, vectors)
    assert torch.allclose(back, hidden, atol=1e-5)

    texts = random_texts(4)
    clean = layer_output(bundle, texts)
    steered = layer_output(bundle, texts, st.intervention(bundle, st.SteerSpec("add", "induce", LAYER, "dim", 0.5), vectors, N_REF))
    assert torch.allclose(steered - clean, 0.5 * N_REF * v.expand_as(clean), atol=1e-4)


def test_add_keeps_dtype_and_tuple_structure(bundle, vectors, hidden):
    spec = st.SteerSpec("add", "induce", LAYER, "dim", 1.0)
    with st.intervention(bundle, spec, vectors, N_REF)():
        hook = list(bundle.layers[LAYER]._forward_hooks.values())[-1]
        half = hidden.to(torch.float16)
        as_tuple = hook(bundle.layers[LAYER], None, (half, "attn"))
        as_tensor = hook(bundle.layers[LAYER], None, half)
    assert isinstance(as_tuple, tuple) and as_tuple[1] == "attn" and as_tuple[0].dtype == torch.float16
    assert torch.is_tensor(as_tensor) and as_tensor.dtype == torch.float16  # transformers 5.x: bare tensor output


def test_add_unknown_vector_raises(bundle, vectors):
    with pytest.raises(KeyError, match="sae_top1"):
        st.intervention(bundle, st.SteerSpec("add", "induce", LAYER, "sae_top1", 0.5), vectors, N_REF)


# --------------------------------------------------------------------------- clamp

def _active_feature(sae, h: torch.Tensor) -> int:
    """A feature that fires at every position of h is not guaranteed, so pick the most active one."""
    return int(sae.encode(h).sum(dim=(0, 1)).argmax())


def test_clamp_to_current_value_is_a_no_op(bundle, vectors, fake_sae):
    h = torch.randn(1, 1, 64, generator=torch.Generator().manual_seed(3)) * 3
    feature = _active_feature(fake_sae, h)
    current = float(fake_sae.encode(h)[0, 0, feature])
    assert current > 0
    a_ref = np.zeros(256, dtype=np.float32)
    a_ref[feature] = current
    out = apply(bundle, st.SteerSpec("clamp", "induce", LAYER, "sae_top1", 1.0, feature), h, vectors, fake_sae, a_ref)
    assert torch.allclose(out, h, atol=1e-5)


def test_clamp_to_zero_zeroes_the_feature(bundle, vectors, fake_sae, hidden):
    feature = _active_feature(fake_sae, hidden)
    before = fake_sae.encode(hidden)
    assert before[..., feature].max() > 0
    a_ref = np.ones(256, dtype=np.float32)
    out = apply(bundle, st.SteerSpec("clamp", "suppress", LAYER, "sae_top1", 0.0, feature), hidden, vectors, fake_sae, a_ref)
    assert fake_sae.encode(out)[..., feature].abs().max() == 0
    # only the feature's decoder direction was touched
    w = fake_sae.W_dec[feature]
    assert torch.allclose(out - hidden, -before[..., feature].unsqueeze(-1) * w, atol=1e-5)


def test_clamp_sets_the_feature_to_alpha_times_reference(bundle, vectors, fake_sae, hidden):
    feature = _active_feature(fake_sae, hidden)
    a_ref = np.full(256, 2.0, dtype=np.float32)
    out = apply(bundle, st.SteerSpec("clamp", "induce", LAYER, "sae_top1", 4.0, feature), hidden, vectors, fake_sae, a_ref)
    before, after = fake_sae.encode(hidden)[..., feature], fake_sae.encode(out)[..., feature]
    active = before > 0
    assert active.any() and not active.all()
    # where the feature was firing it now reads exactly α·a_ref = 8
    assert torch.allclose(after[active], torch.full_like(after[active], 8.0), atol=1e-4)
    # where it was silent the update is +8 along the decoder, on top of the sub-threshold pre-activation
    pre = fake_sae.pre_act(hidden, feature)
    assert torch.allclose(after[~active], (pre + 8.0)[~active], atol=1e-4)
    # the SAE error term is kept: nothing changes orthogonally to the decoder direction
    w = fake_sae.W_dec[feature]
    delta = out - hidden
    assert torch.allclose(delta - (delta @ w).unsqueeze(-1) * w, torch.zeros_like(delta), atol=1e-5)


def test_clamp_requires_sae_and_reference(bundle, vectors, fake_sae):
    spec = st.SteerSpec("clamp", "induce", LAYER, "sae_top1", 1.0, 3)
    with pytest.raises(ValueError):
        st.intervention(bundle, spec, vectors, N_REF)
    with pytest.raises(ValueError):
        st.intervention(bundle, st.SteerSpec("clamp", "induce", LAYER, "sae_top1", 1.0), vectors, N_REF, fake_sae, np.ones(256))


# --------------------------------------------------------------------------- ablate

def test_ablation_removes_the_component(bundle, vectors, hidden):
    v = torch.from_numpy(vectors["dim"])
    assert (hidden @ v).abs().max() > 0.1
    out = apply(bundle, st.SteerSpec("ablate", "suppress", None, "dim"), hidden, vectors)
    assert (out @ v).abs().max() < 1e-5
    # everything orthogonal to v is untouched
    assert torch.allclose(out, hidden - (hidden @ v).unsqueeze(-1) * v, atol=1e-5)


def test_ablation_hooks_every_layer(bundle, vectors):
    before = _hook_counts(bundle)
    factory = st.intervention(bundle, st.SteerSpec("ablate", "suppress", None, "dim"), vectors, N_REF)
    with factory():
        assert _hook_counts(bundle) == [n + 1 for n in before]
    assert _hook_counts(bundle) == before

    v = torch.from_numpy(vectors["dim"])
    texts = random_texts(3)
    for layer in range(4):
        assert (layer_output(bundle, texts, layer=layer) @ v).abs().max() > 1e-3
        assert (layer_output(bundle, texts, factory, layer=layer) @ v).abs().max() < 1e-4


# --------------------------------------------------------------------------- position gating / lifecycle

def test_prompt_and_generated_position_gating(bundle, vectors, hidden):
    prompt_call, generated_call = hidden, hidden[:, :1]  # sequence length 5 vs 1 (cached decoding step)
    for positions, changes_prompt, changes_generated in (("all", True, True), ("prompt", True, False),
                                                         ("generated", False, True)):
        spec = st.SteerSpec("add", "induce", LAYER, "dim", 1.0, None, positions)
        assert torch.equal(apply(bundle, spec, prompt_call, vectors), prompt_call) != changes_prompt
        assert torch.equal(apply(bundle, spec, generated_call, vectors), generated_call) != changes_generated
    with pytest.raises(ValueError):
        apply(bundle, st.SteerSpec("add", "induce", LAYER, "dim", 1.0, None, "sometimes"), hidden, vectors)


def test_none_and_prompt_specs_install_no_hooks(bundle, vectors):
    before = _hook_counts(bundle)
    for kind in ("none", "prompt"):
        with st.intervention(bundle, st.SteerSpec(kind, "induce"), vectors, N_REF)():
            assert _hook_counts(bundle) == before


def test_hooks_are_removed_after_use_and_after_errors(bundle, vectors):
    before = _hook_counts(bundle)
    factory = st.intervention(bundle, st.SteerSpec("add", "induce", LAYER, "dim", 1.0), vectors, N_REF)
    for _ in range(2):  # the factory is reusable: generate enters it once per batch
        with factory():
            assert _hook_counts(bundle)[LAYER] == before[LAYER] + 1
        assert _hook_counts(bundle) == before
    with pytest.raises(RuntimeError):
        with factory():
            raise RuntimeError("boom")
    assert _hook_counts(bundle) == before


def test_steering_changes_generation_only_where_gated(bundle, vectors):
    texts = random_texts(3, seed=5)

    def run(spec):
        outs = M.generate(bundle, texts, max_new_tokens=4, batch_size=3,
                          intervention=st.intervention(bundle, spec, vectors, N_REF), show_progress=False)
        return [o["response_token_ids"] for o in outs]

    clean = run(st.SteerSpec("none", "induce"))
    assert run(st.SteerSpec("add", "induce", LAYER, "dim", 0.0)) == clean
    strong = run(st.SteerSpec("add", "induce", LAYER, "dim", 20.0))
    assert strong != clean
    # steering only the generated positions cannot change the first token, which is predicted from the prompt
    generated_only = run(st.SteerSpec("add", "induce", LAYER, "dim", 20.0, None, "generated"))
    assert [ids[0] for ids in generated_only] == [ids[0] for ids in clean]


# --------------------------------------------------------------------------- specs / helpers

def test_slugs():
    assert st.SteerSpec("add", "induce", 13, "dim", 0.25).slug() == "add__dim__L13__a0.25__all__induce"
    assert st.SteerSpec("add", "suppress", 13, "dim", -0.0).slug() == "add__dim__L13__a0__all__suppress"
    assert st.SteerSpec("clamp", "induce", 13, "sae_top1", 8.0, 123, "prompt").slug() == "clamp__sae_top1_f123__L13__a8__prompt__induce"
    assert st.SteerSpec("ablate", "suppress", None, "dim").slug() == "ablate__dim__all__suppress"
    assert st.SteerSpec("none", "induce").slug() != st.SteerSpec("none", "suppress").slug()


def test_make_specs(real_cfg):
    specs = st.make_specs(real_cfg, "refusal", 13, sae_top1_feature=7)
    slugs = [s.slug() for s in specs]
    assert len(slugs) == len(set(slugs))  # every spec has its own generation cache
    kinds = pd.Series([s.kind for s in specs]).value_counts().to_dict()
    # 6 vectors × 5 alphas × 2 directions; 5 + 4 clamp values; ablate dim + refusal_dir; none × 2
    assert kinds == {"add": 60, "clamp": 9, "ablate": 2, "none": 2}
    assert any(s.vector == "random0" for s in specs)  # R6: a random-direction control is always there
    assert all(s.alpha >= 0 for s in specs if s.kind == "add" and s.direction == "induce")
    assert all(s.alpha <= 0 for s in specs if s.kind == "add" and s.direction == "suppress")
    assert all(s.direction == "suppress" for s in specs if s.kind == "ablate")
    assert all(s.feature == 7 and s.layer == 13 for s in specs if s.kind == "clamp")

    fmt = st.make_specs(real_cfg, "format_break", 13, sae_top1_feature=None)
    assert not any(s.vector == "refusal_dir" or (s.vector or "").startswith("sae_") for s in fmt)
    assert not any(s.kind == "clamp" for s in fmt)


def test_make_specs_prompting_baseline(tiny_cfg):
    from conftest import CONFIG_PATH

    from src.config import load_config

    cfg = load_config(CONFIG_PATH, **{"steering.prompting_baseline": True, "steering.ablate": False})
    specs = st.make_specs(cfg, "refusal", 13, None)
    assert [s.direction for s in specs if s.kind == "prompt"] == ["induce", "suppress"]
    assert not any(s.kind == "ablate" for s in specs)
    assert "{text}" in st.PROMPT_TEMPLATES["refusal"]["induce"] and set(st.PROMPT_TEMPLATES) == {"refusal", "format_break", "hedging"}


def test_is_degenerate():
    assert not st.is_degenerate("A normal, varied answer about bread and ovens.", [5, 6, 7, 8], 96)
    assert st.is_degenerate("the cat sat down " * 4, None, 96)  # a word 4-gram repeated 4 times
    assert not st.is_degenerate("the cat sat down " * 3 + "and then it left", None, 96)
    assert st.is_degenerate("ok " + "�" * 5, None, 96)  # > 5 % replacement characters
    assert st.is_degenerate("x", [9] * 96, 96)  # hit the limit repeating one token
    assert not st.is_degenerate("x", [9] * 40, 96)  # same tokens but stopped early
    assert not st.is_degenerate("", [], 96)


def test_cosine_table(vectors):
    table = st.cosine_table({**vectors, "neg_dim": -vectors["dim"]})
    assert list(table.index) == list(table.columns) == ["dim", "probe_dir", "random0", "neg_dim"]
    assert np.allclose(np.diag(table), 1.0, atol=1e-6)
    assert table.loc["dim", "neg_dim"] == pytest.approx(-1.0, abs=1e-6)
    assert np.allclose(table, table.T)


def _sweep_and_side() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [  # slug, kind, direction, vector, alpha, delta_rate, delta_mmlu
        ("a", "add", "induce", "dim", 0.5, 0.60, -0.10),
        ("b", "add", "induce", "dim", 0.25, 0.40, -0.01),
        ("c", "add", "induce", "random0", 0.5, 0.05, -0.10),   # dominated by a
        ("d", "add", "suppress", "dim", -0.5, -0.50, -0.02),
        ("e", "add", "suppress", "random0", -0.5, 0.02, -0.03),  # wrong direction and costlier: dominated by d
    ]
    sweep = pd.DataFrame([{"slug": s, "kind": k, "direction": d, "vector": v, "alpha": a, "delta_rate": r}
                          for s, k, d, v, a, r, _ in rows])
    side = pd.DataFrame([{"slug": s, "delta_mmlu": m} for s, *_, m in rows])
    return sweep, side


def test_tradeoff_pareto_front(tmp_path):
    sweep, side = _sweep_and_side()
    out = st.tradeoff(sweep, side, out=tmp_path / "tradeoff.parquet").set_index("slug")
    assert (tmp_path / "tradeoff.parquet").exists()
    assert out.loc["a", "effect"] == pytest.approx(0.60) and out.loc["d", "effect"] == pytest.approx(0.50)  # sign by direction
    assert out.loc["a", "cost"] == pytest.approx(0.10)
    assert out["pareto"].to_dict() == {"a": True, "b": True, "c": False, "d": True, "e": False}


def test_direction_coherence():
    sweep, side = _sweep_and_side()
    zero = pd.DataFrame([{"slug": "z", "kind": "add", "direction": "induce", "vector": "dim", "alpha": 0.0, "delta_rate": 0.0}])
    sweep = pd.concat([sweep, zero], ignore_index=True)
    side = pd.concat([side, pd.DataFrame([{"slug": "z", "delta_mmlu": 0.0}])], ignore_index=True)
    out = st.direction_coherence(sweep, side).set_index(["vector", "direction"])
    dim = out.loc[("dim", "induce")]
    # trapezoid over α = 0, 0.25, 0.5 with |Δ| = 0, 0.4, 0.6
    assert dim["auc_abs_delta"] == pytest.approx(0.25 * 0.2 + 0.25 * 0.5)
    # α = 0.5 costs 10 MMLU points, so the largest α within the 2-point budget is 0.25
    assert dim["alpha_at_budget"] == pytest.approx(0.25) and dim["effect_at_budget"] == pytest.approx(0.40)
    assert out.loc[("dim", "suppress"), "effect_at_budget"] == pytest.approx(0.50)
    assert np.isnan(out.loc[("random0", "induce"), "effect_at_budget"])  # no α within the budget
