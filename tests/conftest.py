"""Shared fakes: a tiny Gemma 3 model, a word-level tokenizer with the Gemma chat format, a fake SAE."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # lets pytest run without `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT))

from src import data  # noqa: E402
from src.config import load_config  # noqa: E402
from src.model import EXPECTED_IDS, ModelBundle, get_decoder_layers  # noqa: E402
from src.sae import JumpReLUSAE  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "model_b.yaml"
WORDS = [f"w{i}" for i in range(400)]

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n"
    "{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)

TINY_REGISTRY = dict(
    hf_id="fake/tiny-it", auto_class="causal_lm", n_layers=4, d_model=64, layers_dense="all", layers_sae=[1, 2],
    sae_release="fake-it-res", sae_release_all="fake-it-res-all",
)


def random_texts(n: int, seed: int = 0, min_words: int = 3, max_words: int = 12) -> list[str]:
    rng = np.random.RandomState(seed)
    return [" ".join(rng.choice(WORDS, rng.randint(min_words, max_words + 1))) for _ in range(n)]


@pytest.fixture(scope="session")
def fake_tokenizer():
    """Word-level tokenizer with the Gemma 3 special ids and chat format; `user`, `model` and the
    newline are single tokens; left padding. Unknown words map to <unk>, so tests use the w{i} words."""
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "user": 4, "model": 5, "\n": 6,
             "A": 7, "B": 8, "C": 9, "D": 10, "<start_of_turn>": 105, "<end_of_turn>": 106}
    free = [i for i in range(512) if i not in set(vocab.values())]
    vocab.update(zip(WORDS, free))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>", unk_token="<unk>",
        additional_special_tokens=["<start_of_turn>", "<end_of_turn>"],
    )
    tok.add_tokens([AddedToken("\n", normalized=False)])
    tok.padding_side = "left"
    tok.chat_template = CHAT_TEMPLATE
    return tok


@pytest.fixture(scope="session")
def tiny_model():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, sliding_window=16, max_position_embeddings=512,
        pad_token_id=0, bos_token_id=2, eos_token_id=1,
    )
    return Gemma3ForCausalLM(config).eval()


@pytest.fixture
def tiny_cfg(tmp_path, monkeypatch):
    """The real config pointed at the tiny model, with small grids and outputs under tmp_path."""
    monkeypatch.setenv("SBM_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("SBM_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    return load_config(CONFIG_PATH, **{
        "model_registry.tiny-it": dict(TINY_REGISTRY), "model.name": "tiny-it", "model.device": "cpu",
        "sae.widths": ["16k"], "sae.main_width": "16k", "sae.device": "cpu",
        "eval.n_jobs": 1, "eval.n_bootstrap": 100,
        "probes.logreg_C": [0.01, 1.0], "probes.dense_topk_C": [1.0], "probes.sae_C": [0.1, 1.0],
        "probes.k_grid": [1, 4, 16],
    })


@pytest.fixture
def real_cfg(tmp_path, monkeypatch):
    """The unmodified experiment config (outputs still redirected to tmp_path)."""
    monkeypatch.setenv("SBM_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("SBM_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    return load_config(CONFIG_PATH)


@pytest.fixture
def bundle(tiny_model, fake_tokenizer, tiny_cfg) -> ModelBundle:
    return ModelBundle(
        model=tiny_model, tokenizer=fake_tokenizer, layers=get_decoder_layers(tiny_model, 4), spec=tiny_cfg.model,
        revision="test", ids=dict(EXPECTED_IDS), stop_ids=[1, 106], device=torch.device("cpu"),
    )


@pytest.fixture
def fake_sae() -> JumpReLUSAE:
    """Random JumpReLU SAE, D=64, W=256, positive thresholds. Decoder rows are unit norm and the
    encoder is tied (W_enc = W_dec^T), so removing f_i·W_dec[i] lowers pre-activation i by exactly f_i."""
    g = torch.Generator().manual_seed(1)
    W_dec = torch.randn(256, 64, generator=g)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
    threshold = torch.rand(256, generator=g) + 0.5
    return JumpReLUSAE(W_dec.T.clone(), torch.zeros(256), W_dec, torch.zeros(64), threshold)


@pytest.fixture
def synthetic_dataset() -> SimpleNamespace:
    """200 rows in 40 groups with a planted signal: dense dims 0-2 are shifted by the label and
    sparse feature 7 fires almost only for label 1. Splits come from data.make_splits."""
    rng = np.random.RandomState(0)
    n, n_groups, d, w = 200, 40, 32, 100
    df = pd.DataFrame({
        "prompt_id": [f"p{i:04d}" for i in range(n)],
        "group_id": [f"g{i % n_groups:02d}" for i in range(n)],
        "label": rng.permutation(np.repeat([0, 1], n // 2)),
        "prompt_category": rng.choice(["harmful", "harmless", "borderline_safe"], n),
        "source": rng.choice(["advbench", "alpaca", "xstest"], n),
    })
    df = data.make_splits(df, test_frac=0.2, n_folds=5, seed=0)
    y = df["label"].to_numpy()

    X = rng.randn(n, d).astype(np.float32)
    X[:, :3] += 1.5 * y[:, None]

    codes = (rng.rand(n, w) < 0.1) * rng.rand(n, w)
    fires = rng.rand(n) < np.where(y == 1, 0.9, 0.05)
    codes[:, 7] = fires * (2.0 + rng.rand(n))
    codes = sp.csr_matrix(codes.astype(np.float32))

    return SimpleNamespace(
        df=df, X=X, codes=codes, y=y, groups=df["group_id"].to_numpy(), folds=df["fold"].to_numpy(),
        is_test=(df["split"] == "test").to_numpy(), categories=df["prompt_category"].to_numpy(), planted_feature=7,
    )


def eval_kwargs(ds: SimpleNamespace) -> dict:
    """The (y, groups, folds, is_test) arguments of eval.evaluate_method."""
    return dict(y=ds.y, groups=ds.groups, folds=ds.folds, is_test=ds.is_test)
