"""sae.py: JumpReLU encode/decode, sparse encoding, quality metrics, lookup helpers."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from conftest import CONFIG_PATH

from src import sae as S
from src.config import load_config


@pytest.fixture
def hand_sae() -> S.JumpReLUSAE:
    """D=2, W=3. Feature 0 reads x0, feature 1 reads x1, feature 2 reads x0 + x1 (with a bias)."""
    W_enc = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    b_enc = torch.tensor([0.0, 0.0, -1.0])
    W_dec = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    b_dec = torch.tensor([0.5, -0.5])
    threshold = torch.tensor([1.0, 1.0, 2.0])
    return S.JumpReLUSAE(W_enc, b_enc, W_dec, b_dec, threshold)


def test_encode_is_jumprelu(hand_sae):
    # pre = [x0, x1, x0 + x1 - 1]; a feature is kept only if pre > threshold (strictly)
    x = torch.tensor([[3.0, 0.5], [1.0, 2.5], [-4.0, 9.0]])
    expected = torch.tensor([[3.0, 0.0, 2.5], [0.0, 2.5, 2.5], [0.0, 9.0, 4.0]])
    assert torch.equal(hand_sae.encode(x), expected)


def test_pre_activation_equal_to_threshold_gives_zero(hand_sae):
    # pre = [1.0, 1.0, 1.0]: features 0 and 1 sit exactly on their threshold
    assert hand_sae.encode(torch.tensor([[1.0, 1.0]])).tolist() == [[0.0, 0.0, 0.0]]
    # just above the threshold the full pre-activation passes (a jump, not a shifted ReLU)
    assert hand_sae.encode(torch.tensor([[1.25, 0.0]])).tolist() == [[1.25, 0.0, 0.0]]


def test_decode_and_forward(hand_sae):
    f = torch.tensor([[3.0, 0.0, 2.5]])
    assert torch.equal(hand_sae.decode(f), torch.tensor([[3.0 + 7.5 + 0.5, 10.0 - 0.5]]))
    x = torch.tensor([[3.0, 0.5]])
    assert torch.equal(hand_sae(x), hand_sae.decode(hand_sae.encode(x)))


def test_no_input_normalization_or_b_dec_subtraction(hand_sae):
    # encode must not subtract b_dec: x = b_dec would otherwise give all-zero pre-activations
    x = torch.tensor([[2.0, 0.0]])
    assert hand_sae.encode(x)[0, 0] == 2.0


def test_direction_and_pre_act(hand_sae):
    d = hand_sae.direction(2)
    assert torch.allclose(d, torch.tensor([0.6, 0.8]))
    assert torch.isclose(hand_sae.direction(1).norm(), torch.tensor(1.0))
    x = torch.tensor([[3.0, 0.5], [1.0, 2.5]])
    assert torch.equal(hand_sae.pre_act(x, 2), torch.tensor([2.5, 2.5]))
    assert torch.equal(hand_sae.pre_act(x, 0), x[:, 0])


def test_encode_casts_to_float32_and_keeps_leading_dims(hand_sae, fake_sae):
    x = torch.randn(2, 5, 64, dtype=torch.float64)
    f = fake_sae.encode(x)
    assert f.shape == (2, 5, 256) and f.dtype == torch.float32
    assert fake_sae.decode(f).shape == (2, 5, 64)
    assert (fake_sae.d_model, fake_sae.width) == (64, 256)
    assert (hand_sae.d_model, hand_sae.width) == (2, 3)


def test_encode_matrix_equals_dense_encode(fake_sae):
    X = (np.random.RandomState(0).randn(23, 64) * 3).astype(np.float32)
    codes = S.encode_matrix(fake_sae, X, batch_size=7)
    assert sp.isspmatrix_csr(codes) and codes.shape == (23, 256) and codes.dtype == np.float32
    dense = fake_sae.encode(torch.from_numpy(X)).numpy()
    assert np.allclose(codes.toarray(), dense, atol=1e-6)
    assert codes.nnz == int((dense != 0).sum())


def test_encode_matrix_empty_input(fake_sae):
    codes = S.encode_matrix(fake_sae, np.zeros((0, 64), dtype=np.float32))
    assert codes.shape == (0, 256)


def test_feature_max_act():
    codes = sp.csr_matrix(np.array([[0.0, 2.0, 0.0], [1.0, 5.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32))
    assert S.feature_max_act(codes).tolist() == [3.0, 5.0, 0.0]
    assert S.feature_max_act(codes, np.array([True, True, False])).tolist() == [1.0, 5.0, 0.0]
    assert S.feature_max_act(codes).dtype == np.float32


def test_sae_quality_of_a_perfect_and_a_poor_sae(fake_sae):
    # identity SAE with zero thresholds reconstructs positive inputs exactly
    eye = S.JumpReLUSAE(torch.eye(4), torch.zeros(4), torch.eye(4), torch.zeros(4), torch.zeros(4))
    X = (np.random.RandomState(0).rand(50, 4) + 0.1).astype(np.float32)
    q = S.sae_quality(eye, X)
    assert set(q) == {"l0_mean", "l0_median", "fvu", "cos_mean", "frac_features_active"}
    assert q["fvu"] < 1e-8 and q["cos_mean"] > 0.999
    assert q["l0_mean"] == 4.0 and q["frac_features_active"] == 1.0
    assert S.quality_status(q, l0_target=4.0) == "ok"
    assert S.quality_status(q, l0_target=60.0) == "warning"  # L0 far from the target

    bad = S.sae_quality(fake_sae, (np.random.RandomState(1).randn(50, 64) * 3).astype(np.float32))
    assert bad["fvu"] > 1  # a random SAE does not reconstruct
    assert S.quality_status(bad, l0_target=float("nan")) == "error"


def test_neuronpedia_url():
    info = S.SaeInfo("rel", "layer_13_width_65k_l0_medium", "repo", "folder", 13, "65k", "medium", 60.0,
                     "google/gemma-3-1b-it", "gemma-3-1b-it/13-gemmascope-2-res-65k")
    assert S.neuronpedia_url(info, 42) == "https://www.neuronpedia.org/gemma-3-1b-it/13-gemmascope-2-res-65k/42"
    no_dashboard = S.SaeInfo("rel", "id", "repo", "folder", 13, "65k", "small", 20.0, "google/gemma-3-1b-it", None)
    assert S.neuronpedia_url(no_dashboard, 42) is None
    assert S.fetch_neuronpedia(no_dashboard, 42) is None  # no id → no request


@pytest.mark.hf
def test_real_sae_backends_agree():
    cfg = load_config(CONFIG_PATH, **{"model.name": "gemma-3-270m-it", "sae.device": "cpu"})
    layer = cfg.model.layers_sae[1]
    a = S.load_sae(cfg, layer, "16k", backend="saelens")
    b = S.load_sae(cfg, layer, "16k", backend="raw")
    x = torch.randn(8, cfg.model.d_model) * 10
    assert torch.allclose(a.encode(x), b.encode(x), atol=1e-5)
    assert a.info.model_hf_id.endswith("gemma-3-270m-it") and a.info.sae_id == f"layer_{layer}_width_16k_l0_medium"
