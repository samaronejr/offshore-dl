"""Tests for per-class reconstruction DeepONet classifier."""

import pytest
import torch
import torch.nn as nn
from offshore_dl.models.deeponet_recon_clf import (
    FourierPositionalEncoding,
    LSTMBranch,
    LoRALinear,
    DeepONetReconClassifier,
)


class TestFourierPositionalEncoding:
    def test_output_shape(self):
        enc = FourierPositionalEncoding(n_frequencies=8)
        positions = torch.rand(100, 2)
        out = enc(positions)
        assert out.shape == (100, 2 + 4 * 8)

    def test_frequencies_learnable(self):
        enc = FourierPositionalEncoding(n_frequencies=8)
        assert enc.frequencies.requires_grad


class TestLSTMBranch:
    def test_output_shape(self):
        branch = LSTMBranch(n_vars=27, hidden_size=64, num_layers=2, rank=32)
        x = torch.randn(4, 100, 27)
        out = branch(x)
        assert out.shape == (4, 32)

    def test_gradient_flow(self):
        branch = LSTMBranch(n_vars=27, hidden_size=64, rank=32)
        x = torch.randn(2, 50, 27)
        out = branch(x)
        out.sum().backward()
        for p in branch.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestLoRALinear:
    def test_output_shape(self):
        base = nn.Linear(64, 128)
        lora = LoRALinear(base, lora_rank=4)
        x = torch.randn(8, 64)
        out = lora(x)
        assert out.shape == (8, 128)

    def test_base_frozen(self):
        base = nn.Linear(64, 128)
        lora = LoRALinear(base, lora_rank=4)
        assert not lora.base_layer.weight.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_lora_initial_output_matches_base(self):
        base = nn.Linear(64, 128)
        x = torch.randn(4, 64)
        base_out = base(x).detach()
        lora = LoRALinear(base, lora_rank=4)
        lora_out = lora(x).detach()
        # lora_B initialized to zeros, so output should match base
        torch.testing.assert_close(base_out, lora_out, atol=1e-6, rtol=1e-6)


class TestDeepONetReconClassifier:
    def test_build_base_model(self):
        clf = DeepONetReconClassifier(
            n_classes=3, n_vars=5, window_size=20,
            rank=16, branch_hidden=32, branch_layers=1,
            trunk_hidden=[32], n_frequencies=4,
        )
        branch, pos_enc, trunk, bias = clf._build_base_model()
        assert bias.shape == (20 * 5,)

    def test_forward_reconstructs(self):
        clf = DeepONetReconClassifier(
            n_classes=3, n_vars=5, window_size=20,
            rank=16, branch_hidden=32, branch_layers=1,
            trunk_hidden=[32], n_frequencies=4,
        )
        branch, pos_enc, trunk, bias = clf._build_base_model()
        x = torch.randn(2, 20, 5)
        recon = clf._forward(x, branch, pos_enc, trunk, bias)
        assert recon.shape == (2, 20, 5)

    def test_add_lora_increases_trainable_params(self):
        clf = DeepONetReconClassifier(
            n_classes=3, n_vars=5, window_size=20,
            rank=16, branch_hidden=32, branch_layers=1,
            trunk_hidden=[32], n_frequencies=4, lora_rank=4,
        )
        _, _, trunk, _ = clf._build_base_model()
        # Freeze base trunk
        for p in trunk.parameters():
            p.requires_grad = False
        lora_trunk = clf._add_lora(trunk)
        lora_trainable = sum(p.numel() for p in lora_trunk.parameters() if p.requires_grad)
        assert lora_trainable > 0
