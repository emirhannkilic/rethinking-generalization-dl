"""
test_models.py — Tests for SmallCNN and MLP architectures.
"""

import pytest
import torch

from src.models import SmallCNN, MLP, count_parameters


def test_small_cnn_shape_and_params():
    model = SmallCNN()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)
    
    params = count_parameters(model)
    assert params > 5_000_000
    assert params < 6_000_000


def test_mlp_shape_and_params():
    model = MLP()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)
    
    params = count_parameters(model)
    assert params == (3072 * 512 + 512) + (512 * 10 + 10)  # 1,578,506


def test_disable_regularization():
    cnn = SmallCNN()
    groups = cnn.disable_regularization()
    assert isinstance(groups, list)
    assert "params" in groups[0]
    assert groups[0].get("weight_decay", 0.0) == 0.0
