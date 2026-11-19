"""
test_data.py — Tests for CIFAR-10 loading and corruption pipelines.
"""

import pytest
import torch
import numpy as np

from src.data import (
    apply_label_corruption,
    apply_input_corruption,
    build_tensors,
)


def test_label_corruption_rate():
    """Verify that label corruption corrupts the specified fraction."""
    n = 1000
    y = np.random.randint(0, 10, size=n)
    
    for rate in [0.0, 0.2, 0.5, 0.8, 1.0]:
        y_corrupt, y_orig = apply_label_corruption(y, rate, seed=42)
        assert len(y_corrupt) == n
        if rate == 0.0:
            assert torch.equal(y_corrupt, torch.tensor(y, dtype=torch.long))
        elif rate == 1.0:
            match_rate = (y_corrupt == y_orig).float().mean().item()
            assert match_rate < 0.25
        else:
            diff_fraction = (y_corrupt != y_orig).float().mean().item()
            # Random relabeling might by chance draw the same label for 10% of corruptions,
            # so expected change is ~ rate * 0.9
            assert diff_fraction > 0.0
            assert np.isclose(diff_fraction, rate * 0.9, atol=0.08)


def test_label_corruption_reproducibility():
    """Verify that same seed produces exact same corruption."""
    y = np.random.randint(0, 10, size=500)
    y_c1, _ = apply_label_corruption(y, 0.4, seed=123)
    y_c2, _ = apply_label_corruption(y, 0.4, seed=123)
    y_c3, _ = apply_label_corruption(y, 0.4, seed=456)
    
    assert torch.equal(y_c1, y_c2)
    assert not torch.equal(y_c1, y_c3)


def test_input_corruptions():
    """Test all 4 input modes."""
    # (N, C, H, W)
    images = torch.rand(10, 3, 32, 32)
    
    # 1. True (no change)
    t_img = apply_input_corruption(images, "true", seed=42)
    assert torch.equal(t_img, images)
    
    # 2. Shuffled pixels (same permutation across all images)
    s_img = apply_input_corruption(images, "shuffled_pixels", seed=42)
    assert s_img.shape == images.shape
    assert not torch.equal(s_img, images)
    assert np.isclose(s_img.mean().item(), images.mean().item(), atol=1e-5)
    
    # 3. Random pixels
    r_img = apply_input_corruption(images, "random_pixels", seed=42)
    assert r_img.shape == images.shape
    assert (r_img.min() >= 0.0) and (r_img.max() <= 1.0)
    
    # 4. Gaussian noise
    g_img = apply_input_corruption(images, "gaussian_noise", seed=42)
    assert g_img.shape == images.shape
    assert (g_img.min() >= 0.0) and (g_img.max() <= 1.0)


def test_build_tensors_smoke():
    """Verify tensor builder in smoke_test mode."""
    device = torch.device("cpu")
    train_X, train_y, test_X, test_y = build_tensors(
        data_dir="data/",
        corruption_mode="true",
        label_corruption_rate=0.2,
        seed=42,
        device=device,
        smoke_test=True,
    )
    assert train_X.shape == (500, 3, 32, 32)
    assert train_y.shape == (500,)
    assert test_X.shape == (100, 3, 32, 32)
    assert test_y.shape == (100,)
