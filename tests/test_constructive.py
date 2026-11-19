"""
test_constructive.py — Tests for Theorem 1 finite-sample expressivity proof.
"""

import pytest
import torch
import numpy as np

from src.constructive import fit_constructive_torch, evaluate_constructive


def test_constructive_cpu_exact_interpolation():
    """Verify Theorem 1 on CPU in float64."""
    n, d = 100, 50
    X = torch.randn(n, d, dtype=torch.float64)
    y = torch.randint(0, 10, (n,))
    
    res = fit_constructive_torch(X, y, seed=42, device=torch.device("cpu"))
    
    assert res["residual"] < 1e-6
    assert res["train_acc"] == 100.0
    assert res["n_params"] == 2 * n + d


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_constructive_gpu_cifar_dimensions():
    """Verify Theorem 1 on GPU with CIFAR-10 input dimension (d=3072)."""
    n, d = 200, 3072
    X = torch.randn(n, d, device="cuda", dtype=torch.float64)
    y = torch.randint(0, 10, (n,), device="cuda")
    
    res = fit_constructive_torch(X, y, seed=42, device=torch.device("cuda"))
    
    assert res["residual"] < 1e-6
    assert res["train_acc"] == 100.0
    assert res["n_params"] == 2 * n + d
    
    # Test generalization on separate random points should be ~10%
    X_test = torch.randn(200, d, device="cuda", dtype=torch.float64)
    y_test = torch.randint(0, 10, (200,), device="cuda")
    test_acc = evaluate_constructive(res, X_test, y_test)
    assert 0.0 <= test_acc <= 30.0  # Around random chance 10%
