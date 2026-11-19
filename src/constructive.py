"""
constructive.py — Constructive proof of Theorem 1 from Zhang et al. (2017).

Implements the 2n + d parameter network that can perfectly fit any
n labelled samples in d dimensions using a single hidden ReLU layer.

The construction:
  1. Draw a random direction a ~ N(0, I_d).
  2. Project each sample: z_i = a · x_i.
  3. Choose n biases b_j between consecutive sorted z values so that
     the activation matrix A (A_ij = max(z_i - b_j, 0)) is invertible.
  4. Solve A · w = y_one_hot for w, where A is (n × n) in float64.
  5. Verify the residual is below numerical tolerance and train accuracy is 100%.

Supports both GPU (CUDA) and CPU acceleration using PyTorch in float64.

Usage (standalone):
    python -m src.constructive --device cuda --n 1000 --use-cifar
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

from src.utils import get_device


# ---------------------------------------------------------------------------
# Core construction (PyTorch float64, GPU-compatible)
# ---------------------------------------------------------------------------

def fit_constructive_torch(
    X: torch.Tensor,
    y: torch.Tensor,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> Dict[str, Any]:
    """Construct a 2-layer ReLU network that interpolates every training sample.

    Parameters
    ----------
    X:
        Input tensor of shape ``(n, d)`` in float64 on target device.
    y:
        Label tensor of shape ``(n,)`` with integer entries in [0, 9].
    seed:
        RNG seed for reproducibility of random direction **a**.
    device:
        Torch device to compute on (CPU or CUDA).

    Returns
    -------
    dict with keys:
    ``a``           — shape ``(d,)`` random direction.
    ``biases``      — shape ``(n,)`` threshold biases b_j.
    ``weights``     — shape ``(n, 10)`` output weights w.
    ``residual``    — scalar residual ‖Aw − y_onehot‖_F.
    ``train_acc``   — interpolation accuracy on training samples (0-100%).
    ``n_params``    — total parameter count = 2n + d.
    """
    if device is None:
        device = X.device
    else:
        device = torch.device(device)

    X = X.to(device=device, dtype=torch.float64)
    y = y.to(device=device, dtype=torch.int64)
    n, d = X.shape

    torch.manual_seed(seed)

    # ---- Step 1: Random projection direction ----------------------------- #
    a = torch.randn(d, device=device, dtype=torch.float64)
    a = a / (torch.norm(a) + 1e-12)

    # ---- Step 2: Projections z_i = a · x_i ------------------------------ #
    z = X @ a

    # ---- Step 3: Bias values between consecutive sorted projections ------- #
    sort_idx = torch.argsort(z)
    z_sorted = z[sort_idx]

    # Ensure strict ordering by nudging near-ties.
    for i in range(1, n):
        if z_sorted[i] <= z_sorted[i - 1]:
            z_sorted[i] = z_sorted[i - 1] + 1e-10

    # Place biases at midpoints between consecutive sorted z values.
    b = torch.empty(n, device=device, dtype=torch.float64)
    b[0] = z_sorted[0] - 1.0
    b[1:] = 0.5 * (z_sorted[:-1] + z_sorted[1:])

    # ---- Step 4: Activation matrix A_ij = max(z_i - b_j, 0) ------------- #
    A = torch.clamp(z[:, None] - b[None, :], min=0.0)

    # ---- Step 5: One-hot target ------------------------------------------ #
    num_classes = 10
    y_onehot = torch.zeros(n, num_classes, device=device, dtype=torch.float64)
    y_onehot.scatter_(1, y[:, None], 1.0)

    # ---- Step 6: Solve A · w = y_onehot on GPU/CPU ----------------------- #
    w = torch.linalg.solve(A, y_onehot)

    # ---- Step 7: Verify residual and accuracy --------------------------- #
    residual = torch.norm(A @ w - y_onehot).item()
    preds = (A @ w).argmax(dim=1)
    train_acc = (preds == y).float().mean().item() * 100.0

    n_params = 2 * n + d

    print(
        f"[constructive] device={device.type} | n={n:<5} d={d} | "
        f"residual={residual:.2e} | train_acc={train_acc:.1f}% | "
        f"n_params(2n+d)={n_params}"
    )

    return {
        "a": a,
        "biases": b,
        "weights": w,
        "residual": residual,
        "train_acc": train_acc,
        "n_params": n_params,
    }


def evaluate_constructive(
    model_dict: Dict[str, Any],
    test_X: torch.Tensor,
    test_y: torch.Tensor,
) -> float:
    """Evaluate generalization of the constructed network on test points."""
    device = model_dict["a"].device
    test_X = test_X.to(device=device, dtype=torch.float64)
    test_y = test_y.to(device=device, dtype=torch.int64)

    a = model_dict["a"]
    b = model_dict["biases"]
    w = model_dict["weights"]

    z_test = test_X @ a
    A_test = torch.clamp(z_test[:, None] - b[None, :], min=0.0)
    preds = (A_test @ w).argmax(dim=1)
    test_acc = (preds == test_y).float().mean().item() * 100.0
    return test_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run GPU-accelerated constructive proof of Theorem 1."
    )
    p.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of training samples (default: 1000)",
    )
    p.add_argument(
        "--d",
        type=int,
        default=3072,
        help="Input dimension (default: 3072)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run on (default: auto)",
    )
    p.add_argument(
        "--use-cifar",
        action="store_true",
        help="Use real CIFAR-10 data from data/ directory",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.device == "auto":
        dev = get_device()
    else:
        dev = torch.device(args.device)

    print(f"[constructive] Running on {dev}...")

    if args.use_cifar:
        from src.data import build_tensors
        print("[constructive] Loading real CIFAR-10 data...")
        train_X, train_y, test_X, test_y = build_tensors(
            data_dir="data/",
            corruption_mode="true",
            label_corruption_rate=0.0,
            seed=args.seed,
            device=dev,
        )
        X_flat = train_X.view(train_X.shape[0], -1)
        test_X_flat = test_X.view(test_X.shape[0], -1)

        n = min(args.n, len(X_flat))
        sub_X = X_flat[:n]
        sub_y = train_y[:n]

        res = fit_constructive_torch(sub_X, sub_y, seed=args.seed, device=dev)
        test_acc = evaluate_constructive(res, test_X_flat[:1000], test_y[:1000])
        print(f"[constructive] Generalization on clean test set: {test_acc:.2f}% (random chance is 10.0%)")
    else:
        torch.manual_seed(args.seed)
        X = torch.randn(args.n, args.d, device=dev, dtype=torch.float64)
        y = torch.randint(0, 10, (args.n,), device=dev)
        res = fit_constructive_torch(X, y, seed=args.seed, device=dev)

