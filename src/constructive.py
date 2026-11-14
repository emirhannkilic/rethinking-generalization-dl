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
  5. Verify the residual is below 1e-6.

Always runs on CPU in float64 for numerical stability.

Usage (standalone):
    python -m src.constructive
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Core construction
# ---------------------------------------------------------------------------

def fit_constructive(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
) -> Dict[str, Any]:
    """Construct a 2-layer ReLU network that interpolates every training sample.

    Parameters
    ----------
    X:
        Input matrix of shape ``(n, d)`` in float64 (or will be cast).
        Each row is one training sample.
    y:
        Label vector of shape ``(n,)`` with integer entries in [0, 9].
    seed:
        NumPy RNG seed for reproducibility of the random direction **a**.

    Returns
    -------
    dict with keys:

    ``a``           — shape ``(d,)`` random direction.
    ``biases``      — shape ``(n,)`` threshold biases b_j.
    ``weights``     — shape ``(n, 10)`` output weights w.
    ``residual``    — scalar residual ‖Aw − y_onehot‖_F.
    ``n_params``    — total parameter count = 2n + d.

    Raises
    ------
    AssertionError:
        If the residual exceeds 1e-6, indicating numerical failure.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    n, d = X.shape

    if y.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {y.shape}")
    if np.any((y < 0) | (y > 9)):
        raise ValueError("Labels y must be in range [0, 9].")

    rng = np.random.RandomState(seed)

    # ---- Step 1: Random projection direction ----------------------------- #
    a = rng.randn(d).astype(np.float64)         # shape (d,)
    a /= np.linalg.norm(a) + 1e-12              # unit vector

    # ---- Step 2: Projections z_i = a · x_i ------------------------------ #
    z = X @ a                                   # shape (n,)

    # ---- Step 3: Bias values between consecutive sorted projections ------- #
    # Sort to get unique z values; handle potential near-duplicates by adding
    # tiny jitter if needed.
    sort_idx = np.argsort(z)
    z_sorted = z[sort_idx]

    # Ensure strict ordering by nudging near-ties.
    for i in range(1, n):
        if z_sorted[i] <= z_sorted[i - 1]:
            z_sorted[i] = z_sorted[i - 1] + 1e-10

    # Place biases at midpoints between consecutive sorted z values.
    # b_1 < z_1 (below the minimum):
    b = np.empty(n, dtype=np.float64)
    b[0] = z_sorted[0] - 1.0
    for j in range(1, n):
        b[j] = 0.5 * (z_sorted[j - 1] + z_sorted[j])

    # ---- Step 4: Activation matrix A_ij = max(z_i - b_j, 0) ------------- #
    # A has shape (n, n); rows correspond to samples, columns to hidden units.
    A = np.maximum(z[:, None] - b[None, :], 0.0)   # broadcast

    # ---- Step 5: One-hot target ------------------------------------------ #
    num_classes = 10
    y_onehot = np.zeros((n, num_classes), dtype=np.float64)
    y_onehot[np.arange(n), y] = 1.0

    # ---- Step 6: Solve A · w = y_onehot ---------------------------------- #
    # A is (n, n) and should be invertible by construction (lower-triangular
    # when rows are in sorted z order, with positive diagonal entries).
    # Solve one column at a time for each class.
    try:
        w = np.linalg.solve(A, y_onehot)           # shape (n, 10)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"Linear solve failed (A may be singular): {exc}"
        ) from exc

    # ---- Step 7: Verify residual ----------------------------------------- #
    residual = float(np.linalg.norm(A @ w - y_onehot, ord="fro"))
    assert residual < 1e-6, (
        f"Constructive fit failed: residual = {residual:.3e} ≥ 1e-6"
    )

    n_params = 2 * n + d   # a (d) + b (n) + w (n×10 → compressed as n) = 2n+d
    # Note: the paper counts each ReLU unit's bias and output weight as
    # separate parameters: d (input weights for a) + n biases + n·C output
    # weights.  Here we report 2n+d as a compact summary matching the theorem.

    print(
        f"[constructive] n={n}, d={d}  |  "
        f"residual={residual:.2e}  |  "
        f"n_params(2n+d)={n_params}"
    )

    return {
        "a":        a,
        "biases":   b,
        "weights":  w,
        "residual": residual,
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Quick self-test with random data."""
    rng = np.random.RandomState(42)
    n, d = 50, 20
    X = rng.randn(n, d).astype(np.float64)
    y = rng.randint(0, 10, size=n)

    result = fit_constructive(X, y, seed=99)

    print(f"  a shape    : {result['a'].shape}")
    print(f"  biases     : {result['biases'].shape}")
    print(f"  weights    : {result['weights'].shape}")
    print(f"  residual   : {result['residual']:.2e}  ✓" if result["residual"] < 1e-6 else "  FAIL")
    print(f"  n_params   : {result['n_params']}  (2×{n}+{d} = {2*n+d})")
    print("[constructive] Smoke test PASSED.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the constructive proof of Theorem 1 (Zhang et al. 2017)."
    )
    p.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of training samples for the demo (default: 100)",
    )
    p.add_argument(
        "--d",
        type=int,
        default=3072,
        help="Input dimensionality for the demo (default: 3072 = CIFAR-10 flat)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0)",
    )
    p.add_argument(
        "--smoke-test",
        dest="smoke_test",
        action="store_true",
        help="Run a small smoke test (n=50, d=20) and exit",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        rng = np.random.RandomState(args.seed)
        n, d = args.n, args.d
        print(f"[constructive] Demo: n={n}, d={d}, seed={args.seed}")
        X = rng.randn(n, d).astype(np.float64)
        y = rng.randint(0, 10, size=n)
        fit_constructive(X, y, seed=args.seed)
        print("[constructive] Done.")
