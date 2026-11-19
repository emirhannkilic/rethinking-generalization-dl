"""
data.py — CIFAR-10 loading and label/input corruption utilities.

Implements the data pipeline described in Zhang et al. (2017):
  - True labels (baseline)
  - Partially/fully randomized labels
  - Corrupted input pixels (shuffled, random, Gaussian noise)

All corruptions are applied ONCE per run (seeded), not re-randomized each epoch.
Test labels are NEVER corrupted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Raw dataset helpers
# ---------------------------------------------------------------------------

def get_cifar10(
    data_dir: str | Path,
    train: bool = True,
    download: bool = True,
) -> torchvision.datasets.CIFAR10:
    """Return raw torchvision CIFAR-10 dataset (images as PIL, no transforms).

    Args:
        data_dir: Directory where the dataset is stored / will be downloaded.
        train:    True → training split; False → test split.
        download: If True, download the dataset if not already present.

    Returns:
        A :class:`torchvision.datasets.CIFAR10` instance.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Minimal transform: convert PIL to tensor in [0, 1].
    transform = T.ToTensor()

    return torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=train,
        download=download,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Label corruption
# ---------------------------------------------------------------------------

def apply_label_corruption(
    dataset: torchvision.datasets.CIFAR10,
    corruption_rate: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace a fraction of labels with uniform random classes.

    Corruption is applied ONCE (seeded) so it is deterministic for a given
    (corruption_rate, seed) pair.  The original labels are returned unchanged
    so accuracy on clean labels can still be computed.

    Args:
        dataset:         CIFAR-10 training split.
        corruption_rate: Fraction of labels to randomise, in [0.0, 1.0].
                         0.0 → no corruption; 1.0 → fully random labels.
        seed:            NumPy RNG seed for reproducibility.

    Returns:
        (corrupted_labels, original_labels) — both LongTensors of shape (N,).

    Raises:
        ValueError: If *corruption_rate* is outside [0, 1].
    """
    if not (0.0 <= corruption_rate <= 1.0):
        raise ValueError(
            f"corruption_rate must be in [0, 1], got {corruption_rate}"
        )

    if hasattr(dataset, "targets"):
        original_labels = torch.tensor(dataset.targets, dtype=torch.long)
    elif isinstance(dataset, torch.Tensor):
        original_labels = dataset.clone().long()
    else:
        original_labels = torch.tensor(dataset, dtype=torch.long)

    corrupted_labels = original_labels.clone()

    if corruption_rate == 0.0:
        return corrupted_labels, original_labels

    rng = np.random.RandomState(seed)
    n = len(original_labels)
    num_corrupt = int(round(n * corruption_rate))
    corrupt_idx = rng.choice(n, size=num_corrupt, replace=False)

    # Assign uniformly random classes (0–9) to the selected indices.
    random_labels = rng.randint(0, 10, size=num_corrupt)
    corrupted_labels[corrupt_idx] = torch.tensor(random_labels, dtype=torch.long)

    # Sanity check: class distribution should be roughly uniform.
    if corruption_rate == 1.0:
        counts = torch.bincount(corrupted_labels, minlength=10).float()
        expected = n / 10.0
        assert torch.all(torch.abs(counts - expected) < 0.20 * expected), (
            f"Random label distribution too skewed: {counts.tolist()}"
        )

    return corrupted_labels, original_labels


# ---------------------------------------------------------------------------
# Input corruption
# ---------------------------------------------------------------------------

def apply_input_corruption(
    images: torch.Tensor,
    mode: str,
    seed: int,
) -> torch.Tensor:
    """Apply pixel-level corruption to a batch of images.

    Args:
        images: Float tensor of shape (N, 3, 32, 32) in [0, 1].
        mode:   One of:
                  ``"true"``           — no corruption (identity).
                  ``"shuffled_pixels"``— apply the same fixed random pixel
                                         permutation to every image.
                  ``"random_pixels"``  — replace every pixel with i.i.d.
                                         Uniform[0, 1] noise (per image).
                  ``"gaussian_noise"`` — replace every pixel with
                                         N(0.5, 0.5²) noise clamped to [0, 1].
        seed:   RNG seed for reproducibility.

    Returns:
        Corrupted float tensor, same shape as *images*.

    Raises:
        ValueError: If *mode* is not recognised.
    """
    valid_modes = {"true", "shuffled_pixels", "random_pixels", "gaussian_noise"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown corruption mode '{mode}'. Choose from {valid_modes}")

    if mode == "true":
        return images.clone()

    rng = np.random.RandomState(seed)
    n, c, h, w = images.shape
    n_pixels = c * h * w  # 3*32*32 = 3072

    if mode == "shuffled_pixels":
        # One fixed permutation applied identically to all images.
        perm = torch.tensor(rng.permutation(n_pixels), dtype=torch.long)
        flat = images.view(n, n_pixels)
        corrupted = flat[:, perm].view(n, c, h, w)

    elif mode == "random_pixels":
        # Each image gets independent uniform noise.
        noise = torch.tensor(
            rng.uniform(0.0, 1.0, size=(n, c, h, w)),
            dtype=images.dtype,
        )
        corrupted = noise

    elif mode == "gaussian_noise":
        # Each pixel ~ N(0.5, 0.5²), clamped to [0, 1].
        noise = torch.tensor(
            rng.normal(0.5, 0.5, size=(n, c, h, w)),
            dtype=images.dtype,
        ).clamp(0.0, 1.0)
        corrupted = noise

    return corrupted


# ---------------------------------------------------------------------------
# Main tensor builder
# ---------------------------------------------------------------------------

def build_tensors(
    data_dir: str | Path,
    corruption_mode: str,
    label_corruption_rate: float,
    seed: int,
    device: torch.device,
    smoke_test: bool = False,
    n_train_smoke: int = 500,
    n_test_smoke: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 into device tensors and apply all corruptions.

    The full dataset is kept in GPU/MPS/CPU memory for fast per-epoch sampling.
    Images are normalised to [0, 1] (ToTensor already does this).

    Corruptions applied:
      1. Input corruption (before moving to device) — via *corruption_mode*.
      2. Label corruption — via *label_corruption_rate*.

    Test labels are NEVER corrupted.

    Args:
        data_dir:              Root directory for CIFAR-10 data.
        corruption_mode:       One of ``"true"``, ``"shuffled_pixels"``,
                               ``"random_pixels"``, ``"gaussian_noise"``.
        label_corruption_rate: Fraction in [0, 1] of training labels to corrupt.
        seed:                  Master RNG seed (passed to both corruption functions).
        device:                Target torch device.
        smoke_test:            If True, subsample to tiny sizes for quick checks.
        n_train_smoke:         Training samples in smoke-test mode.
        n_test_smoke:          Test samples in smoke-test mode.

    Returns:
        ``(train_X, train_y, test_X, test_y)`` where:
          - ``train_X``: (N_train, 3, 32, 32) float on *device*
          - ``train_y``: (N_train,) long on *device*  — CORRUPTED labels
          - ``test_X``:  (N_test, 3, 32, 32) float on *device*
          - ``test_y``:  (N_test,) long on *device*  — ALWAYS clean
    """
    # ---- Training split -------------------------------------------------- #
    train_ds = get_cifar10(data_dir, train=True, download=True)
    test_ds  = get_cifar10(data_dir, train=False, download=True)

    # Stack all images into tensors (each sample is already a (3,32,32) tensor
    # from the ToTensor transform applied in get_cifar10).
    train_images = torch.stack([train_ds[i][0] for i in range(len(train_ds))])  # (50000,3,32,32)
    test_images  = torch.stack([test_ds[i][0]  for i in range(len(test_ds))])   # (10000,3,32,32)

    # Subsample for smoke-test runs.
    if smoke_test:
        train_images = train_images[:n_train_smoke]
        # Temporarily shrink dataset for label corruption.
        train_ds.targets = train_ds.targets[:n_train_smoke]
        if hasattr(train_ds, "data"):
            train_ds.data = train_ds.data[:n_train_smoke]
        test_images  = test_images[:n_test_smoke]
        test_ds.targets  = test_ds.targets[:n_test_smoke]
        if hasattr(test_ds, "data"):
            test_ds.data = test_ds.data[:n_test_smoke]

    # ---- Apply input corruption ------------------------------------------ #
    train_images = apply_input_corruption(train_images, mode=corruption_mode, seed=seed)

    # ---- Apply label corruption ------------------------------------------ #
    corrupted_labels, _original_labels = apply_label_corruption(
        train_ds, corruption_rate=label_corruption_rate, seed=seed
    )

    # ---- Move to device -------------------------------------------------- #
    train_X = train_images.to(device)
    train_y = corrupted_labels.to(device)
    test_X  = test_images.to(device)
    test_y  = torch.tensor(test_ds.targets, dtype=torch.long).to(device)

    return train_X, train_y, test_X, test_y
