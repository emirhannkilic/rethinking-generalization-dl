"""
train.py — Single training run for the rethinking-generalization study.

Implements the main training loop following Zhang et al. (2017):
  - SGD with momentum=0.9 (no augmentation, weight_decay=0 by default)
  - In-memory data tensors on GPU/CPU (no DataLoader shuffle overhead)
  - Per-epoch tracking of train loss/accuracy (on CORRUPTED labels)
    and test loss/accuracy (always clean labels)
  - JSON result persistence and optional checkpoint saving

Usage as a module:
    from src.train import train_one_run
    result = train_one_run(config)

Usage from command line:
    python -m src.train --mode true_labels --epochs 100 --model cnn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_device, set_seed, save_results
from src.data import build_tensors
from src.models import SmallCNN, MLP, count_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_id(config: Dict[str, Any]) -> str:
    """Create a short, human-readable + unique run identifier from *config*.

    Format: ``{model}_{input_mode}_lc{label_corruption}_s{seed}``
    followed by a 6-char hex hash for full uniqueness.
    """
    base = (
        f"{config['model_type']}"
        f"_{config['input_mode']}"
        f"_lc{config['label_corruption']:.2f}"
        f"_s{config['seed']}"
    )
    # Append a short hash of the full config for safety.
    cfg_str = json.dumps(config, sort_keys=True)
    short_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:6]
    return f"{base}_{short_hash}"


def _build_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Instantiate model from config and move to *device*."""
    mtype = config.get("model_type", "cnn").lower()
    dropout = config.get("dropout", 0.0)

    if mtype == "cnn":
        model = SmallCNN(dropout=dropout)
    elif mtype == "mlp":
        model = MLP(dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type '{mtype}'. Choose 'cnn' or 'mlp'.")

    return model.to(device)


def _iter_batches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
):
    """Yield (x_batch, y_batch) mini-batches from in-memory tensors.

    Args:
        X:          Input tensor (N, …).
        y:          Label tensor (N,).
        batch_size: Mini-batch size.
        shuffle:    Whether to shuffle indices each call.

    Yields:
        Tuple ``(x_batch, y_batch)``.
    """
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for start in range(0, n, batch_size):
        batch_idx = idx[start: start + batch_size]
        yield X[batch_idx], y[batch_idx]


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 512,
) -> tuple[float, float]:
    """Compute loss and accuracy over an in-memory dataset without gradients.

    Args:
        model:      Eval-mode model.
        X:          Input tensor (N, 3, 32, 32).
        y:          Label tensor (N,) — long.
        batch_size: Evaluation batch size.

    Returns:
        ``(mean_loss, accuracy)`` as Python floats.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = X.shape[0]

    for x_b, y_b in _iter_batches(X, y, batch_size, shuffle=False):
        logits = model(x_b)
        loss = F.cross_entropy(logits, y_b, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == y_b).sum().item()

    return total_loss / n, total_correct / n


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_one_run(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single training run and return per-epoch metrics.

    Args:
        config: Dictionary with the following keys:

            ============ =====================================================
            Key          Description
            ============ =====================================================
            seed         int — master RNG seed
            epochs       int — number of training epochs
            lr           float — initial learning rate (default 0.01)
            batch_size   int — mini-batch size (default 128)
            model_type   str — "cnn" or "mlp"
            input_mode   str — "true" / "shuffled_pixels" / "random_pixels"
                               / "gaussian_noise"
            label_corruption float — fraction in [0,1] of labels to corrupt
            weight_decay float — L2 penalty (0.0 for E1/E2 experiments)
            dropout      float — dropout probability (0.0 for E1/E2)
            data_dir     str | Path — root for CIFAR-10 data
            results_dir  str | Path — where to write JSON / .pt files
            save_checkpoint bool — whether to save model weights
            smoke_test   bool — if True, use tiny subset for debugging
            ============ =====================================================

    Returns:
        Dict containing:
          - ``config``: the full config dict
          - ``run_id``: string identifier
          - ``n_parameters``: int
          - ``epochs``: list of epoch indices (1-based)
          - ``train_loss``, ``train_acc``: lists of per-epoch floats
          - ``test_loss``, ``test_acc``: lists of per-epoch floats
          - ``wall_seconds``: total wall-clock time
    """
    # --- Defaults ---------------------------------------------------------- #
    cfg = {
        "lr": 0.01,
        "batch_size": 128,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "save_checkpoint": False,
        "smoke_test": False,
    }
    cfg.update(config)

    seed       = cfg["seed"]
    epochs     = cfg["epochs"]
    lr         = cfg["lr"]
    batch_size = cfg["batch_size"]
    wd         = cfg["weight_decay"]
    data_dir   = Path(cfg["data_dir"])
    results_dir = Path(cfg["results_dir"])
    smoke_test = cfg.get("smoke_test", False)

    # --- Reproducibility --------------------------------------------------- #
    set_seed(seed)
    device = get_device()

    # --- Data -------------------------------------------------------------- #
    print(f"[train] Loading data  (input_mode={cfg['input_mode']}, "
          f"label_corruption={cfg['label_corruption']:.2f}, seed={seed}) …")

    train_X, train_y, test_X, test_y = build_tensors(
        data_dir=data_dir,
        corruption_mode=cfg["input_mode"],
        label_corruption_rate=cfg["label_corruption"],
        seed=seed,
        device=device,
        smoke_test=smoke_test,
    )
    print(f"[train] train={train_X.shape[0]}, test={test_X.shape[0]}")

    # --- Model ------------------------------------------------------------- #
    model = _build_model(cfg, device)
    n_params = count_parameters(model)
    print(f"[train] {cfg['model_type'].upper()} — {n_params:,} parameters")

    # --- Optimiser --------------------------------------------------------- #
    # Always use disable_regularization() param groups, then override weight_decay.
    param_groups = model.disable_regularization()
    param_groups[0]["weight_decay"] = wd

    optimizer = torch.optim.SGD(
        param_groups,
        lr=lr,
        momentum=0.9,
    )
    criterion = nn.CrossEntropyLoss()

    # --- Run ID & result paths -------------------------------------------- #
    run_id = _make_run_id(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{run_id}.json"
    ckpt_path   = results_dir / f"{run_id}.pt"

    # --- Training loop ----------------------------------------------------- #
    history: Dict[str, List] = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0

        for x_b, y_b in _iter_batches(train_X, train_y, batch_size, shuffle=True):
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            bs = x_b.shape[0]
            epoch_loss    += loss.item() * bs
            epoch_correct += (logits.detach().argmax(dim=1) == y_b).sum().item()
            epoch_n       += bs

        train_loss = epoch_loss    / epoch_n
        train_acc  = epoch_correct / epoch_n

        # Evaluation on test set (always clean labels).
        test_loss, test_acc = _evaluate(model, test_X, test_y, batch_size=512)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}"
            )

    wall_seconds = time.time() - t0

    # --- Persist ----------------------------------------------------------- #
    result = {
        "run_id": run_id,
        "config": cfg,
        "n_parameters": n_params,
        "wall_seconds": wall_seconds,
        **history,
    }
    save_results(result_path, result)
    print(f"[train] Results → {result_path}")

    if cfg["save_checkpoint"]:
        torch.save(model.state_dict(), ckpt_path)
        print(f"[train] Checkpoint → {ckpt_path}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a network under various label/input corruptions."
    )
    p.add_argument(
        "--mode",
        choices=["randomization", "corruption_sweep", "true_labels"],
        default="true_labels",
        help="Experiment mode (default: true_labels)",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        metavar="SEED",
        help="One or more integer seeds (default: 42 123 456)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)",
    )
    p.add_argument(
        "--corruption",
        type=float,
        default=0.0,
        help="Label corruption rate in [0,1] (default: 0.0)",
    )
    p.add_argument(
        "--input-mode",
        dest="input_mode",
        choices=["true", "shuffled_pixels", "random_pixels", "gaussian_noise"],
        default="true",
        help="Input corruption mode (default: true)",
    )
    p.add_argument(
        "--model",
        choices=["cnn", "mlp"],
        default="cnn",
        help="Model architecture (default: cnn)",
    )
    p.add_argument(
        "--smoke-test",
        dest="smoke_test",
        action="store_true",
        help="Use 500 train / 100 test samples and 1 epoch for debugging",
    )
    p.add_argument(
        "--data-dir",
        dest="data_dir",
        default="data/",
        help="Directory for CIFAR-10 data (default: data/)",
    )
    p.add_argument(
        "--results-dir",
        dest="results_dir",
        default="results/",
        help="Directory for result JSON files (default: results/)",
    )
    return p.parse_args()


def _run_mode(args: argparse.Namespace) -> None:
    """Dispatch experiment runs based on --mode."""
    base_config = {
        "epochs":          1 if args.smoke_test else args.epochs,
        "lr":              0.01,
        "batch_size":      128,
        "model_type":      args.model,
        "input_mode":      args.input_mode,
        "label_corruption": args.corruption,
        "weight_decay":    0.0,
        "dropout":         0.0,
        "data_dir":        args.data_dir,
        "results_dir":     args.results_dir,
        "save_checkpoint": False,
        "smoke_test":      args.smoke_test,
    }

    if args.mode == "true_labels":
        # Baseline: no corruption, repeat over seeds.
        for seed in args.seeds:
            cfg = {**base_config, "seed": seed, "label_corruption": 0.0, "input_mode": "true"}
            train_one_run(cfg)

    elif args.mode == "randomization":
        # E1: fully random labels for different input corruption modes.
        for mode in ["true", "shuffled_pixels", "random_pixels", "gaussian_noise"]:
            for seed in args.seeds:
                cfg = {**base_config, "seed": seed, "label_corruption": 1.0, "input_mode": mode}
                train_one_run(cfg)

    elif args.mode == "corruption_sweep":
        # E2: vary label corruption rate.
        rates = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        for rate in rates:
            for seed in args.seeds:
                cfg = {**base_config, "seed": seed, "label_corruption": rate, "input_mode": "true"}
                train_one_run(cfg)
    else:
        raise ValueError(f"Unknown mode '{args.mode}'")


if __name__ == "__main__":
    _run_mode(_parse_args())
