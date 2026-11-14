"""
plot.py — Figure generation for the rethinking-generalization study.

Produces publication-quality plots reproducing key figures from
Zhang et al. (2017):
  - Learning curves (train / test accuracy vs epoch) grouped by corruption type,
    with ± 1 std shading across seeds.
  - Corruption sweep: test error vs label corruption rate.

All figures are saved as vector PDFs.

Usage:
    python -m src.plot --results-dir results/ --output-dir figures/
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils import load_results


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FONT_SIZE   = 12
FIG_SIZE    = (10, 6)
LINE_STYLES = ["-", "--", "-.", ":"]
COLORS      = plt.cm.tab10.colors  # type: ignore[attr-defined]

# Human-readable labels for input corruption modes.
MODE_LABELS: Dict[str, str] = {
    "true":           "True labels",
    "shuffled_pixels": "Shuffled pixels",
    "random_pixels":  "Random pixels",
    "gaussian_noise": "Gaussian noise",
}


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    runs_data: List[Dict[str, Any]],
    output_path: str | Path,
    title: str = "Learning Curves",
    shaded_std: bool = True,
    group_by: str = "input_mode",
) -> None:
    """Plot train and test accuracy curves grouped by a config key.

    When multiple runs share the same group key (e.g. same ``input_mode`` but
    different seeds), their curves are averaged and ± 1 std is drawn as a
    shaded band.

    Args:
        runs_data:   List of result dicts (each from :func:`train_one_run`).
        output_path: Destination PDF path.
        title:       Figure title.
        shaded_std:  Draw ± 1 std shading across seeds when True.
        group_by:    Config key to group by (``"input_mode"`` or
                     ``"label_corruption"``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group runs by their config key value.
    groups: Dict[Any, List[Dict]] = collections.defaultdict(list)
    for run in runs_data:
        key = run.get("config", {}).get(group_by, run.get(group_by, "unknown"))
        groups[key].append(run)

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=False)
    fig.suptitle(title, fontsize=FONT_SIZE + 2)

    for ax in axes:
        ax.tick_params(labelsize=FONT_SIZE - 1)
        ax.set_xlabel("Epoch", fontsize=FONT_SIZE)

    axes[0].set_ylabel("Training Accuracy", fontsize=FONT_SIZE)
    axes[1].set_ylabel("Test Accuracy",     fontsize=FONT_SIZE)

    legend_handles = []

    for idx, (key, run_list) in enumerate(sorted(groups.items(), key=lambda x: str(x[0]))):
        color = COLORS[idx % len(COLORS)]
        ls    = LINE_STYLES[idx % len(LINE_STYLES)]

        # Determine the common epoch axis (use the shortest run to be safe).
        min_len = min(len(r["train_acc"]) for r in run_list)
        epochs_arr = np.array(run_list[0]["epochs"][:min_len])

        train_matrix = np.array([r["train_acc"][:min_len] for r in run_list])  # (seeds, E)
        test_matrix  = np.array([r["test_acc"][:min_len]  for r in run_list])

        train_mean = train_matrix.mean(axis=0)
        test_mean  = test_matrix.mean(axis=0)

        # Readable group label.
        if group_by == "input_mode":
            label = MODE_LABELS.get(str(key), str(key))
        elif group_by == "label_corruption":
            label = f"Corruption {float(key):.0%}"
        else:
            label = str(key)

        for ax, mean_arr, matrix in [
            (axes[0], train_mean, train_matrix),
            (axes[1], test_mean,  test_matrix),
        ]:
            ax.plot(epochs_arr, mean_arr, color=color, linestyle=ls, linewidth=2.0)
            if shaded_std and len(run_list) > 1:
                std_arr = matrix.std(axis=0)
                ax.fill_between(
                    epochs_arr,
                    mean_arr - std_arr,
                    mean_arr + std_arr,
                    color=color,
                    alpha=0.15,
                )

        legend_handles.append(
            mpatches.Patch(color=color, label=label)
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        fontsize=FONT_SIZE - 1,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved learning curves → {output_path}")


# ---------------------------------------------------------------------------
# Corruption sweep
# ---------------------------------------------------------------------------

def plot_corruption_sweep(
    sweep_data: List[Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Plot test error (1 − accuracy) vs label corruption rate.

    *sweep_data* should contain one dict per (corruption_rate × seed)
    combination.  Runs are grouped by corruption rate and averaged across
    seeds.

    Args:
        sweep_data:  List of result dicts from a corruption sweep experiment.
        output_path: Destination PDF path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect final test accuracy per corruption rate.
    rate_to_errs: Dict[float, List[float]] = collections.defaultdict(list)
    for run in sweep_data:
        rate = float(run.get("config", {}).get("label_corruption", 0.0))
        final_test_acc = run["test_acc"][-1] if run["test_acc"] else float("nan")
        rate_to_errs[rate].append(1.0 - final_test_acc)

    rates = sorted(rate_to_errs.keys())
    means = [np.mean(rate_to_errs[r]) for r in rates]
    stds  = [np.std(rate_to_errs[r])  for r in rates]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.errorbar(
        rates,
        means,
        yerr=stds,
        fmt="o-",
        color=COLORS[0],
        linewidth=2.0,
        capsize=4,
        markersize=6,
    )
    ax.set_xlabel("Label Corruption Rate", fontsize=FONT_SIZE)
    ax.set_ylabel("Test Error",            fontsize=FONT_SIZE)
    ax.set_title("Test Error vs Label Corruption Rate", fontsize=FONT_SIZE + 2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved corruption sweep → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate plots from experiment result JSON files."
    )
    p.add_argument(
        "--results-dir",
        dest="results_dir",
        default="results/",
        help="Directory containing result JSON files (default: results/)",
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        default="figures/",
        help="Directory to write PDF figures (default: figures/)",
    )
    p.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for result files (default: '*.json')",
    )
    p.add_argument(
        "--group-by",
        dest="group_by",
        default="input_mode",
        choices=["input_mode", "label_corruption"],
        help="Config key to group curves by (default: input_mode)",
    )
    p.add_argument(
        "--title",
        default="Learning Curves",
        help="Figure title for learning curves plot",
    )
    p.add_argument(
        "--no-shading",
        dest="shaded_std",
        action="store_false",
        help="Disable ± std shading",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)

    files = sorted(results_dir.glob(args.pattern))
    if not files:
        print(f"[plot] No files matching '{args.pattern}' in '{results_dir}'")
        raise SystemExit(1)

    runs_data = [load_results(fp) for fp in files]
    print(f"[plot] Loaded {len(runs_data)} result file(s).")

    # Learning curves.
    lc_path = output_dir / "learning_curves.pdf"
    plot_learning_curves(
        runs_data,
        output_path=lc_path,
        title=args.title,
        shaded_std=args.shaded_std,
        group_by=args.group_by,
    )

    # Corruption sweep (only makes sense when group_by == "label_corruption").
    if args.group_by == "label_corruption":
        sweep_path = output_dir / "corruption_sweep.pdf"
        plot_corruption_sweep(runs_data, output_path=sweep_path)
