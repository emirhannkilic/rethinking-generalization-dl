"""
evaluate.py — Aggregation and summary utilities for experiment results.

Reads per-run JSON files produced by train.py, computes mean ± std across
seeds for key metrics, and prints a formatted summary table.

Usage:
    python -m src.evaluate --results-dir results/ --pattern "cnn_true*.json"
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import load_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(
    results_dir: str | Path,
    pattern: str = "*.json",
) -> Dict[str, Any]:
    """Aggregate metrics across multiple seeded runs matching *pattern*.

    For each matched JSON file (one per run), extracts:
      - ``final_train_acc``: training accuracy at the last epoch.
      - ``final_test_acc``: test accuracy at the last epoch.
      - ``convergence_epoch``: first epoch where ``train_acc > 0.95``
        (``None`` if never reached).

    Then computes mean and std across all matched runs.

    Args:
        results_dir: Directory containing result JSON files.
        pattern:     Glob pattern relative to *results_dir*.

    Returns:
        Dict with keys:
          ``n_runs``, ``pattern``,
          ``final_train_acc_mean``, ``final_train_acc_std``,
          ``final_test_acc_mean``,  ``final_test_acc_std``,
          ``convergence_epoch_mean``, ``convergence_epoch_std``,
          ``runs``: list of per-file dicts.

    Raises:
        FileNotFoundError: If no files match *pattern* in *results_dir*.
    """
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in '{results_dir}'"
        )

    per_run: List[Dict[str, Any]] = []

    for fp in files:
        data = load_results(fp)

        train_accs: List[float] = data.get("train_acc", [])
        test_accs:  List[float] = data.get("test_acc",  [])

        final_train = train_accs[-1] if train_accs else float("nan")
        final_test  = test_accs[-1]  if test_accs  else float("nan")

        # Convergence: first epoch index (1-based) where train_acc > 95 %.
        epochs = data.get("epochs", list(range(1, len(train_accs) + 1)))
        conv_epoch: Optional[int] = None
        for ep, acc in zip(epochs, train_accs):
            if acc > 0.95:
                conv_epoch = int(ep)
                break

        per_run.append(
            {
                "file": str(fp.name),
                "run_id": data.get("run_id", fp.stem),
                "config": data.get("config", {}),
                "final_train_acc": final_train,
                "final_test_acc": final_test,
                "convergence_epoch": conv_epoch,
            }
        )

    def _mean(vals: List[Optional[float]]) -> float:
        clean = [v for v in vals if v is not None and not (isinstance(v, float) and v != v)]
        return statistics.mean(clean) if clean else float("nan")

    def _std(vals: List[Optional[float]]) -> float:
        clean = [v for v in vals if v is not None and not (isinstance(v, float) and v != v)]
        return statistics.stdev(clean) if len(clean) > 1 else 0.0

    final_trains = [r["final_train_acc"]   for r in per_run]
    final_tests  = [r["final_test_acc"]    for r in per_run]
    conv_epochs  = [r["convergence_epoch"] for r in per_run]

    return {
        "n_runs":   len(per_run),
        "pattern":  pattern,
        "results_dir": str(results_dir),
        "final_train_acc_mean": _mean(final_trains),
        "final_train_acc_std":  _std(final_trains),
        "final_test_acc_mean":  _mean(final_tests),
        "final_test_acc_std":   _std(final_tests),
        "convergence_epoch_mean": _mean(conv_epochs),  # type: ignore[arg-type]
        "convergence_epoch_std":  _std(conv_epochs),   # type: ignore[arg-type]
        "runs": per_run,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_summary_table(stats: Dict[str, Any]) -> None:
    """Print a human-readable summary table for aggregated stats.

    Args:
        stats: Dict returned by :func:`aggregate_runs`.
    """
    sep = "─" * 72

    print(sep)
    print(f"  Results dir : {stats.get('results_dir', '?')}")
    print(f"  Pattern     : {stats.get('pattern', '?')}")
    print(f"  Runs matched: {stats.get('n_runs', 0)}")
    print(sep)
    print(
        f"  {'Metric':<30}  {'Mean':>10}  {'Std':>10}"
    )
    print(sep)
    print(
        f"  {'Final train accuracy':<30}  "
        f"{stats['final_train_acc_mean']:>9.4f}  "
        f"{stats['final_train_acc_std']:>9.4f}"
    )
    print(
        f"  {'Final test accuracy':<30}  "
        f"{stats['final_test_acc_mean']:>9.4f}  "
        f"{stats['final_test_acc_std']:>9.4f}"
    )
    ce_mean = stats["convergence_epoch_mean"]
    ce_std  = stats["convergence_epoch_std"]
    ce_str  = f"{ce_mean:>9.1f}" if ce_mean == ce_mean else "       N/A"  # nan check
    print(
        f"  {'Convergence epoch (>95% train)':<30}  "
        f"{ce_str}  "
        f"{ce_std:>9.1f}"
    )
    print(sep)
    print()

    # Per-run breakdown.
    print(f"  {'File':<40}  {'TrainAcc':>9}  {'TestAcc':>9}  {'ConvEp':>7}")
    print("  " + "─" * 68)
    for r in stats["runs"]:
        ce = r["convergence_epoch"]
        ce_disp = f"{ce:>7d}" if ce is not None else "    N/A"
        print(
            f"  {r['file']:<40}  "
            f"{r['final_train_acc']:>9.4f}  "
            f"{r['final_test_acc']:>9.4f}  "
            f"{ce_disp}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate and display experiment results."
    )
    p.add_argument(
        "--results-dir",
        dest="results_dir",
        default="results/",
        help="Directory containing result JSON files (default: results/)",
    )
    p.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for result files (default: '*.json')",
    )
    p.add_argument(
        "--eval-only",
        dest="eval_only",
        action="store_true",
        help="If set, only print evaluation and exit (no additional side effects)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        stats = aggregate_runs(args.results_dir, args.pattern)
        print_summary_table(stats)
    except FileNotFoundError as exc:
        print(f"[evaluate] ERROR: {exc}")
        raise SystemExit(1)
