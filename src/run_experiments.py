"""
Run all paper experiments (E1 randomization + E2 corruption sweep).

Usage:
    python -m src.run_experiments [--smoke-test] [--epochs N] [--seeds S S S]

E1: Randomization tests — 5 input modes × N seeds
E2: Corruption sweep    — 6 corruption rates × N seeds
Then generates PDF figures and prints a summary table.
"""

import argparse
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, unit="it", ncols=80, desc=""):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if iterable else 0)
            self.unit = unit
            self.desc = desc
            self.n = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            print()
        def update(self, n=1):
            self.n += n
            prefix = f"{self.desc}: " if self.desc else ""
            print(f"\r{prefix}{self.n}/{self.total} {self.unit}s completed", end="", flush=True)
        def set_postfix_str(self, s):
            pass
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

E1_INPUT_MODES = [
    "true",             # clean labels, clean pixels (baseline)
    "random_labels",    # 100 % label corruption  (handled via label_corruption=1.0)
    "shuffled_pixels",
    "random_pixels",
    "gaussian_noise",
]

E2_CORRUPTION_RATES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _base_config(args: argparse.Namespace) -> dict:
    """Return a config dict with defaults shared across all runs."""
    return dict(
        epochs=args.epochs,
        lr=0.01,
        batch_size=128,
        model_type="cnn",
        weight_decay=0.0,
        dropout=0.0,
        data_dir="data/",
        results_dir="results/",
        save_checkpoint=False,
        smoke_test=args.smoke_test,
    )


def build_e1_configs(args: argparse.Namespace) -> list[dict]:
    """E1: 5 input modes × len(seeds) runs."""
    configs = []
    for mode in E1_INPUT_MODES:
        for seed in args.seeds:
            cfg = _base_config(args)
            if mode == "random_labels":
                cfg["input_mode"] = "true"
                cfg["label_corruption"] = 1.0
            else:
                cfg["input_mode"] = mode
                cfg["label_corruption"] = 0.0
            cfg["seed"] = seed
            cfg["experiment"] = "E1"
            cfg["experiment_label"] = mode
            configs.append(cfg)
    return configs


def build_e2_configs(args: argparse.Namespace) -> list[dict]:
    """E2: 6 corruption rates × len(seeds) runs."""
    configs = []
    for rate in E2_CORRUPTION_RATES:
        for seed in args.seeds:
            cfg = _base_config(args)
            cfg["input_mode"] = "true"
            cfg["label_corruption"] = rate
            cfg["seed"] = seed
            cfg["experiment"] = "E2"
            cfg["experiment_label"] = f"corruption_{rate:.1f}"
            configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict]) -> None:
    """Print a formatted summary table of final metrics."""
    print("\n" + "=" * 72)
    print(f"{'Experiment':<12} {'Label':>22} {'Seed':>6} {'TrainAcc':>10} {'TestAcc':>10}")
    print("-" * 72)
    for r in all_results:
        cfg = r.get("config", {})
        train_hist = r.get("train_acc", [])
        test_hist  = r.get("test_acc",  [])
        final_train = train_hist[-1] * 100 if train_hist else float("nan")
        final_test  = test_hist[-1]  * 100 if test_hist  else float("nan")
        print(
            f"{cfg.get('experiment', '?'):<12}"
            f" {cfg.get('experiment_label', '?'):>22}"
            f" {cfg.get('seed', '?'):>6}"
            f" {final_train:>9.1f}%"
            f" {final_test:>9.1f}%"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all CMP4509 reproduction experiments (E1 + E2)."
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick sanity check: 500 train / 100 test samples, 1 epoch.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="Number of training epochs per run (default: 100).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        metavar="S",
        help="Random seeds to use (default: 42 123 456).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Import here so the module can be imported without torch installed
    try:
        from src.train import train_one_run
    except ImportError as exc:
        sys.exit(
            f"[run_experiments] Cannot import src.train: {exc}\n"
            "Make sure you have activated the virtual environment and "
            "installed all dependencies (see README.md)."
        )

    all_configs = build_e1_configs(args) + build_e2_configs(args)
    total = len(all_configs)

    smoke_tag = " [SMOKE TEST]" if args.smoke_test else ""
    print(
        f"\n[run_experiments] Starting {total} runs"
        f" ({len(args.seeds)} seeds){smoke_tag}\n"
        f"  E1: {len(E1_INPUT_MODES)} modes × {len(args.seeds)} seeds"
        f" = {len(E1_INPUT_MODES) * len(args.seeds)} runs\n"
        f"  E2: {len(E2_CORRUPTION_RATES)} rates × {len(args.seeds)} seeds"
        f" = {len(E2_CORRUPTION_RATES) * len(args.seeds)} runs\n"
    )

    all_results: list[dict] = []
    t0 = time.time()

    with tqdm(total=total, unit="run", ncols=80) as pbar:
        for cfg in all_configs:
            label = f"{cfg['experiment']} | {cfg['experiment_label']} | seed={cfg['seed']}"
            pbar.set_description(label[:40])
            result = train_one_run(cfg)
            # Attach experiment metadata to the result for the summary table
            result.setdefault("config", {}).update(
                experiment=cfg["experiment"],
                experiment_label=cfg["experiment_label"],
                seed=cfg["seed"],
            )
            all_results.append(result)
            pbar.update(1)

    elapsed = time.time() - t0
    print(f"\n[run_experiments] All {total} runs finished in {elapsed:.1f}s.")

    # --- Generate figures ---
    try:
        from src.plot import generate_all_figures
        print("[run_experiments] Generating figures …")
        generate_all_figures(results_dir=Path("results/"), output_dir=Path("figures/"))
        print("[run_experiments] Figures saved to figures/")
    except Exception as exc:
        print(f"[run_experiments] Figure generation skipped: {exc}")

    # --- Print summary ---
    print_summary(all_results)


if __name__ == "__main__":
    main()
