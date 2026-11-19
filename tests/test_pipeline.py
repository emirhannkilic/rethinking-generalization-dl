"""
test_pipeline.py — Integration test for end-to-end train, evaluate, and plot pipeline.
"""

import pytest
import shutil
from pathlib import Path

from src.train import train_one_run
from src.evaluate import aggregate_runs
from src.plot import plot_learning_curves


def test_end_to_end_smoke_pipeline(tmp_path):
    """Run 1-epoch smoke test on CNN, aggregate results, and generate a plot."""
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = {
        "seed": 42,
        "epochs": 1,
        "lr": 0.01,
        "batch_size": 64,
        "model_type": "cnn",
        "input_mode": "true",
        "label_corruption": 0.0,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "data_dir": "data/",
        "results_dir": str(results_dir),
        "save_checkpoint": False,
        "smoke_test": True,
    }
    
    # 1. Train one run
    res = train_one_run(cfg)
    assert "run_id" in res
    assert len(res["train_acc"]) == 1
    assert (results_dir / f"{res['run_id']}.json").exists()
    
    # 2. Evaluate aggregation
    stats = aggregate_runs(results_dir, pattern="*.json")
    assert stats["n_runs"] == 1
    assert stats["final_train_acc_mean"] > 0.0
    
    # 3. Plot generation
    pdf_path = figures_dir / "test_curves.pdf"
    plot_learning_curves([res], output_path=pdf_path, title="Test Curves")
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
