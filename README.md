# Rethinking Generalization — CMP4509 Reproduction

Reproduction of **Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization"** (ICLR 2017, [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)) for CMP4509 Introduction to Deep Learning, Fall 2026.

**Paper claim:** Standard deep networks can memorize completely random labels on CIFAR-10 to near-zero training error, proving that model capacity alone cannot explain generalization.

## Repository
[https://github.com/emirhannkilic/rethinking-generalization-dl](https://github.com/emirhannkilic/rethinking-generalization-dl)

## Project Structure
```
rethinking-generalization-dl/
├── src/
│   ├── utils.py          # Device selection, seed setting, I/O helpers
│   ├── data.py           # CIFAR-10 loading + label/input corruptions
│   ├── models.py         # SmallCNN and MLP architectures
│   ├── train.py          # Single training run + CLI entry point
│   ├── evaluate.py       # Aggregate results across seeds
│   ├── plot.py           # Generate PDF figures
│   └── constructive.py   # Theorem 1 constructive proof (CPU/float64)
├── report/
│   ├── main.tex          # LaTeX report (course template)
│   └── references.bib    # Bibliography
├── figures/              # Generated PDF plots (git-ignored)
├── results/              # JSON/CSV metrics per run (checkpoints git-ignored)
├── data/                 # CIFAR-10 download (git-ignored)
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites
- Python 3.10 or later
- Git

### 1. Create virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch

Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your platform.

**Windows with NVIDIA GPU (CUDA 12.x):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**macOS (Apple Silicon — MPS backend):**
```bash
pip install torch torchvision
```

**CPU-only (any platform):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **macOS note:** If you encounter missing MPS operations, set:
> ```bash
> export PYTORCH_ENABLE_MPS_FALLBACK=1
> ```

### 3. Install remaining dependencies
```bash
pip install -r requirements.txt
```

## Smoke Test (1 epoch, ~500 samples — verify your setup)

**Windows PowerShell:**
```powershell
python -m src.train --smoke-test --epochs 1 --mode true_labels
```

**macOS/Linux:**
```bash
python -m src.train --smoke-test --epochs 1 --mode true_labels
```

Expected output: device announcement (cuda/mps/cpu), download progress, 1-epoch training, results saved to `results/`.

## Running Experiments

### E1 — Randomization Tests (Figure 1a/1b of paper)
```bash
# True labels (baseline)
python -m src.train --mode randomization --input-mode true --seeds 42 123 456 --epochs 100

# Random labels (100% corruption)
python -m src.train --mode randomization --input-mode true --corruption 1.0 --seeds 42 123 456 --epochs 100

# Shuffled pixels
python -m src.train --mode randomization --input-mode shuffled_pixels --seeds 42 123 456 --epochs 100

# Random pixels
python -m src.train --mode randomization --input-mode random_pixels --seeds 42 123 456 --epochs 100

# Gaussian noise
python -m src.train --mode randomization --input-mode gaussian_noise --seeds 42 123 456 --epochs 100
```

### E2 — Label Corruption Sweep (Figure 1c of paper)
```bash
python -m src.train --mode corruption_sweep --seeds 42 123 456 --epochs 100
# (Runs all corruption rates: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 automatically)
```

### Evaluate & Generate Figures
```bash
# Aggregate results (mean ± std across seeds)
python -m src.evaluate --results-dir results/ --pattern "*.json"

# Generate PDF plots
python -m src.plot --results-dir results/ --output-dir figures/
```

### Theorem 1 Constructive Proof
```bash
python -m src.constructive
```

## Device Selection
Device is selected automatically in `src/utils.py`: `cuda → mps → cpu`. No manual configuration needed.

## Reproducibility
- All experiments use 3 seeds: 42, 123, 456
- Results reported as mean ± std across seeds
- Corruption applied once per seed (not re-randomized per epoch)
- Test set always uses true (clean) labels
- No data augmentation, weight decay, or dropout for E1/E2

## Authors
CMP4509 term project group — Fall 2026.
