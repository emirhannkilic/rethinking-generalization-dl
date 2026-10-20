# Rethinking Generalization: a reproduction

Reproduction of Zhang et al., *Understanding Deep Learning Requires Rethinking Generalization* (ICLR 2017, [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)), done for the CMP4509 Introduction to Deep Learning term project (Fall 2026).

The paper's central claim: standard networks can fit random labels on CIFAR-10 to (near) zero training error, so capacity alone cannot explain why they generalize, and explicit regularizers are neither necessary nor sufficient.

## Status

Work in progress. Experiments and results will be listed here as they are completed.

| ID | Experiment | Paper reference | Status |
|----|------------|-----------------|--------|
| E1 | Randomization tests (true / random labels, shuffled / random pixels, Gaussian noise) | Fig. 1a, 1b | todo |
| E2 | Label-corruption sweep | Fig. 1c | todo |
| E3 | Explicit regularizers on/off | Table 1 | todo |
| E4 | Constructive proof of Theorem 1 (2n+d parameters fit any labels) | Sec. 4 | todo |

Every experiment is scaled down to fit free GPU platforms; each reduction is documented in the report.

## Setup

Python 3.12 is recommended (matches Colab/Kaggle).

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Install PyTorch and torchvision with the command for your platform from <https://pytorch.org> (pick the CUDA build on Windows/Linux with an NVIDIA GPU, otherwise training silently falls back to CPU). Then:

```bash
pip install -r requirements.txt
```

On Apple Silicon, if an operation is missing on the MPS backend:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

The device is selected in one place, in the order `cuda` -> `mps` -> `cpu`.

## Running

Planned (not implemented yet): a smoke test that runs one epoch on a small subset, to check that the setup works on your machine.

```bash
python -m src.train --config smoke
```

The notebooks in `notebooks/` only call code from `src/` and plot results from `results/`; they run top to bottom on Colab and Kaggle.

## Repository layout

```
src/         data loading and corruptions, models, training, constructive proof
notebooks/   one notebook per experiment (calls src/, plots results)
results/     per-run metrics (CSV/JSON); figures are generated from these
tests/       tests for corruptions and the constructive solution
report/      LaTeX report (course template)
docs/        course material (not tracked)
```

## Notes on reproducibility

- Each configuration is run with several seeds and reported as mean ± std.
- All seeds of a configuration run on the same device; results can differ slightly across CUDA, MPS and CPU.
- The constructive proof (E4) always runs on CPU in float64.

## Authors

CMP4509 term project group (3 students). Names to be added.
