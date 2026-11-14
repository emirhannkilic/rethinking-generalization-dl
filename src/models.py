"""
models.py — Neural network architectures for the rethinking-generalization study.

Provides:
  - SmallCNN: A four-block convolutional network (inspired by the paper).
  - MLP:      Single hidden-layer fully-connected network.
  - count_parameters: Utility to count trainable parameters.

Both models expose a ``disable_regularization()`` method that returns a list of
parameter groups with ``weight_decay=0``, suitable for passing directly to an
optimiser.  This matches the paper's E1/E2 experiments which use no explicit
regularisation.
"""

from __future__ import annotations

from typing import List, Dict, Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SmallCNN
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """Small convolutional network inspired by Zhang et al. (2017).

    Architecture (input: N × 3 × 32 × 32 in [0, 1]):

        Block 1:  Conv(3→64,   3×3, pad=1) → BatchNorm2d → ReLU
        Block 2:  Conv(64→128, 3×3, pad=1) → BatchNorm2d → ReLU → MaxPool(2)
        Block 3:  Conv(128→256,3×3, pad=1) → BatchNorm2d → ReLU
        Block 4:  Conv(256→256,3×3, pad=1) → BatchNorm2d → ReLU → MaxPool(2)
        Head:     Flatten → Linear(256·8·8, 256) → ReLU → Linear(256, 10)

    After two 2×2 MaxPool operations on 32×32 images, the spatial size is 8×8.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        """
        Args:
            dropout: Dropout probability applied before the final linear layer.
                     Set to 0.0 (default) to disable, matching E1/E2 protocol.
        """
        super().__init__()

        # Block 1 — no pooling
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Block 2 — pooling
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32→16
        )

        # Block 3 — no pooling
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Block 4 — pooling
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16→8
        )

        # Classification head
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(256, 10),
        )

    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape (N, 3, 32, 32) with values in [0, 1].

        Returns:
            Logit tensor of shape (N, 10).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)

    def disable_regularization(self) -> List[Dict[str, Any]]:
        """Return a parameter-group list with ``weight_decay=0``.

        Usage::

            model = SmallCNN()
            optimizer = torch.optim.SGD(
                model.disable_regularization(), lr=0.01, momentum=0.9
            )
        """
        return [{"params": self.parameters(), "weight_decay": 0.0}]


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Single hidden-layer multilayer perceptron.

    Architecture (input: N × 3072 — flattened 3×32×32 image):

        Linear(3072, 512) → ReLU → [optional Dropout] → Linear(512, 10)

    The input is flattened inside ``forward`` so raw image tensors of shape
    (N, 3, 32, 32) can be passed directly.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 512,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            input_dim:   Flattened input dimensionality (default: 3·32·32=3072).
            hidden_dim:  Number of hidden units (default: 512).
            num_classes: Output classes (default: 10 for CIFAR-10).
            dropout:     Dropout probability; 0.0 disables it.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape (N, 3, 32, 32) or (N, 3072).

        Returns:
            Logit tensor of shape (N, 10).
        """
        return self.net(x)

    def disable_regularization(self) -> List[Dict[str, Any]]:
        """Return a parameter-group list with ``weight_decay=0``."""
        return [{"params": self.parameters(), "weight_decay": 0.0}]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Integer count of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
