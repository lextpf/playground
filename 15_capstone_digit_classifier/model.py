"""Model definition for the digit classifier capstone.

Learning goal: write a small, self-contained ``nn.Module`` that I can import
from both train.py and evaluate.py. I went with a tiny CNN: the 64-length
feature vector is reshaped back into its native 8x8 image so the convolutions
have spatial structure to work with, a nice payoff after a month of building
up to "real" architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """A compact CNN for 8x8 grayscale digit images.

    Input is the flattened (N, 64) feature tensor produced by data.py. The
    forward pass reshapes it to (N, 1, 8, 8), runs two conv blocks, then a
    small classifier head. Padding keeps the spatial size at 8x8 so a single
    2x2 pool brings it down to 4x4 before flattening.
    """

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)            # 8x8 -> 4x4
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # Reshape the flat feature vector back into a single-channel image.
        x = x.view(-1, 1, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)               # raw logits; CrossEntropyLoss applies softmax
        return x


def build_model(n_classes=10):
    """Factory so callers don't depend on the class name directly."""
    return DigitCNN(n_classes=n_classes)


def main():
    """Smoke test: push a fake batch through and check the output shape."""
    torch.manual_seed(0)
    model = build_model()
    dummy = torch.randn(4, 64)
    logits = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(logits.shape)}  (expected (4, 10))")
    print(f"trainable params: {n_params:,}")


if __name__ == "__main__":
    main()
