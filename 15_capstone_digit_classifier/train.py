"""Training loop for the digit classifier capstone.

Learning goal: tie the data and model modules together into a proper training
script, optimizer, loss, an epoch loop with per-epoch loss and train
accuracy, plus a small argparse so I can tweak epochs/lr without editing code.
The trained weights are saved next to this file so evaluate.py can pick them up.

Run this from INSIDE this folder:  python train.py
"""

import argparse
import os

import torch
import torch.nn as nn

from data import load_data, seed_everything
from model import build_model

# Save the checkpoint beside this script, regardless of the caller's cwd.
CKPT_PATH = os.path.join(os.path.dirname(__file__), "digit_cnn.pt")


def train_one_epoch(model, loader, optimizer, criterion):
    """Run a single pass over the training data; return (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += xb.size(0)

    return running_loss / total, correct / total


def train(epochs=15, lr=1e-3):
    """Train the CNN and persist its weights. Returns the trained model."""
    seed_everything()

    data = load_data()
    model = build_model(n_classes=data["n_classes"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, data["train_loader"], optimizer, criterion)
        print(f"epoch {epoch:2d}/{epochs}  loss={loss:.4f}  train_acc={acc:.4f}")

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"saved weights -> {CKPT_PATH}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the digit classifier CNN.")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
