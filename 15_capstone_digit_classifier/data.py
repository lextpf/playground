"""Data loading for the digit classifier capstone.

Learning goal: package an offline dataset into the PyTorch data pipeline I
keep reaching for, tensors wrapped in TensorDataset, served by DataLoader
so the training and evaluation scripts can stay clean. Everything here is
CPU-only and uses scikit-learn's built-in 8x8 ``load_digits`` set, so there is
no download and no internet dependency.
"""

import random

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 0


def seed_everything(seed=SEED):
    """Pin every RNG I rely on so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(batch_size=64, test_size=0.2):
    """Load, scale, split, and wrap the digits data.

    Returns a dict with the train/test DataLoaders plus the raw test tensors
    (handy for evaluate.py, which wants the full arrays for sklearn metrics).
    """
    seed_everything()

    # 1797 samples, each an 8x8 image flattened to 64 features; 10 classes.
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    # Standardize on the train split only, then apply to test (no leakage).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    # Seeded generator so shuffling is reproducible across runs.
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "X_test": X_test_t,
        "y_test": y_test_t,
        "n_features": X.shape[1],
        "n_classes": int(y.max()) + 1,
    }


def main():
    """Quick sanity check that the pipeline assembles as expected."""
    data = load_data()
    xb, yb = next(iter(data["train_loader"]))
    print(f"features={data['n_features']} classes={data['n_classes']}")
    print(f"first train batch: X={tuple(xb.shape)} y={tuple(yb.shape)}")
    print(f"test set: X={tuple(data['X_test'].shape)} y={tuple(data['y_test'].shape)}")


if __name__ == "__main__":
    main()
