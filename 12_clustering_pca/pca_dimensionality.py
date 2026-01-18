"""Learning goal: understand PCA as a dimensionality-reduction tool on Iris.

Iris lives in 4D (sepal length/width, petal length/width). I can't visualize 4D
in my head, so I want to project it down to 2D while keeping as much of the
spread (variance) as possible.

What PCA is actually doing:
  - It finds new orthogonal axes (principal components) ordered by how much
    variance of the data they capture.
  - PC1 is the direction of greatest variance, PC2 the greatest remaining
    variance orthogonal to PC1, and so on.
  - Keeping the top-2 components is a *lossy* compression: I trade some
    information for a 4D -> 2D view that's easy to plot and reason about.

Note: PCA is sensitive to feature scale, so I standardize first. Iris features
are all in cm and roughly comparable, but standardizing is the correct habit.
"""

import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names


def run_pca(X, n_components=2, random_state=0):
    # Standardize so each feature contributes on equal footing (zero mean, unit
    # variance) before PCA measures "variance along a direction".
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)
    return pca, X_reduced


def maybe_plot(X_reduced, y, target_names):
    """Optional 2D scatter of the projected data, colored by species. Saved as a
    PNG next to this script (headless 'Agg' backend, never shown)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for class_idx, name in enumerate(target_names):
        mask = y == class_idx
        ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], label=name, alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Iris projected to 2 principal components")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "pca_iris.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


def main():
    np.random.seed(0)

    X, y, feature_names, target_names = load_data()
    print(f"Original shape: {X.shape}  ({len(feature_names)} features: {feature_names})")

    pca, X_reduced = run_pca(X, n_components=2)
    print(f"Reduced shape:  {X_reduced.shape}  (4D -> 2D)")

    # explained_variance_ratio_ tells me the fraction of total variance each
    # component captures. Summing them is how much information I kept overall.
    evr = pca.explained_variance_ratio_
    print("\nExplained variance ratio per component:")
    for i, ratio in enumerate(evr, start=1):
        print(f"  PC{i}: {ratio:.4f}")

    cumulative = np.cumsum(evr)
    print(f"\nCumulative variance retained by 2 components: {cumulative[-1]:.4f}")
    print("So with just 2 of the original 4 dimensions I keep most of the signal.")

    maybe_plot(X_reduced, y, target_names)


if __name__ == "__main__":
    main()
