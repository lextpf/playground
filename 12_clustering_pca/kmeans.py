"""Learning goal: get a feel for k-means clustering on synthetic blobs.

Today I want to internalize that k-means is *unsupervised*, it never sees the
true labels. So I generate data where I happen to know the ground-truth blob
each point came from, fit KMeans without those labels, and then sneak a peek
afterwards to see how well the discovered clusters line up with reality.

Two things I'm trying to understand:
  1. inertia_ = within-cluster sum of squared distances (lower = tighter blobs).
  2. how do I even pick k? The elbow idea (see comment in main()).
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def make_data(n_samples=300, centers=4, random_state=0):
    """Synthetic, well-separated-ish blobs. y_true is the ground truth I'll
    pretend not to know while clustering."""
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=0.90,
        random_state=random_state,
    )
    return X, y_true


def fit_kmeans(X, k, random_state=0):
    # n_init="auto" lets sklearn pick a sensible number of restarts; multiple
    # restarts matter because k-means only finds a *local* optimum and is
    # sensitive to the initial centroid placement.
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels


def elbow_inertias(X, k_values, random_state=0):
    """Compute inertia for a range of k. The 'elbow' is where adding another
    cluster stops buying us much reduction in inertia, that bend is a heuristic
    for a good k. Here I just print the numbers and eyeball the drop-off."""
    inertias = []
    for k in k_values:
        km, _ = fit_kmeans(X, k, random_state=random_state)
        inertias.append(km.inertia_)
    return inertias


def main():
    np.random.seed(0)

    X, y_true = make_data()
    print(f"Data shape: {X.shape}  (no labels are given to the clusterer)")

    # I generated 4 blobs, so I cluster with k=4. In a real problem I wouldn't
    # know this and would lean on the elbow heuristic below.
    k = 4
    km, labels = fit_kmeans(X, k)

    print(f"\nFitted KMeans with k={k}")
    print(f"  inertia (within-cluster SSE): {km.inertia_:.2f}")

    # Adjusted Rand Score compares the discovered clustering to the true labels.
    # It's invariant to how the cluster *numbers* get assigned (cluster 0 here
    # might be blob 2 in y_true), which is exactly what I want, I only care
    # whether the same points got grouped together. 1.0 = perfect, ~0 = random.
    ari = adjusted_rand_score(y_true, labels)
    print(f"  adjusted_rand_score vs true blobs: {ari:.3f}")

    # Elbow exploration: watch how inertia falls as k grows. Because I built 4
    # blobs, I expect a noticeable bend around k=4, after which extra clusters
    # only nibble at the inertia.
    print("\nElbow exploration (k -> inertia):")
    k_values = range(1, 8)
    for k_i, inertia in zip(k_values, elbow_inertias(X, k_values)):
        print(f"  k={k_i}: {inertia:8.2f}")


if __name__ == "__main__":
    main()
