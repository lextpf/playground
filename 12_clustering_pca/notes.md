# Notes: Unsupervised learning, k-means & PCA

_Personal learning journal, 2026-06-21._

## Supervised vs unsupervised, in my own words

The thing that finally clicked today: **supervised learning has labels, unsupervised does not.**

- **Supervised**: I have `(X, y)` pairs and I'm learning a mapping `X -> y`. The
  ground truth tells me when I'm wrong, so I can measure accuracy/error directly.
  Examples: classification (predict a category) and regression (predict a number).
- **Unsupervised**: I only have `X`. There's no answer key. I'm trying to find
  *structure* in the data itself, groups, directions of variation, density.
  Examples: clustering (k-means) and dimensionality reduction (PCA).

A subtle point I want to remember: in my k-means script I *did* have true blob
labels, but I deliberately hid them from the algorithm. I only used them
afterwards (via adjusted Rand score) to grade the clustering. In a real
unsupervised problem those labels simply wouldn't exist, and evaluation is much
harder because there's nothing to compare against.

## Clustering vs classification, when do I reach for each?

- Use **classification** when I already know the categories and have labeled
  examples of each. The goal is to assign *new* points to *known* classes.
- Use **clustering** when I don't have labels (or want to discover groups I
  didn't predefine). The goal is to *find* the groups. The cluster IDs are
  arbitrary, "cluster 0" has no inherent meaning, which is why a metric like
  adjusted Rand score ignores label numbering.

Rough rule: **labels available + fixed set of classes -> classify. No labels +
"what natural groups exist here?" -> cluster.**

### k-means gotchas I hit / want to watch for
- It only finds a **local optimum** and depends on centroid initialization, so
  multiple restarts (`n_init`) matter.
- I have to choose **k** up front. The **elbow method** helps: plot inertia
  (within-cluster SSE) vs k and look for the bend where extra clusters stop
  helping much. With my 4 synthetic blobs the elbow showed up around k=4.
- It assumes roughly spherical, similarly-sized clusters and is scale-sensitive,
  so standardizing features is usually wise.

## What PCA is good for

PCA finds new orthogonal axes ordered by how much **variance** they capture, then
lets me keep just the top few. It's useful for:

- **Visualization**: squashing 4D Iris down to 2D so I can actually plot it.
- **Compression / noise reduction**: dropping low-variance directions that are
  often just noise.
- **Speeding up / de-correlating** downstream models by feeding them fewer,
  uncorrelated features.

Caveats to keep in mind:
- It's **lossy**, I checked `explained_variance_ratio_` to see how much
  information the 2 components retained (most of it, for Iris).
- It's **linear** and **scale-sensitive** (standardize first!).
- Components are combinations of original features, so they can be harder to
  interpret than the raw columns.

## One-line takeaway
Unsupervised methods don't predict a target, they reveal structure: k-means
groups points, PCA reshapes the axes to where the variance actually lives.
