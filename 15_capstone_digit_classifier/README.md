# Capstone: Handwritten Digit Classifier (PyTorch)

My capstone project. A small but complete PyTorch project that classifies 8x8
handwritten digits with a tiny convolutional neural network. It's split into
focused modules like a real mini-project, so the data and model code are
reusable from both training and evaluation.

## What it does

- Loads the offline **scikit-learn `load_digits`** dataset (1797 images, 8x8,
  10 classes), no internet, no downloads.
- Standardizes the features and makes a seeded, stratified train/test split.
- Trains a compact CNN (`DigitCNN`) that reshapes each 64-length feature vector
  back into its native 8x8 image before convolving.
- Evaluates with scikit-learn metrics: overall accuracy, a per-class report,
  and a confusion matrix.

## Dataset

`sklearn.datasets.load_digits` ships *inside* scikit-learn, so it loads
instantly and works fully offline. Each sample is an 8x8 grayscale image of a
handwritten digit (0-9), flattened to 64 features.

## Files

| File          | Role                                                        |
| ------------- | ----------------------------------------------------------- |
| `data.py`     | Load, scale, split, and wrap data into DataLoaders.         |
| `model.py`    | The `DigitCNN` `nn.Module` plus a `build_model` factory.    |
| `train.py`    | Training loop with argparse; saves `digit_cnn.pt`.          |
| `evaluate.py` | Loads the checkpoint and reports test metrics.              |

## How to run

Run everything **from inside this folder** (the scripts import each other by
module name):

```bash
cd 15_capstone_digit_classifier
python train.py                 # defaults: 15 epochs, lr=1e-3
python train.py --epochs 25 --lr 5e-4   # or tweak the knobs
python evaluate.py              # after training, prints test metrics
```

`train.py` writes the trained weights to `digit_cnn.pt` in this folder, which
`evaluate.py` then loads.

## Expected accuracy

The digits set is small and clean, so the CNN trains quickly on CPU. With the
defaults I see **train accuracy near ~99%** and **test accuracy around 96-98%**.
Numbers vary slightly run to run, but with the seeds pinned a given config is
reproducible.

## Reflection

This one pulls all the threads together. I started with plain Python and numpy,
moved through scikit-learn for loading and preprocessing (train/test splits,
`StandardScaler`, metrics), then picked up PyTorch piece by piece: tensors,
`nn.Module`, autograd, optimizers, the `DataLoader` pipeline. All of those
threads meet here: a clean data module, an importable model, a real training
loop with pinned seeds, and an evaluation step that goes beyond a single
accuracy number. Splitting it across files instead of one big script is the
habit I most wanted to build, and it finally feels natural. Good place to stop.
