# AI / ML Playground

Just me messing around with Python and some machine learning ideas, learning as
I go. Nothing serious here, mostly little scripts to try things out and see how
they work.

Each folder is a small experiment, very roughly building up from basic NumPy
toward a little neural network.

## Running things

```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python 01_numpy_basics/arrays_and_dtypes.py
```

## Stuff I poked at

- NumPy and vectorization
- pandas
- plotting
- a bit of linear algebra and stats
- regression and gradient descent
- scikit-learn and model evaluation
- trees, forests, clustering
- a neural net from scratch (a perceptron, then an MLP with backprop)
- PyTorch
- a tiny digit classifier to tie things together

The little digit classifier ended up reading handwritten digits at around 97%,
which felt like a fun place to stop. Mostly I just wanted to see how all this
stuff actually works under the hood.
