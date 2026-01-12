from __future__ import annotations

from typing import Tuple
import numpy as np

from mlp import MLP

def make_blobs(n_per_class=10, spread=0.25, seed=0):
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=(-0.7, -0.7), scale=spread, size=(n_per_class, 2))
    c2 = rng.normal(loc=(+0.7, +0.7), scale=spread, size=(n_per_class, 2))
    X = np.vstack([c1, c2])
    y = np.array([-1.0]*n_per_class + [1.0]*n_per_class)
    return X.tolist(), y.tolist()

if __name__ == "__main__":

    # Testing the classes on a very simple example

    model = MLP(2, [4, 4, 1])
    
    x, y = make_blobs()

    model.train(x, y)