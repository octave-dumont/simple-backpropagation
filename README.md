# Micro-Autograd & MLP Engine


A lightweight backpropagation engine for building and training neural networks from scratch using NumPy. It implements scalar-level differentiation, and has a multi-layer perceptron (MLP) built on top of it. 

> This code is mainly based on Karpathy's backpropagation course. It has been adjusted and contains additional features (such as custom-functions handling). 

---

## Features

- **Scalar Autograd Engine**: Tracks computation graphs at the scalar level and computes gradients via backpropagation.
- **Custom Function Support**: Differentiate any callable using the central-difference approximation formula:

  $$ f'(x)\approx\frac{f(x+h) - f(x-h)}{2h} $$
  
- **Neural Network Stack**: Abstractions for `Neuron`, `Layer` and `MLP`.
- **Training**: The `MLP` class includes a simple training loop with adjustable learning rate and loss.

---

## Structure

```
backpropagator.py   # Defines Value class (core autograd engine)
mlp.py              # Defines Neuron, Layer, MLP, and training loop
training_example.py # Demo script: generates data and trains a model
```

---

## Basic Usage

### 1) Autograd Example

```python
from backpropagator import Value

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = a * b + Value(10.0)
c.backward()

print(a.grad)  # dc/da = b
```

### 2) Custom Function (Numerical Derivative)

```python
import numpy as np
from backpropagator import Value

x = Value(0.5)
y = x.func(np.exp)  # Differentiates np.exp numerically
y.backward()

print(x.grad)
```

### 3) MLP Example

```python
from mlp import MLP

# 2 inputs → two hidden layers (4,4) → 1 output
model = MLP(2, [4, 4, 1])

x = [2.0, 3.0]
print(model(x))
```

---

## Training Example

Run the included demo to train a small network on synthetic blobs:

```bash
python training_example.py
```

This script generates two data clusters and trains the MLP using stochastic gradient descent to separate them.

---

## How It Works

The engine builds a **computational graph** where each node is a `Value`.  
Calling `.backward()` performs reverse-mode autodiff using the chain rule:

$$\frac{dL}{dx} = \sum_i \frac{dL}{dy_i} \frac{dy_i}{dx}$$

Optimization is made using an MSE loss:

$$L = \sum_i (y_\text{pred} - y_\text{true})^2$$

> Remark: it is extremely easy to implement any loss you want inside the code.

---

## Notes

Used for learning how backpropagation and neural networks work internally, this module does not aim for optimizing runtime/performance.
