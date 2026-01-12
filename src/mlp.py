from __future__ import annotations

import numpy as np
import numpy.random as random
from typing import Iterable, List, Union, Callable

from backpropagator import Value

Scalar = Union[float, int, Value]

class Neuron:
    
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-0.1,0.1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x: Iterable[Scalar]) -> Value:
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.func(np.tanh)
        return out
    
    def parameters(self) -> List[Value]:
        return self.w + [self.b]
    
class Layer:

    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: Iterable[Scalar]) -> Union[Value, List[Value]]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]

def mse(ypred: Iterable[Scalar], ytrue: Iterable[Value]):
    return sum((yp - yt)**2 for yp, yt in zip(ypred, ytrue))

class MLP:
    """
    Designs a Multi Layer Perceptron.
    Args:
        'nin': The dimension of each input vector.
        'nouts': A list of all the number of neurons per layer. 
                 Each element of the list is one layer.
    """
    def __init__(self, nin: int, nouts: List[int]):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x: Iterable[Scalar]) -> Union[Value, List[Value]]:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
    
    def train(self, X: Iterable[Scalar], y: Iterable[Scalar], 
              step_size: float=0.05, loss_f: Callable=mse, steps: int=100, verbose: bool=True) -> Iterable[Value]:
        """
        Trains the MLP on X input data to fit the y input. 
        Outputs the predicted value.
        """
        
        for k in range(steps):
            ypred = [self(x) for x in X]
            loss = loss_f(ypred, y)

            for p in self.parameters():
                p.grad = 0.0

            loss.backward()

            for p in self.parameters():
                p.data -= step_size * p.grad
                
            if verbose and (k == 1 or k % (steps//5) == steps//5 - 1):
                if k == 1 or k % 20 == 19:
                    print(f"Training step {k}, loss={float(loss.data):.4f}")
        return ypred



