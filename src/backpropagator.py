from __future__ import annotations

import numpy as np
from typing import Union, Iterable, Callable

class Value:
    """
    Allow to store operations and gradients applied to a number.
    """

    def __init__ (self, data: float, _children: Iterable=(), _op: str='', label: str=''):
        if isinstance(data, Value):
            data = data.data
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda : None
        self.label = label


    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self, other: Value) -> Value:
        other = other if  isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other: Value) -> Value:
        return self + other
    
    
    def __mul__(self, other: Value) -> Value:
        other = other if  isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self) -> Value:
        return self * -1
    
    def __sub__(self, other: Value) -> Value:
        return self + (-other)
        
    def __rmul__(self, other: Value) -> Value:
        return self * other 
    
    def __rsub__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)
    
    
    def __pow__(self, other: Union[Value, int, float]):
        if isinstance(other, Value):
            out = Value(self.data ** other.data, (self, other), f'**')
            base, exp = self, other

            def _backward():
                base.grad += exp.data * (base.data ** (exp.data - 1)) * out.grad
                exp.grad += (out.data * np.log(base.data)) * out.grad

            out._backward = _backward

        else:
            assert isinstance(other, (int, float))
            out = Value(self.data ** other, (self,), f'**{other}')
            base = self

            def _backward():
                base.grad += other * (base.data ** (other - 1)) * out.grad

            out._backward = _backward

        return out

    
    def __truediv__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**(-1))
    
    def __rtruediv__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return other * (self ** (-1))
    
    def func(self, function: Callable) -> Value:
        """
        An addition to Karpathy's code. 
        Allow the user to apply any function to a value.
        """
        out = Value(function(self.data), (self,), f'{function}({self.label})')
        def _backward():
            h = 1e-5
            dfdx = (function(self.data + h) - function(self.data - h)) / (2 * h)
            self.grad += dfdx * out.grad
        
        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Apply backpropagation through the network.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

