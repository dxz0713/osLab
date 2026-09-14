from __future__ import annotations

import math


class Value:
    """A scalar node in a dynamically built computation graph.

    Wraps a single number and records how it was produced, so that the
    gradient of any later result with respect to it can be recovered.

    Attributes:
        data: the value itself, a Python float.
        grad: the derivative of the final output w.r.t. this node. Starts at
            zero and is filled in by backward().
        _prev: the set of nodes this one was computed from.
        _op: the name of the operation that produced this node, for debugging.
        _backward: a no-argument function. When it runs, this node's grad is
            already correct, and its job is to pass that gradient on to the
            nodes in _prev.
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        """Add two nodes, or a node and a plain number.

        The construction of `out` is given here as a worked example: note how
        the operands are passed as _children so that the graph stays connected.
        The remaining operators follow the same pattern.

        Args:
            other: another Value, or an int/float that will be wrapped in one.

        Returns:
            A new Value holding the sum, linked to both operands.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Multiply two nodes, or a node and a plain number.

        Args:
            other: another Value, or an int/float that will be wrapped in one.

        Returns:
            A new Value holding the product, linked to both operands.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        """Raise this node to a constant power.

        Args:
            other: an int or float exponent. Raising to a Value power is not
                supported.

        Returns:
            A new Value holding the result, linked to this node.
        """
        assert isinstance(other, (int, float)), "only constant exponents are supported"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """Clamp negative values to zero.

        Returns:
            A new Value holding max(0, data), linked to this node.
        """
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """Squash the value into the range (-1, 1).

        Returns:
            A new Value holding tanh(data), linked to this node.
        """
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """Take the natural exponential.

        Returns:
            A new Value holding e ** data, linked to this node.
        """
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        """Take the natural logarithm.

        Returns:
            A new Value holding ln(data), linked to this node.
        """
        assert self.data > 0, "log requires a positive input"

        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Fill in the grad of every node this one was built from.

        Treats this node as the final output, so its own grad starts at one.
        Nothing is returned; the results are written into the grad attribute
        of each node in the graph.
        """
        topo = []
        visited = set()

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    dfs(child)
                topo.append(node)

        dfs(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # The methods below are already written and need no changes.

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"