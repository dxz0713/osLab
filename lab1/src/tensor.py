from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
Shape = tuple[int, ...]


def _unbroadcast(grad: Array, shape: Shape) -> Array:
    """Sum a gradient back down to the shape it was broadcast from.

    The inverse of NumPy's broadcasting rules. Whatever the forward pass
    stretched, the backward pass has to sum back up, or the gradient will
    not have the same shape as the operand it belongs to.

    Must handle any shape pair NumPy would accept, not just the (C,) bias
    against an (N, C) output that this lab happens to use.

    Args:
        grad: gradient carrying the broadcast (result) shape.
        shape: the operand shape to reduce back to.

    Returns:
        Array of exactly `shape`.
    """
    while grad.shape != shape:
        if len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        else:
            for i, (g, s) in enumerate(zip(grad.shape, shape)):
                if g != s:
                    grad = grad.sum(axis=i, keepdims=True)
                    break
    return grad


class Tensor:
    """An array-valued node in a dynamically built computation graph.

    Same machinery as Value in engine.py -- children, a local backward
    closure, and one topological sweep -- but every node now holds a whole
    NumPy array instead of a single number. The scheduling logic is
    unchanged; only the local derivatives are written in matrix form.

    Attributes:
        data: the value itself, a float64 ndarray of any shape. A scalar is
            stored as a 0-d array, not a Python float.
        grad: derivative of the final output w.r.t. this node. Always the
            same shape as `data`. Starts at zero and is filled in by
            backward().
        _prev: the nodes this one was computed from.
        _op: name of the operation that produced this node, for debugging.
        _backward: a no-argument function. When it runs, this node's grad is
            already correct, and its job is to pass that gradient on to the
            nodes in _prev.
    """

    data: Array
    grad: Array

    def __init__(self, data: ArrayLike, _children: tuple[Tensor, ...] = (), _op: str = "") -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self) -> Shape:
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def __add__(self, other: Tensor) -> Tensor:
        """Add two tensors, broadcasting if their shapes differ.

        This one is given as a worked example, and shows how _unbroadcast
        is meant to be called: without it, a (C,) bias added to an (N, C)
        score matrix would come back with the wrong shape. Addition has a
        local derivative of 1 on both sides, so the upstream gradient is
        passed straight through.

        Args:
            other: another Tensor, of any shape NumPy can broadcast against
                this one.

        Returns:
            A new Tensor of the broadcast shape, linked to both operands.
        """
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: Tensor) -> Tensor:
        """Multiply two tensors elementwise, broadcasting if shapes differ.

        Compare this with __add__ once you have written it: the second half
        of each _backward is the same _unbroadcast call, and only the first
        half -- the local derivative -- differs between the two operators.

        Args:
            other: another Tensor, of any shape NumPy can broadcast against
                this one.

        Returns:
            A new Tensor of the broadcast shape, linked to both operands.
        """
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix-multiply two 2-D tensors.

        Shapes: (N, K) @ (K, M) -> (N, M). Batched inputs are out of scope
        for this lab.

        Args:
            other: a 2-D Tensor whose first dimension matches this one's
                second dimension.

        Returns:
            A new Tensor of shape (N, M), linked to both operands.
        """
        out = Tensor(np.matmul(self.data, other.data), (self, other), "@")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad @ other.data.T, self.data.shape)
            other.grad += _unbroadcast(self.data.T @ out.grad, other.data.shape)

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        """Clamp negative entries to zero, elementwise.

        Returns:
            A new Tensor of the same shape as this one, linked to it.
        """
        out = Tensor(np.maximum(0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Fill in the grad of every node this one was built from.

        Treats this node as the final output, so its own grad starts at all
        ones -- an array of the same shape as its data, not the scalar 1.0
        that Value.backward used. Nothing is returned; results are written
        into each node's grad.
        """
        topo = []
        visited = set()

        def dfs(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    dfs(child)
                topo.append(node)

        dfs(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"


def softmax_cross_entropy(logits: Tensor, y: NDArray[np.int64]) -> Tensor:
    """Turn raw class scores into a loss, and route its gradient back.

    Fuses log-softmax and the negative log-likelihood into a single graph
    node, the way torch.nn.CrossEntropyLoss does. Fusing them keeps the
    exponentials from overflowing and lets the gradient collapse to a much
    simpler expression than either step would give on its own.

    Args:
        logits: Tensor of shape (N, C) holding unnormalized class scores
            for N samples over C classes.
        y: integer array of shape (N,), each entry in [0, C). These are
            class indices, not one-hot rows, and not a Tensor -- labels are
            data, and gradients do not flow into them.

    Returns:
        A Tensor wrapping a 0-d array: the mean cross-entropy over the
        batch. Its grad will therefore also be 0-d.
    """
    y = np.asarray(y)
    N = y.shape[0]

    # 数值稳定的 log-softmax: 减去 row max 防止 exp 上溢
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    # 平均负对数似然
    out = Tensor(-np.mean(log_probs[np.arange(N), y]), (logits,), "cross_entropy")

    def _backward() -> None:
        # softmax 概率 (重用 shifted 无需重新计算 max)
        probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        # dL/dz_i = (p_i - 1_{i==y}) / N
        probs[np.arange(N), y] -= 1.0
        logits.grad += probs / N

    out._backward = _backward
    return out
