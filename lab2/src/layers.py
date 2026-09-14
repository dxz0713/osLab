from __future__ import annotations

import numpy as np


def affine_forward(x, w, b):
    """Forward pass for an affine (fully connected) layer.

    Inputs:
    - x: (N, d_1, ..., d_k), flattened internally to (N, D) with D = prod(d_i)
    - w: (D, M)
    - b: (M,)

    Returns a tuple of:
    - out: (N, M)
    - cache: (x, w, b)
    """
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Backward pass for an affine layer.

    Inputs:
    - dout: (N, M) upstream gradient
    - cache: (x, w, b) from affine_forward

    Returns a tuple of:
    - dx: same shape as x
    - dw: (D, M)
    - db: (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """ReLU forward pass. Returns (out, cache)."""
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """ReLU backward pass. Gradient is zero where the input was non-positive."""
    x = cache
    dx = dout * (x > 0)
    return dx


def softmax_loss(scores, y):
    """Softmax cross-entropy loss and its gradient w.r.t. scores.

    Inputs:
    - scores: (N, C)
    - y: (N,) integer labels in [0, C)

    Returns a tuple of:
    - loss: float
    - dscores: (N, C)
    """
    shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = scores.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N
    return loss, dscores
