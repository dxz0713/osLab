import numpy as np
import pytest

from src.layers import (
    affine_backward,
    affine_forward,
    relu_backward,
    relu_forward,
    softmax_loss
)
from tests.gradient_check import numeric_gradient, rel_error


def test_affine_forward_shapes_and_value():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    b = np.array([1.0, 2.0, 3.0])
    out, cache = affine_forward(x, w, b)
    assert out.shape == (2, 3)
    assert np.allclose(out, np.array([[2.0, 4.0, 4.0], [4.0, 6.0, 4.0]]))
    assert len(cache) == 3


def test_affine_forward_flattens_higher_dims():
    """Inputs of shape (N, d1, d2) must be flattened to (N, d1*d2)."""
    rng = np.random.RandomState(0)
    x = rng.randn(4, 2, 3)
    w = rng.randn(6, 5)
    b = rng.randn(5)
    out, _ = affine_forward(x, w, b)
    assert out.shape == (4, 5)
    assert np.allclose(out, x.reshape(4, 6) @ w + b)


def test_affine_backward_gradients():
    rng = np.random.RandomState(1)
    for shape, M in [((5, 4), 3), ((7, 2, 3), 6), ((3, 8), 2)]:
        x = rng.randn(*shape)
        D = int(np.prod(shape[1:]))
        w = rng.randn(D, M)
        b = rng.randn(M)
        dout = rng.randn(shape[0], M)

        _, cache = affine_forward(x, w, b)
        dx, dw, db = affine_backward(dout, cache)
        assert dx.shape == x.shape
        assert dw.shape == w.shape
        assert db.shape == b.shape

        f = lambda: float(np.sum(affine_forward(x, w, b)[0] * dout))  # noqa: E731
        assert rel_error(dx, numeric_gradient(f, x)) < 1e-7
        assert rel_error(dw, numeric_gradient(f, w)) < 1e-7
        assert rel_error(db, numeric_gradient(f, b)) < 1e-7


def test_relu_forward_and_backward():
    x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    out, cache = relu_forward(x)
    assert np.allclose(out, np.array([[0.0, 0.0, 0.0, 0.5, 2.0]]))

    dout = np.ones_like(x)
    dx = relu_backward(dout, cache)
    assert np.allclose(dx, np.array([[0.0, 0.0, 0.0, 1.0, 1.0]]))


def test_relu_gradient_numeric():
    rng = np.random.RandomState(2)
    x = rng.randn(6, 5)
    x[np.abs(x) < 1e-3] = 0.5  # avoid the kink at zero
    dout = rng.randn(6, 5)
    _, cache = relu_forward(x)
    dx = relu_backward(dout, cache)
    f = lambda: float(np.sum(relu_forward(x)[0] * dout))  # noqa: E731
    assert rel_error(dx, numeric_gradient(f, x)) < 1e-7


def test_softmax_loss_value_and_gradient():
    N, C = 8, 5
    loss, dscores = softmax_loss(np.zeros((N, C)), np.arange(N) % C)
    assert loss == pytest.approx(np.log(C))
    assert np.allclose(dscores.sum(axis=1), 0.0, atol=1e-12)

    rng = np.random.RandomState(4)
    scores = rng.randn(10, 4) * 3
    y = rng.randint(0, 4, size=10)
    _, dscores = softmax_loss(scores, y)
    num = numeric_gradient(lambda: softmax_loss(scores, y)[0], scores)
    assert rel_error(dscores, num) < 1e-5


def test_softmax_loss_numerical_stability():
    scores = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1002.0, -1001.0]])
    loss, dscores = softmax_loss(scores, np.array([1, 0]))
    assert np.isfinite(loss)
    assert np.all(np.isfinite(dscores))