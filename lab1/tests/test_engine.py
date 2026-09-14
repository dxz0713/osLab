import math

import numpy as np
import pytest

from src.engine import Value


def test_forward_basic_ops():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == pytest.approx(-1.0)
    assert (a * b).data == pytest.approx(-6.0)
    assert (a**3).data == pytest.approx(8.0)
    assert (a + 1.0).data == pytest.approx(3.0)
    assert (2.0 * a).data == pytest.approx(4.0)
    assert (a - b).data == pytest.approx(5.0)
    assert (a / b).data == pytest.approx(2.0 / -3.0)


def test_backward_single_op():
    a, b = Value(2.0), Value(-3.0)
    c = a * b
    c.backward()
    assert a.grad == pytest.approx(-3.0)
    assert b.grad == pytest.approx(2.0)


def test_backward_accumulates_gradients():
    """y = a + a must give dy/da = 2, which requires += instead of =."""
    a = Value(3.0)
    y = a + a
    y.backward()
    assert a.grad == pytest.approx(2.0)


def test_backward_diamond_graph():
    """Two paths from a to d; a topological order is required."""
    a = Value(4.0)
    b = a * 2
    c = a * 3
    d = b + c
    d.backward()
    assert d.data == pytest.approx(20.0)
    assert a.grad == pytest.approx(5.0)


def test_backward_shared_subgraph():
    """out = 9a^2 + 3a, so d(out)/da = 18a + 3 = 39 at a = 2."""
    a = Value(2.0)
    m = a * 3
    out = m * m + m
    out.backward()
    assert a.grad == pytest.approx(39.0)


def test_activation_forward_values():
    x = Value(0.5)
    assert x.relu().data == pytest.approx(0.5)
    assert Value(-0.5).relu().data == pytest.approx(0.0)
    assert x.tanh().data == pytest.approx(math.tanh(0.5))
    assert x.exp().data == pytest.approx(math.exp(0.5))
    assert x.log().data == pytest.approx(math.log(0.5))


def test_activation_gradients():
    for x0 in [-1.7, -0.3, 0.4, 2.1]:
        x = Value(x0)
        x.tanh().backward()
        assert x.grad == pytest.approx(1 - math.tanh(x0) ** 2, rel=1e-9)

        x = Value(x0)
        x.exp().backward()
        assert x.grad == pytest.approx(math.exp(x0), rel=1e-9)

        x = Value(x0)
        x.relu().backward()
        assert x.grad == pytest.approx(1.0 if x0 > 0 else 0.0)

        if x0 > 0:
            x = Value(x0)
            x.log().backward()
            assert x.grad == pytest.approx(1.0 / x0, rel=1e-9)


def test_composite_expression_against_numeric_gradient():
    """Random expressions checked against central differences."""
    rng = np.random.RandomState(0)

    def build(x, y, z):
        a, b, c = Value(x), Value(y), Value(z)
        out = (a * b + c**2).tanh() + (a - c).relu() * b + (a * c).log()
        return a, b, c, out

    for _ in range(8):
        p = rng.uniform(0.3, 2.0, size=3)
        a, b, c, out = build(*p)
        out.backward()
        h = 1e-6
        for var, i in [(a, 0), (b, 1), (c, 2)]:
            hi, lo = p.copy(), p.copy()
            hi[i] += h
            lo[i] -= h
            num = (build(*hi)[3].data - build(*lo)[3].data) / (2 * h)
            assert var.grad == pytest.approx(num, abs=1e-4)


def test_long_dependency_chain():
    """Gradients must stay finite and positive along a long chain.

    The chain is kept short enough that a recursive topological sort fits
    inside Python's default recursion limit; this test is about numerical
    behaviour, not about stack depth.
    """
    x = Value(1.0)
    y = x
    for _ in range(200):
        y = y * 1.001 + 0.001
    y.backward()
    assert math.isfinite(x.grad) and x.grad > 0


def test_mlp_can_learn():
    """Train a 2-layer MLP on noisy sin(x) using only the engine."""
    rng = np.random.RandomState(0)
    xs = np.linspace(-3.0, 3.0, 40)
    ys = np.sin(xs) + 0.05 * rng.randn(40)

    H = 8
    W1 = [Value(rng.randn() * 0.8) for _ in range(H)]
    b1 = [Value(0.0) for _ in range(H)]
    W2 = [Value(rng.randn() * 0.8) for _ in range(H)]
    b2 = Value(0.0)
    params = W1 + b1 + W2 + [b2]

    def forward(x):
        h = [(W1[j] * x + b1[j]).tanh() for j in range(H)]
        out = b2
        for j in range(H):
            out = out + W2[j] * h[j]
        return out

    def epoch(lr):
        for p in params:
            p.grad = 0.0
        loss = Value(0.0)
        for x, y in zip(xs, ys):
            diff = forward(float(x)) - float(y)
            loss = loss + diff * diff
        loss = loss * (1.0 / len(xs))
        loss.backward()
        for p in params:
            p.data -= lr * p.grad
        return loss.data

    first = epoch(0.05)
    last = first
    for _ in range(150):
        last = epoch(0.05)

    assert last < first * 0.2
    assert last < 0.05
