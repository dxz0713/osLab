import numpy as np
import pytest

from src.data import iou, make_ring
from src.nn import MLP, Linear, predict, train
from src.tensor import Tensor, softmax_cross_entropy


def test_linear_shapes_and_bias_init():
    rng = np.random.RandomState(0)
    layer = Linear(5, 3, rng=rng)
    assert layer.W.data.shape == (5, 3)
    assert layer.b.data.shape == (3,)
    assert np.allclose(layer.b.data, 0.0)

    out = layer(Tensor(rng.randn(7, 5)))
    assert out.data.shape == (7, 3)


def test_linear_weight_scale():
    """Kaiming init: std should track sqrt(2 / fan_in), not be 1 or 0."""
    rng = np.random.RandomState(1)
    for fan_in in [16, 64, 256]:
        layer = Linear(fan_in, 200, rng=rng)
        expected = np.sqrt(2.0 / fan_in)
        assert layer.W.data.std() == pytest.approx(expected, rel=0.15)


def test_mlp_shapes_and_parameters():
    rng = np.random.RandomState(2)
    model = MLP([2, 8, 8, 2], rng=rng)
    out = model(Tensor(rng.randn(10, 2)))
    assert out.data.shape == (10, 2)

    params = model.parameters()
    assert len(params) == 6
    assert [p.data.shape for p in params] == [
        (2, 8), (8,), (8, 8), (8,), (8, 2), (2,),
    ]


def test_mlp_hidden_layers_are_activated():
    """Without ReLU a deep stack collapses to one linear map."""
    rng = np.random.RandomState(3)
    model = MLP([2, 16, 16, 2], rng=rng)
    X = np.array([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    out = model(Tensor(X)).data
    midpoint = (out[0] + out[1]) / 2
    assert not np.allclose(midpoint, out[2], atol=1e-6)


def test_ring_dataset():
    n, r_inner, r_outer = 32, 0.35, 0.65
    X, y = make_ring(n=n, r_inner=r_inner, r_outer=r_outer)

    assert X.shape == (n * n, 2)
    assert y.shape == (n * n,)
    assert np.issubdtype(y.dtype, np.integer), "labels must be an integer dtype, not bool"
    assert set(np.unique(y)).issubset({0, 1})

    # Each coordinate takes exactly n equally spaced values spanning [-1, 1].
    # Both linspace endpoints and pixel-centre sampling are accepted; an
    # arbitrary cloud of points in the square is not.
    for c in (0, 1):
        vals = np.unique(X[:, c])
        assert len(vals) == n, "each coordinate should take exactly n distinct values"
        assert np.allclose(np.diff(vals), vals[1] - vals[0]), "the grid must be evenly spaced"
        assert -1.0 <= vals[0] and vals[-1] <= 1.0
        assert vals[0] < -1.0 + 2.0 / n and vals[-1] > 1.0 - 2.0 / n

    # reshape(n, n) must recover the grid: one coordinate varies down the rows
    # and the other across the columns. render() and visualize.py rely on this.
    grid = X.reshape(n, n, 2)
    const_along_rows = [bool(np.allclose(grid[:, :, c], grid[:, :1, c])) for c in (0, 1)]
    const_along_cols = [bool(np.allclose(grid[:, :, c], grid[:1, :, c])) for c in (0, 1)]
    assert (const_along_rows[0] and const_along_cols[1]) or (
        const_along_cols[0] and const_along_rows[1]
    ), "rows of X must be ordered so that reshape(n, n) recovers the grid"

    radius = np.sqrt((X**2).sum(axis=1))
    assert np.all(y[radius < r_inner] == 0)
    assert np.all(y[radius > r_outer] == 0)
    assert np.all(y[(radius > r_inner) & (radius < r_outer)] == 1)
    assert y.mean() > 0.1


def test_iou_metric():
    y = np.array([1, 1, 0, 0])
    assert iou(y, y) == pytest.approx(1.0)
    assert iou(np.zeros(4, dtype=int), y) == pytest.approx(0.0)
    assert iou(np.array([1, 0, 0, 0]), y) == pytest.approx(0.5)


def test_train_history_and_progress():
    """history[0] is the loss before any update, and training must help."""
    X, y = make_ring(n=16)
    rng = np.random.RandomState(4)
    model = MLP([2, 16, 2], rng=rng)

    first_loss = float(softmax_cross_entropy(model(Tensor(X)), y).data)
    history = train(model, X, y, lr=0.2, num_epochs=400)
    assert history.shape == (400,)
    assert history[0] == pytest.approx(first_loss, rel=1e-9)
    assert history[-1] < history[0] * 0.5


def test_train_updates_every_parameter():
    """A parameter left un-updated means its gradient never arrived."""
    X, y = make_ring(n=16)
    rng = np.random.RandomState(5)
    model = MLP([2, 12, 12, 2], rng=rng)
    before = [p.data.copy() for p in model.parameters()]
    train(model, X, y, lr=0.2, num_epochs=100)
    for old, p in zip(before, model.parameters()):
        assert not np.allclose(old, p.data)


def test_mlp_fits_the_ring():
    """The whole point: a trained MLP should reproduce the shape."""
    n = 24
    X, y = make_ring(n)
    rng = np.random.RandomState(0)
    model = MLP([2, 32, 32, 2], rng=rng)

    assert iou(predict(model, X), y) < 0.5
    train(model, X, y, lr=0.2, num_epochs=1500)
    assert iou(predict(model, X), y) > 0.9


def test_network_without_hidden_layer_fails():
    """No hidden layer means no non-linearity, and the ring is unreachable."""
    n = 24
    X, y = make_ring(n)
    rng = np.random.RandomState(6)
    model = MLP([2, 2], rng=rng)
    train(model, X, y, lr=0.2, num_epochs=1500)
    assert iou(predict(model, X), y) < 0.5
