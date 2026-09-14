import numpy as np
import pytest

from src.cnn import ConvNet
from src.optim import adam
from tests.gradient_check import numeric_gradient, rel_error


def make_net(**kwargs):
    return ConvNet(num_classes=10, seed=0, **kwargs)


def test_params_shapes_two_blocks():
    net = make_net(num_blocks=2, base_filters=32)
    shapes = {k: v.shape for k, v in net.params.items()}
    assert shapes["W0"] == (32, 3, 3, 3)
    assert shapes["b0"] == (32,)
    assert shapes["W1"] == (64, 32, 3, 3)
    assert shapes["b1"] == (64,)
    assert shapes["Wfc"] == (64 * 8 * 8, 10)
    assert shapes["bfc"] == (10,)


def test_params_shapes_three_blocks():
    net = make_net(num_blocks=3, base_filters=16)
    shapes = {k: v.shape for k, v in net.params.items()}
    assert shapes["W0"] == (16, 3, 3, 3)
    assert shapes["W1"] == (32, 16, 3, 3)
    assert shapes["W2"] == (64, 32, 3, 3)
    assert shapes["Wfc"] == (64 * 4 * 4, 10)


def test_forward_scores_shape():
    rng = np.random.RandomState(0)
    net = make_net()
    x = rng.rand(6, 3, 32, 32)
    scores = net.loss(x)
    assert scores.shape == (6, 10)
    assert np.all(np.isfinite(scores))


def test_gradients_match_numeric():
    """The full network's analytic gradients vs central differences."""
    rng = np.random.RandomState(1)
    net = ConvNet(num_blocks=2, base_filters=4, input_side=16, seed=0)
    x = rng.randn(3, 3, 16, 16)  # smaller spatial size keeps the check fast
    y = np.array([0, 4, 9])

    loss, grads = net.loss(x, y)

    def loss_fn():
        return net.loss(x, y)[0]

    # every parameter must receive a gradient of the right shape
    for name in net.params:
        assert grads[name].shape == net.params[name].shape, name

    # numeric check on a subset: biases (cheap) and a slice of each weight
    for name in ["b0", "b1", "bfc"]:
        assert rel_error(grads[name], numeric_gradient(loss_fn, net.params[name])) < 1e-6, name

    w = net.params["W0"]
    mask = np.zeros_like(w)
    mask[:, 0, 0, 0] = 1.0
    assert rel_error(grads["W0"] * mask, numeric_gradient(loss_fn, w) * mask) < 1e-6


def test_sanity_check_initial_loss():
    """On mean-centered input, the initial loss should be close to log(10).

    With properly scaled inputs the untrained network is confidently wrong
    in both directions, so the loss hovers around the chance level rather
    than exactly at it.
    """
    rng = np.random.RandomState(2)
    net = make_net()
    x = rng.rand(200, 3, 32, 32)
    x = x - x.mean(axis=0)  # same preprocessing as Task 2
    y = rng.randint(0, 10, size=200)
    loss, _ = net.loss(x, y)
    assert loss < 3.0 and loss > 1.8


def test_overfit_small_subset():
    """A correct CNN can memorize a handful of images quickly."""
    rng = np.random.RandomState(3)
    net = make_net(num_blocks=2, base_filters=16)
    X = rng.rand(20, 3, 32, 32)
    y = rng.randint(0, 10, size=20)

    configs = {k: {"learning_rate": 1e-3} for k in net.params}
    first_loss = None
    for _ in range(80):
        loss, grads = net.loss(X, y)
        first_loss = loss if first_loss is None else first_loss
        for name in net.params:
            net.params[name], configs[name] = adam(net.params[name], grads[name], configs[name])

    final_loss, _ = net.loss(X, y)
    assert final_loss < first_loss * 0.3
    acc = (np.argmax(net.loss(X), axis=1) == y).mean()
    assert acc > 0.8, f"accuracy only {acc:.2%}"
