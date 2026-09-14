import numpy as np
import pytest

from src.train import TrainConfig, evaluate, predict, train


def make_fake_cifar(rng, n_train=600, n_test=200):
    """Synthetic CIFAR-like data with a learnable class signal.

    Each class gets a fixed random template image; samples are the template
    plus noise. This is exactly the 'nearest template' problem a CNN solves
    easily, without needing the real dataset.
    """
    n_class = 10
    templates = rng.rand(n_class, 3, 32, 32)
    Xtr = np.stack([templates[i % n_class] for i in range(n_train)]) + 0.15 * rng.randn(
        n_train, 3, 32, 32
    )
    ytr = np.arange(n_train) % n_class
    Xte = np.stack([templates[i % n_class] for i in range(n_test)]) + 0.15 * rng.randn(
        n_test, 3, 32, 32
    )
    yte = np.arange(n_test) % n_class
    return np.clip(Xtr, 0, 1), ytr, np.clip(Xte, 0, 1), yte


@pytest.fixture(scope="module")
def fake_data():
    return make_fake_cifar(np.random.RandomState(0))


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.num_blocks == 2
    assert cfg.base_filters == 32
    assert cfg.epochs == 3
    assert cfg.batch_size == 128
    assert cfg.lr == 1e-3
    assert cfg.optimizer == "adam"
    assert cfg.num_val == 1000
    assert cfg.seed == 0


def test_predict_returns_labels(fake_data):
    Xtr, ytr, Xte, yte = fake_data
    from src.cnn import ConvNet

    net = ConvNet(num_classes=10, num_blocks=1, base_filters=8, seed=0)
    preds = predict(net, Xte, batch_size=64)
    assert preds.shape == (yte.shape[0],)
    assert preds.dtype in (np.int64, np.int32)
    assert preds.min() >= 0 and preds.max() < 10


def test_evaluate_returns_loss_and_acc(fake_data):
    Xtr, ytr, Xte, yte = fake_data
    from src.cnn import ConvNet

    net = ConvNet(num_classes=10, num_blocks=1, base_filters=8, seed=0)
    loss, acc = evaluate(net, Xte, yte, batch_size=100)
    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    # untrained net on 10 classes
    assert acc == pytest.approx(0.1, abs=0.08)


def test_train_reduces_loss(fake_data):
    Xtr, ytr, Xte, yte = fake_data
    cfg = TrainConfig(num_blocks=1, base_filters=8, epochs=2, batch_size=64, lr=1e-3)
    net, history = train(Xtr, ytr, cfg, X_val=None, verbose=False)

    assert len(history["train_loss"]) >= 2
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["train_acc"][-1] > 0.15  # beats chance


def test_train_with_validation(fake_data):
    Xtr, ytr, Xte, yte = fake_data
    cfg = TrainConfig(num_blocks=1, base_filters=8, epochs=2, batch_size=64, lr=1e-3)
    n_val = 100
    net, history = train(Xtr, ytr, cfg, X_val=Xtr[:n_val], y_val=ytr[:n_val], verbose=False)

    assert len(history["val_acc"]) == cfg.epochs
    assert all(0.0 <= a <= 1.0 for a in history["val_acc"])


def test_train_sgd_optimizer(fake_data):
    Xtr, ytr, Xte, yte = fake_data
    cfg = TrainConfig(
        num_blocks=1, base_filters=8, epochs=1, batch_size=64, lr=0.05, optimizer="sgd"
    )
    net, history = train(Xtr, ytr, cfg, X_val=None, verbose=False)
    assert len(history["train_loss"]) >= 1
