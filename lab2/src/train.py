from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.cnn import ConvNet
from src.layers import softmax_loss
from src.optim import adam, sgd, sgd_momentum


@dataclass
class TrainConfig:
    """Hyperparameters for Task 3.

    Students are expected to tweak these (especially epochs, lr, and the
    architecture knobs) to push test accuracy higher.
    """

    num_blocks: int = 2
    base_filters: int = 32
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    optimizer: str = "adam"
    num_val: int = 1000
    seed: int = 0


def predict(net, X, batch_size=200):
    """Predict class labels for input data in batches to manage memory usage."""
    return net.predict(X, batch_size=batch_size)


def evaluate(net, X, y, batch_size=200, verbose=False, description="evaluation"):
    """Compute the average loss and accuracy of the network on a dataset.

    Args:
        net: A ConvNet instance.
        X: Input images of shape (N, 3, 32, 32).
        y: True labels of shape (N,), integers in [0, 9].
        batch_size: Number of samples to process at once. Default is 200.

    Returns:
        avg_loss: Average cross-entropy loss over the entire dataset (float).
        accuracy: Fraction of correctly classified samples (float between 0 and 1).

    """
    total_loss = 0.0
    correct = 0
    num_samples = X.shape[0]

    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, num_samples, batch_size), start=1):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]
        # Evaluation only needs a forward pass. Calling net.loss with labels
        # also computes a full backward pass, then predict would repeat the
        # forward pass again.
        scores = net.loss(X_batch)
        loss, _ = softmax_loss(scores, y_batch)
        total_loss += loss * len(y_batch)
        correct += np.count_nonzero(np.argmax(scores, axis=1) == y_batch)
        if verbose and batch_idx % 10 == 0:
            print(
                f"{description}: Batch {batch_idx}/{num_batches}",
                flush=True,
            )

    return total_loss / num_samples, correct / num_samples


def train(X, y, cfg: TrainConfig, X_val=None, y_val=None, verbose=True):
    """Minibatch training loop for Task 3.

    Returns a tuple of (net, history). The history dict holds per-epoch
    lists: train_loss, train_acc, and (if a validation set is given)
    val_loss, val_acc.

    Args:
        X: Training images of shape (N, 3, 32, 32).
        y: Training labels of shape (N,), integers in [0, 9].
        cfg: TrainConfig object containing all hyperparameters.
        X_val: Optional validation images of shape (M, 3, 32, 32).
        y_val: Optional validation labels of shape (M,).
        verbose: If True, print progress after each epoch.

    Returns:
        net: The trained ConvNet instance.
        history: A dict with keys:
            - "train_loss": list of average training loss per epoch
            - "train_acc": list of training accuracy per epoch
            - "val_loss": (if X_val provided) list of validation loss per epoch
            - "val_acc": (if X_val provided) list of validation accuracy per epoch

    """
    rng = np.random.RandomState(cfg.seed)
    net = ConvNet(num_classes=10, num_blocks=cfg.num_blocks, base_filters=cfg.base_filters, seed=cfg.seed)

    update = {"sgd": sgd, "sgd_momentum": sgd_momentum, "adam": adam}[cfg.optimizer]
    configs = {name: {"learning_rate": cfg.lr} for name in net.params}

    history = {"train_loss": [], "train_acc": []}

    if X_val is not None and y_val is not None:
        history["val_loss"] = []
        history["val_acc"] = []

    for epoch in range(cfg.epochs):
        # Shuffle the training data at the start of each epoch
        perm = rng.permutation(X.shape[0])
        X_shuffled, y_shuffled = X[perm], y[perm]

        # Mini-batch training
        num_batches = (X.shape[0] + cfg.batch_size - 1) // cfg.batch_size
        for batch_idx, i in enumerate(range(0, X.shape[0], cfg.batch_size), start=1):
            X_batch = X_shuffled[i:i + cfg.batch_size]
            y_batch = y_shuffled[i:i + cfg.batch_size]

            loss, grads = net.loss(X_batch, y_batch)

            # Update parameters using the chosen optimizer
            for name in net.params:
                net.params[name], configs[name] = update(net.params[name], grads[name], configs[name])

            if verbose and batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs}, "
                    f"Batch {batch_idx}/{num_batches}: loss={loss:.4f}"
                )

        # Evaluate on training set
        eval_batch_size = max(cfg.batch_size, 256)
        train_loss, train_acc = evaluate(
            net, X, y, batch_size=eval_batch_size,
            verbose=verbose, description="train evaluation",
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if X_val is not None and y_val is not None:
            val_loss, val_acc = evaluate(
                net, X_val, y_val, batch_size=eval_batch_size,
                verbose=verbose, description="validation",
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if verbose:
            msg = f"Epoch {epoch + 1}/{cfg.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
            if X_val is not None and y_val is not None:
                msg += f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            print(msg)

    return net, history
