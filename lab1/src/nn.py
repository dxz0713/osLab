from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.tensor import Tensor, softmax_cross_entropy


class Linear:
    """A fully connected layer: out = X @ W + b.

    Mirrors torch.nn.Linear, except that the weight is stored as
    (in_features, out_features) so that the forward pass reads left to
    right with no transpose.

    Attributes:
        W: Tensor of shape (in_features, out_features).
        b: Tensor of shape (out_features,).
    """

    W: Tensor
    b: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.RandomState | None = None,
    ) -> None:
        """Create the layer and initialize its parameters.

        Weights use Kaiming (He) initialization -- normal noise scaled by
        sqrt(2 / in_features) -- which keeps the signal from shrinking as it
        passes through a stack of ReLU layers. Biases start at zero.

        Args:
            in_features: length of each input vector.
            out_features: length of each output vector.
            rng: source of randomness. Draw the weights from it rather than
                from np.random, so that a fixed seed gives a fixed network.
        """
        rng = np.random.RandomState() if rng is None else rng

        self.W = Tensor(rng.normal(0, np.sqrt(2 / in_features), (in_features, out_features)))
        self.b = Tensor(np.zeros((out_features,)))


    def __call__(self, x: Tensor) -> Tensor:
        """Run the forward pass.

        Args:
            x: Tensor of shape (N, in_features).

        Returns:
            Tensor of shape (N, out_features).
        """
        return x @ self.W + self.b

    def parameters(self) -> list[Tensor]:
        return [self.W, self.b]


class MLP:
    """A stack of Linear layers with ReLU between them.

    The last layer is left un-activated: it produces the logits that
    softmax_cross_entropy expects.

    Attributes:
        layers: the Linear layers, in forward order. len(sizes) - 1 of them.
    """

    layers: list[Linear]

    def __init__(self, sizes: Sequence[int], rng: np.random.RandomState | None = None) -> None:
        """Build the layer stack.

        Args:
            sizes: widths from input to output, e.g. [2, 32, 32, 2] for a
                network taking 2-D coordinates and emitting 2 class scores.
                Consecutive pairs give each layer's in/out features.
            rng: source of randomness, passed through to each layer.
        """
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng))

    def __call__(self, x: Tensor) -> Tensor:
        """Run the forward pass.

        Args:
            x: Tensor of shape (N, sizes[0]).

        Returns:
            Tensor of shape (N, sizes[-1]) holding logits -- unnormalized
            and unactivated.
        """
        for layer in self.layers:
            x = layer(x)
            if layer is not self.layers[-1]:
                x = x.relu()
        return x

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def predict(model: MLP, X: NDArray[np.float64]) -> NDArray[np.int64]:
    """Pick the highest-scoring class for every row of X.

    Args:
        model: an MLP.
        X: float array of shape (N, D). A plain array, not a Tensor.

    Returns:
        Integer array of shape (N,) holding predicted class indices.
    """
    logits = model(Tensor(X))
    return np.argmax(logits.data, axis=1)


def train(
    model: MLP,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    lr: float,
    num_epochs: int,
    log_every: int = 0,
) -> NDArray[np.float64]:
    """Fit the model by full-batch gradient descent.

    One epoch is: zero every parameter's gradient, run the forward pass,
    compute the loss, call backward, then step every parameter against its
    gradient. This is the same four-line rhythm a PyTorch training loop has.
    Zeroing first is not optional -- grads accumulate with +=, so skipping
    it makes every epoch inherit the last one's gradient.

    Args:
        model: an MLP.
        X: float array of shape (N, D).
        y: integer array of shape (N,) holding class indices.
        lr: learning rate, a positive float.
        num_epochs: number of full-batch steps to take.
        log_every: if positive, print the loss every this many epochs.

    Returns:
        Float array of shape (num_epochs,): the loss at each epoch, each
        recorded *before* the corresponding update, so history[0] is the
        loss of the initial parameters.
    """
    history = np.empty((num_epochs,), dtype=np.float64)
    for epoch in range(num_epochs):
        for p in model.parameters():
            p.grad = np.zeros_like(p.data)

        logits = model(Tensor(X))
        loss = softmax_cross_entropy(logits, y)
        history[epoch] = loss.data.item()

        loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad

        if log_every > 0 and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch + 1}: loss {history[epoch]}")

    return history
