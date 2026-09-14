"""Train the MLP on the ring and plot what it learned.

Run with:  uv run python visualize.py

Produces ring.png, three panels side by side: the target ring, what an
untrained network predicts, and what the same network predicts after
training. Nothing here is graded -- it is the payoff for getting the
three tasks right.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import iou, make_ring, render
from src.nn import MLP, predict, train

N_GRID = 48
EPOCHS = 3000
LR = 0.2


def main():
    X, y = make_ring(N_GRID)
    rng = np.random.RandomState(3)
    model = MLP([2, 32, 32, 2], rng=rng)

    before = predict(model, X)
    print(f"IoU before training: {iou(before, y):.3f}")

    history = train(model, X, y, lr=LR, num_epochs=EPOCHS, log_every=EPOCHS // 8)

    after = predict(model, X)
    print(f"IoU after training:  {iou(after, y):.3f}")
    print()
    print(render(after, N_GRID))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, grid, title in [
        (axes[0], y, "target"),
        (axes[1], before, "before training"),
        (axes[2], after, "after training"),
    ]:
        ax.imshow(grid.reshape(N_GRID, N_GRID), cmap="viridis", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[3].plot(history)
    axes[3].set_title("training loss")
    axes[3].set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig("ring.png", dpi=120)
    print("\nsaved ring.png")


if __name__ == "__main__":
    main()
