from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_ring(
    n: int = 32, r_inner: float = 0.35, r_outer: float = 0.65
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build the ring classification dataset.

    Lays an n-by-n grid over the square [-1, 1] x [-1, 1] and labels every
    grid point by whether it falls inside an annulus centred on the origin.
    Coordinates are scaled to [-1, 1] rather than left as raw indices: an
    input range of 0..n-1 is both large and off-centre, which makes plain
    gradient descent crawl.

    The ring is not linearly separable, and unlike a filled disc it cannot
    be carved out by a single closed boundary either -- the network has to
    learn an inside edge and an outside edge.

    Args:
        n: grid resolution; the dataset has n * n samples.
        r_inner: inner radius of the ring, in [-1, 1] coordinates.
        r_outer: outer radius of the ring. A point is labelled 1 when its
            distance from the origin lies in the closed interval
            [r_inner, r_outer].

    Returns:
        X: float array of shape (n * n, 2), one (x, y) coordinate per row.
            Rows must be ordered so that reshape(n, n) recovers the grid --
            that is how render() and visualize.py turn a flat array of
            per-point values back into a picture.
        y: array of shape (n * n,) and integer dtype; 1 inside the ring, 0
            outside. It must be integer, not bool: softmax_cross_entropy
            uses these values as class indices, and NumPy would read a bool
            array as a mask instead.
    """
    x = np.linspace(-1, 1, n)
    X = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
    r = np.linalg.norm(X, axis=1)
    y = ((r >= r_inner) & (r <= r_outer)).astype(np.int64)
    return X, y


def iou(y_pred: NDArray[np.int64], y_true: NDArray[np.int64]) -> float:
    """Intersection over union for the positive class.

    Accuracy is misleading here: the ring covers roughly a fifth of the
    grid, so a model that predicts 0 everywhere already scores about 80%.
    IoU only rewards overlap with the ring itself, so that degenerate
    solution scores 0.

    Args:
        y_pred: integer array of shape (N,) holding predicted labels.
        y_true: integer array of shape (N,) holding true labels.

    Returns:
        Python float in [0, 1].
    """
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    intersection = np.sum((y_pred == 1) & (y_true == 1))
    union = np.sum((y_pred == 1) | (y_true == 1))
    return float(intersection / union) if union else 1.0


def render(y: NDArray[np.int64], n: int) -> str:
    """Turn a flat label array back into a printable n-by-n picture.

    Args:
        y: array of shape (n * n,) holding 0/1 labels.
        n: grid resolution.

    Returns:
        A multi-line string, '#' for 1 and '.' for 0.
    """
    grid = np.asarray(y).reshape(n, n)
    return "\n".join("".join("#" if v else "." for v in row) for row in grid)
