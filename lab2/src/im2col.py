from __future__ import annotations

import numpy as np


def conv_output_size(H, W, HH, WW, stride, pad):
    """Spatial size of a convolution output.

    H' = 1 + (H + 2 * pad - HH) // stride, and likewise for W'.

    Returns a tuple of (out_h, out_w).
    """
    out_h = 1 + (H + 2 * pad - HH) // stride
    out_w = 1 + (W + 2 * pad - WW) // stride
    return out_h, out_w



def im2col(x, HH, WW, stride, pad):
    """Rearrange every convolution patch into a row.

    This is the trick that turns convolution into a single matrix multiply.
    Each row of the result holds one flattened patch of the (padded) input,
    covering all C channels, so multiplying by the flattened filters computes
    every output position at once.

    The loop below runs over kernel positions only, i.e. HH * WW iterations,
    which is small. All of N and the output spatial positions are handled by
    strided slicing, so this stays fast even though it is written as a loop.

    Inputs:
    - x: (N, C, H, W)
    - HH, WW: filter height and width
    - stride, pad: convolution parameters

    Returns:
    - col: (N * out_h * out_w, C * HH * WW)
    """
    N, C, H, W = x.shape
    out_h, out_w = conv_output_size(H, W, HH, WW, stride, pad)
    col = np.zeros((N * out_h * out_w, C * HH * WW), dtype=x.dtype)
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
    )

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + HH
            w_start = j * stride
            w_end = w_start + WW
            col[i * out_w + j::out_h * out_w] = x_padded[
                :, :, h_start:h_end, w_start:w_end
            ].reshape(N, -1)

    return col


def col2im(col, x_shape, HH, WW, stride, pad):
    """Inverse of im2col in the sense required by backpropagation.

    Note that this is not a true inverse: when patches overlap, a single input
    pixel appears in several rows of col, and its gradient is the sum of all
    those contributions. Accumulating with += rather than assigning is
    therefore essential.

    Inputs:
    - col: (N * out_h * out_w, C * HH * WW)
    - x_shape: the shape (N, C, H, W) of the original input

    Returns:
    - x: (N, C, H, W)
    """
    N, C, H, W = x_shape
    out_h, out_w = conv_output_size(H, W, HH, WW, stride, pad)
    x = np.zeros((N, C, H, W), dtype=col.dtype)
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
    )
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + HH
            w_start = j * stride
            w_end = w_start + WW
            x_padded[:, :, h_start:h_end, w_start:w_end] += col[
                i * out_w + j :: out_h * out_w
            ].reshape(N, C, HH, WW)

    return x_padded[:, :, pad : pad + H, pad : pad + W]