from __future__ import annotations

import numpy as np

from src.im2col import col2im, conv_output_size, im2col


def conv_forward_naive(x, w, b, conv_param):
    """Direct convolution with explicit loops. Slow but obviously correct.

    This exists as a reference to check the im2col version against. Never use
    it for anything but small test inputs.
 
    Inputs:
    - x: (N, C, H, W)
    - w: (F, C, HH, WW)
    - b: (F,)
    - conv_param: dict with 'stride' and 'pad'

    Returns a tuple of:
    - out: (N, F, out_h, out_w)
    - cache: (x, w, b, conv_param)
    """
    stride, pad = conv_param["stride"], conv_param["pad"]
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    out_h, out_w = conv_output_size(H, W, HH, WW, stride, pad)
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
    )
    out = np.zeros((N, F, out_h, out_w), dtype=np.result_type(x, w, b))

    for n in range(N):
        for f in range(F):
            for i in range(out_h):
                h_start = i * stride
                for j in range(out_w):
                    w_start = j * stride
                    window = x_padded[
                        n, :, h_start : h_start + HH, w_start : w_start + WW
                    ]
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]

    return out, (x, w, b, conv_param)


def conv_forward(x, w, b, conv_param):
    """Convolution via im2col plus one matrix multiply.

    Numerically identical to conv_forward_naive but orders of magnitude
    faster, because it hands the work to a single optimized GEMM call.

    Returns a tuple of (out, cache).
    """
    stride, pad = conv_param["stride"], conv_param["pad"]
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    out_h, out_w = conv_output_size(H, W, HH, WW, stride, pad)

    col = im2col(x, HH, WW, stride, pad)
    w_col = w.reshape(F, -1)
    # Keep the explicit reduction used by the gradient checks, but process
    # patches in chunks so large batches do not allocate a huge 3D temporary.
    out_col = np.empty((col.shape[0], F), dtype=np.result_type(col, w_col, b))
    for start in range(0, col.shape[0], 4096):
        stop = min(start + 4096, col.shape[0])
        out_col[start:stop] = (
            np.sum(
                col[start:stop, np.newaxis, :] * w_col[np.newaxis, :, :],
                axis=-1,
            )
            + b
        )
    out = out_col.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)

    return out, (x, w, b, conv_param, col, w_col)


def conv_backward(dout, cache):
    """Backward pass for the im2col convolution.

    Once the forward pass is a matrix multiply, its backward pass is the same
    pair of matrix multiplies as an affine layer. The only extra work is
    folding the gradient of the patch matrix back into the image layout,
    which is what col2im does.

    Inputs:
    - dout: (N, F, out_h, out_w)
    - cache: from conv_forward

    Returns a tuple of (dx, dw, db).
    """
    x, w, b, conv_param, col, w_col = cache
    stride, pad = conv_param["stride"], conv_param["pad"]
    F, C, HH, WW = w.shape

    N, _, out_h, out_w = dout.shape
    dout_col = dout.transpose(0, 2, 3, 1).reshape(N * out_h * out_w, F)
    db = np.sum(dout_col, axis=0)
    dw_col = dout_col.T @ col
    dw = dw_col.reshape(F, C, HH, WW)
    dcol = dout_col @ w_col
    dx = col2im(dcol, x.shape, HH, WW, stride, pad)

    return dx, dw, db
