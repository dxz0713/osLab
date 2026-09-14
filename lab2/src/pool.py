from __future__ import annotations

import numpy as np

from src.im2col import col2im, conv_output_size, im2col


def max_pool_forward(x, pool_param):
    """Max pooling forward pass, implemented with im2col.

    Reshaping so that each pooling window becomes a row lets the whole layer
    reduce to a single argmax along one axis. Pooling acts per channel, so
    the channel dimension is folded into the batch dimension first.

    Inputs:
    - x: (N, C, H, W)
    - pool_param: dict with 'pool_height', 'pool_width', 'stride'

    Returns a tuple of:
    - out: (N, C, out_h, out_w)
    - cache: (x, pool_param, argmax)
    """
    PH, PW = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]
    N, C, H, W = x.shape
    out_h, out_w = conv_output_size(H, W, PH, PW, stride, 0)

    col = im2col(x.reshape(N * C, 1, H, W), PH, PW, stride, 0)
    out = np.max(col, axis=1).reshape(N, C, out_h, out_w)
    argmax = np.argmax(col, axis=1).reshape(N, C, out_h, out_w)

    return out, (x, pool_param, argmax)


def max_pool_backward(dout, cache):
    """Max pooling backward pass.

    Pooling routes the entire upstream gradient to whichever input was the
    maximum, and gives zero to all the others. Positions that never won any
    window receive no gradient at all, which is why deep stacks of pooling
    layers can starve early features.

    Returns:
    - dx: same shape as the pooling input
    """
    x, pool_param, argmax = cache
    PH, PW = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]
    N, C, H, W = x.shape
    out_h, out_w = conv_output_size(H, W, PH, PW, stride, 0)

    dcol_sample = np.zeros(
        (N * C, out_h * out_w, PH * PW), dtype=dout.dtype
    )
    sample_argmax = argmax.reshape(N * C, out_h * out_w)
    sample_dout = dout.reshape(N * C, out_h * out_w)
    sample_indices = np.arange(N * C)[:, None], np.arange(out_h * out_w)
    dcol_sample[sample_indices[0], sample_indices[1], sample_argmax] = sample_dout
    dcol = dcol_sample.reshape(N * C * out_h * out_w, PH * PW)
    dx = col2im(dcol, (N * C, 1, H, W), PH, PW, stride, 0).reshape(x.shape)

    return dx



def receptive_field(layers):
    """Receptive field size of one output unit of a stacked network.

    Applies the standard recursion from the output back to the input:
        r_{l-1} = (r_l - 1) * stride_l + kernel_l
    starting from r = 1. Also returns the cumulative stride, which is how
    many input pixels one step in the output corresponds to.

    Inputs:
    - layers: list of (kernel_size, stride) tuples in forward order

    Returns a tuple of:
    - r: receptive field size in input pixels
    - jump: cumulative stride
    """
    r = 1
    jump = 1
    for kernel_size, stride in reversed(layers):
        r = (r - 1) * stride + kernel_size
        jump *= stride
    return r, jump



def count_conv_parameters(in_channels, out_channels, kernel_size, use_bias=True):
    """Number of learnable parameters in one convolution layer.

    The count is independent of the input's spatial size, which is exactly
    the point of weight sharing: a fully connected layer over the same input
    would need a separate weight per pixel.
    """
    weights = out_channels * in_channels * kernel_size * kernel_size
    bias = out_channels if use_bias else 0
    return weights + bias
