from __future__ import annotations

import numpy as np


def he_conv(fan_in_channels, fan_out_channels, kernel_size, rng):
    """He/Kaiming normal initialization for a conv weight tensor.

    A conv weight of shape (F, C, k, k) behaves like a dense matrix with
    fan_in = C * k * k, because each output value is a dot product over the
    C * k * k values inside one window. The sample is drawn from
    N(0, 2 / fan_in).

    Returns an array of shape (fan_out_channels, fan_in_channels, k, k).
    """
    fan_in = fan_in_channels * kernel_size * kernel_size
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_out_channels, fan_in_channels, kernel_size, kernel_size))
