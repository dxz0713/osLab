import time

import numpy as np
import pytest

from src.conv import conv_backward, conv_forward, conv_forward_naive
from src.im2col import col2im, conv_output_size, im2col
from tests.gradient_check import numeric_gradient, rel_error


def test_conv_output_size_formula():
    assert conv_output_size(7, 7, 3, 3, 1, 0) == (5, 5)
    assert conv_output_size(7, 7, 3, 3, 1, 1) == (7, 7)
    assert conv_output_size(7, 7, 3, 3, 2, 1) == (4, 4)
    assert conv_output_size(32, 32, 5, 5, 1, 2) == (32, 32)
    assert conv_output_size(8, 6, 2, 3, 2, 0) == (4, 2)


def test_im2col_shape():
    rng = np.random.RandomState(0)
    x = rng.randn(4, 3, 8, 8)
    col = im2col(x, 3, 3, 1, 1)
    assert col.shape == (4 * 8 * 8, 3 * 3 * 3)


def test_im2col_extracts_correct_patch():
    """A hand-checkable case: single sample, single channel, no padding."""
    x = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
    col = im2col(x, 2, 2, 2, 0)
    assert col.shape == (4, 4)
    # top-left window is [[0,1],[4,5]]
    assert np.allclose(col[0], np.array([0.0, 1.0, 4.0, 5.0]))
    # top-right window is [[2,3],[6,7]]
    assert np.allclose(col[1], np.array([2.0, 3.0, 6.0, 7.0]))
    # bottom-right window is [[10,11],[14,15]]
    assert np.allclose(col[3], np.array([10.0, 11.0, 14.0, 15.0]))


def test_im2col_applies_zero_padding():
    x = np.ones((1, 1, 2, 2))
    col = im2col(x, 3, 3, 1, 1)
    assert col.shape == (4, 9)
    # centre of each 3x3 window is a real pixel, corners are padding
    assert col[0, 4] == 1.0
    assert col[0, 0] == 0.0


def test_im2col_matches_manual_slicing():
    rng = np.random.RandomState(1)
    x = rng.randn(2, 3, 5, 5)
    HH = WW = 3
    stride, pad = 1, 0
    col = im2col(x, HH, WW, stride, pad)
    out_h, out_w = conv_output_size(5, 5, HH, WW, stride, pad)

    row = 0
    for n in range(2):
        for i in range(out_h):
            for j in range(out_w):
                patch = x[n, :, i : i + HH, j : j + WW].reshape(-1)
                assert np.allclose(col[row], patch)
                row += 1


def test_col2im_accumulates_overlaps():
    """Overlapping patches must sum, not overwrite."""
    x_shape = (1, 1, 3, 3)
    col = np.ones((4, 4))  # 2x2 kernel, stride 1 -> 4 patches
    out = col2im(col, x_shape, 2, 2, 1, 0)
    # the centre pixel belongs to all four windows
    assert out[0, 0, 1, 1] == pytest.approx(4.0)
    # corners belong to exactly one window
    assert out[0, 0, 0, 0] == pytest.approx(1.0)
    assert out[0, 0, 2, 2] == pytest.approx(1.0)


def test_col2im_is_transpose_of_im2col():
    """<im2col(x), c> must equal <x, col2im(c)> for all x and c."""
    rng = np.random.RandomState(2)
    for stride, pad, HH in [(1, 0, 3), (1, 1, 3), (2, 1, 3), (2, 0, 2)]:
        x = rng.randn(2, 3, 6, 6)
        col = im2col(x, HH, HH, stride, pad)
        c = rng.randn(*col.shape)
        lhs = float(np.sum(col * c))
        rhs = float(np.sum(x * col2im(c, x.shape, HH, HH, stride, pad)))
        assert lhs == pytest.approx(rhs, rel=1e-10)


def test_conv_forward_naive_hand_computed():
    """One 2x2 filter on a 3x3 input, stride 1, no padding."""
    x = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    b = np.array([0.5])
    out, _ = conv_forward_naive(x, w, b, {"stride": 1, "pad": 0})
    # each output is x[i,j] - x[i+1,j+1] + 0.5
    expected = np.array([[[[1 - 5 + 0.5, 2 - 6 + 0.5], [4 - 8 + 0.5, 5 - 9 + 0.5]]]])
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out, expected)


def test_conv_forward_matches_naive():
    """The whole point of im2col: identical results, different speed."""
    rng = np.random.RandomState(3)
    for stride, pad in [(1, 0), (1, 1), (2, 0), (2, 1), (3, 2)]:
        x = rng.randn(3, 2, 7, 8)
        w = rng.randn(4, 2, 3, 3)
        b = rng.randn(4)
        conv_param = {"stride": stride, "pad": pad}
        naive, _ = conv_forward_naive(x, w, b, conv_param)
        fast, _ = conv_forward(x, w, b, conv_param)
        assert fast.shape == naive.shape
        assert np.allclose(fast, naive, atol=1e-12)


def test_conv_forward_shapes():
    rng = np.random.RandomState(4)
    x = rng.randn(5, 3, 32, 32)
    w = rng.randn(8, 3, 5, 5)
    b = rng.randn(8)
    out, _ = conv_forward(x, w, b, {"stride": 1, "pad": 2})
    assert out.shape == (5, 8, 32, 32)


def test_conv_bias_shifts_each_filter():
    rng = np.random.RandomState(5)
    x = rng.randn(2, 1, 5, 5)
    w = rng.randn(3, 1, 3, 3)
    zero_b, some_b = np.zeros(3), np.array([1.0, -2.0, 0.5])
    out0, _ = conv_forward(x, w, zero_b, {"stride": 1, "pad": 1})
    out1, _ = conv_forward(x, w, some_b, {"stride": 1, "pad": 1})
    for f in range(3):
        assert np.allclose(out1[:, f] - out0[:, f], some_b[f])


def test_conv_backward_gradients():
    rng = np.random.RandomState(6)
    for stride, pad in [(1, 1), (1, 0), (2, 1), (2, 0)]:
        x = rng.randn(2, 3, 6, 6)
        w = rng.randn(4, 3, 3, 3)
        b = rng.randn(4)
        conv_param = {"stride": stride, "pad": pad}

        out, cache = conv_forward(x, w, b, conv_param)
        dout = rng.randn(*out.shape)
        dx, dw, db = conv_backward(dout, cache)

        assert dx.shape == x.shape
        assert dw.shape == w.shape
        assert db.shape == b.shape

        def f():
            return float(np.sum(conv_forward(x, w, b, conv_param)[0] * dout))

        assert rel_error(dx, numeric_gradient(f, x)) < 1e-6
        assert rel_error(dw, numeric_gradient(f, w)) < 1e-6
        assert rel_error(db, numeric_gradient(f, b)) < 1e-8


def test_conv_backward_non_square_input():
    rng = np.random.RandomState(7)
    x = rng.randn(2, 2, 5, 8)
    w = rng.randn(3, 2, 3, 3)
    b = rng.randn(3)
    conv_param = {"stride": 1, "pad": 1}
    out, cache = conv_forward(x, w, b, conv_param)
    dout = rng.randn(*out.shape)
    dx, dw, db = conv_backward(dout, cache)

    def f():
        return float(np.sum(conv_forward(x, w, b, conv_param)[0] * dout))

    assert rel_error(dx, numeric_gradient(f, x)) < 1e-6
    assert rel_error(dw, numeric_gradient(f, w)) < 1e-6


def test_im2col_is_faster_than_naive():
    """Not a correctness test, but the reason im2col exists."""
    rng = np.random.RandomState(8)
    x = rng.randn(8, 3, 24, 24)
    w = rng.randn(12, 3, 3, 3)
    b = rng.randn(12)
    conv_param = {"stride": 1, "pad": 1}

    t0 = time.perf_counter()
    conv_forward_naive(x, w, b, conv_param)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    conv_forward(x, w, b, conv_param)
    t_fast = time.perf_counter() - t0

    assert t_fast < t_naive, f"im2col ({t_fast:.4f}s) should beat naive ({t_naive:.4f}s)"
