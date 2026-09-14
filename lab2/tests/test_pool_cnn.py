import numpy as np
import pytest

from src.conv import conv_backward, conv_forward
from src.layers import affine_backward, affine_forward, relu_backward, relu_forward, softmax_loss
from src.pool import (
    count_conv_parameters,
    max_pool_backward,
    max_pool_forward,
    receptive_field,
)
from tests.gradient_check import numeric_gradient, rel_error


# ===================================================================== pooling
def test_max_pool_forward_hand_computed():
    x = np.array(
        [[[[1.0, 5.0, 2.0, 3.0], [4.0, 2.0, 1.0, 0.0], [0.0, 1.0, 9.0, 2.0], [3.0, 2.0, 4.0, 8.0]]]]
    )
    out, _ = max_pool_forward(x, {"pool_height": 2, "pool_width": 2, "stride": 2})
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out, np.array([[[[5.0, 3.0], [3.0, 9.0]]]]))


def test_max_pool_is_per_channel():
    """Pooling never mixes channels."""
    rng = np.random.RandomState(0)
    x = rng.randn(2, 4, 6, 6)
    out, _ = max_pool_forward(x, {"pool_height": 2, "pool_width": 2, "stride": 2})
    assert out.shape == (2, 4, 3, 3)
    for n in range(2):
        for c in range(4):
            single = x[n : n + 1, c : c + 1]
            single_out, _ = max_pool_forward(
                single, {"pool_height": 2, "pool_width": 2, "stride": 2}
            )
            assert np.allclose(out[n, c], single_out[0, 0])


def test_max_pool_output_shapes():
    rng = np.random.RandomState(1)
    x = rng.randn(3, 2, 8, 8)
    for ph, stride, expected in [(2, 2, 4), (4, 4, 2), (2, 1, 7), (3, 1, 6)]:
        out, _ = max_pool_forward(
            x, {"pool_height": ph, "pool_width": ph, "stride": stride}
        )
        assert out.shape == (3, 2, expected, expected)


def test_max_pool_backward_routes_to_argmax():
    """Gradient goes entirely to the winning input, zero elsewhere."""
    x = np.array([[[[1.0, 5.0], [4.0, 2.0]]]])
    pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
    out, cache = max_pool_forward(x, pool_param)
    assert out[0, 0, 0, 0] == 5.0

    dx = max_pool_backward(np.array([[[[1.0]]]]), cache)
    assert np.allclose(dx, np.array([[[[0.0, 1.0], [0.0, 0.0]]]]))


def test_max_pool_backward_gradients():
    rng = np.random.RandomState(2)
    for ph, stride in [(2, 2), (3, 1), (2, 1), (4, 4)]:
        x = rng.randn(2, 3, 8, 8)
        pool_param = {"pool_height": ph, "pool_width": ph, "stride": stride}
        out, cache = max_pool_forward(x, pool_param)
        dout = rng.randn(*out.shape)
        dx = max_pool_backward(dout, cache)
        assert dx.shape == x.shape

        def f():
            return float(np.sum(max_pool_forward(x, pool_param)[0] * dout))

        assert rel_error(dx, numeric_gradient(f, x)) < 1e-6


# ============================================================= receptive field
def test_receptive_field_single_layer():
    assert receptive_field([(3, 1)]) == (3, 1)
    assert receptive_field([(5, 1)]) == (5, 1)
    assert receptive_field([(3, 2)]) == (3, 2)


def test_receptive_field_stacked_3x3_equals_one_5x5():
    """Two stacked 3x3 convolutions see the same 5x5 region as one 5x5 conv,
    but with fewer parameters. This is the core argument of VGG."""
    r_two, _ = receptive_field([(3, 1), (3, 1)])
    r_one, _ = receptive_field([(5, 1)])
    assert r_two == r_one == 5

    params_two = 2 * count_conv_parameters(64, 64, 3, use_bias=False)
    params_one = count_conv_parameters(64, 64, 5, use_bias=False)
    assert params_two < params_one


def test_receptive_field_grows_with_pooling():
    """conv3 -> pool2 -> conv3 sees 8 input pixels.

    Traced by hand from the output backwards: one output unit needs 3 pooled
    positions, which need 6 positions after the first conv, which need 8
    input pixels.
    """
    r, jump = receptive_field([(3, 1), (2, 2), (3, 1)])
    assert r == 8
    assert jump == 2


def test_receptive_field_order_matters():
    """An asymmetric stack distinguishes the correct recursion direction.

    conv3s1 -> conv3s2 has receptive field 5, verified by tracing the
    dependency intervals by hand. Applying the recursion in forward order
    instead of from the output backwards would give 7.
    """
    assert receptive_field([(3, 1), (3, 2)]) == (5, 2)
    assert receptive_field([(3, 2), (3, 1)]) == (7, 2)


def test_receptive_field_deep_stack():
    layers = [(3, 1)] * 5
    r, jump = receptive_field(layers)
    assert r == 11  # 1 + 5 * 2
    assert jump == 1


def test_conv_parameter_count_is_independent_of_input_size():
    """Weight sharing: a conv layer's size does not depend on the image size."""
    assert count_conv_parameters(3, 16, 3) == 16 * 3 * 3 * 3 + 16
    assert count_conv_parameters(3, 16, 3, use_bias=False) == 432

    # a fully connected layer over a 32x32x3 image producing 16x32x32 outputs
    fc_params = (3 * 32 * 32) * (16 * 32 * 32)
    assert count_conv_parameters(3, 16, 3) < fc_params / 1000


def test_conv_is_translation_equivariant():
    """Shifting the input shifts the output by the same amount.

    This is the property that makes weight sharing sensible: the same feature
    detector works wherever the feature happens to appear.
    """
    rng = np.random.RandomState(4)
    x = np.zeros((1, 1, 9, 9))
    x[0, 0, 2:5, 2:5] = rng.randn(3, 3)
    w = rng.randn(2, 1, 3, 3)
    b = np.zeros(2)
    conv_param = {"stride": 1, "pad": 0}

    out_a, _ = conv_forward(x, w, b, conv_param)
    shifted = np.roll(x, shift=2, axis=3)
    out_b, _ = conv_forward(shifted, w, b, conv_param)

    assert np.allclose(out_b[:, :, :, 2:], out_a[:, :, :, :-2], atol=1e-12)


# ================================================================== small CNN
def cnn_forward(x, params, conv_param, pool_param):
    """conv -> relu -> pool -> affine -> scores."""
    w1, b1, w2, b2 = params
    a1, conv_cache = conv_forward(x, w1, b1, conv_param)
    r1, relu_cache = relu_forward(a1)
    p1, pool_cache = max_pool_forward(r1, pool_param)
    scores, aff_cache = affine_forward(p1, w2, b2)
    return scores, (conv_cache, relu_cache, pool_cache, aff_cache, p1.shape)


def cnn_backward(dscores, cache):
    conv_cache, relu_cache, pool_cache, aff_cache, pooled_shape = cache
    dp1, dw2, db2 = affine_backward(dscores, aff_cache)
    dr1 = max_pool_backward(dp1.reshape(pooled_shape), pool_cache)
    da1 = relu_backward(dr1, relu_cache)
    dx, dw1, db1 = conv_backward(da1, conv_cache)
    return dx, [dw1, db1, dw2, db2]


def make_cnn_params(rng, C=1, F=4, HH=3, pooled_dim=None, num_classes=3):
    w1 = rng.randn(F, C, HH, HH) * 0.3
    b1 = np.zeros(F)
    w2 = rng.randn(pooled_dim, num_classes) * 0.3
    b2 = np.zeros(num_classes)
    return [w1, b1, w2, b2]


def test_cnn_end_to_end_gradients():
    """Full conv-relu-pool-affine stack checked against numeric gradients."""
    rng = np.random.RandomState(3)
    conv_param = {"stride": 1, "pad": 1}
    pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

    x = rng.randn(4, 2, 8, 8)
    y = rng.randint(0, 3, size=4)
    params = make_cnn_params(rng, C=2, F=3, pooled_dim=3 * 4 * 4)

    scores, cache = cnn_forward(x, params, conv_param, pool_param)
    assert scores.shape == (4, 3)
    _, dscores = softmax_loss(scores, y)
    _, grads = cnn_backward(dscores, cache)

    def loss_fn():
        s, _ = cnn_forward(x, params, conv_param, pool_param)
        return softmax_loss(s, y)[0]

    for p, g in zip(params, grads):
        assert rel_error(g, numeric_gradient(loss_fn, p)) < 1e-5


def test_cnn_learns_a_simple_visual_pattern():
    """Classify images by which oriented bar they contain.

    A convolution should pick this up easily, because the discriminative
    feature is local and appears at varying positions.
    """
    rng = np.random.RandomState(0)
    N_PER = 40
    imgs, labels = [], []
    for cls in range(2):
        for _ in range(N_PER):
            img = rng.randn(1, 8, 8) * 0.1
            r, c = rng.randint(0, 5), rng.randint(0, 5)
            if cls == 0:
                img[0, r : r + 4, c] += 2.0  # vertical bar
            else:
                img[0, r, c : c + 4] += 2.0  # horizontal bar
            imgs.append(img)
            labels.append(cls)
    X = np.stack(imgs)
    y = np.array(labels)

    conv_param = {"stride": 1, "pad": 1}
    pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
    params = make_cnn_params(rng, C=1, F=4, pooled_dim=4 * 4 * 4, num_classes=2)

    losses = []
    for _ in range(120):
        scores, cache = cnn_forward(X, params, conv_param, pool_param)
        loss, dscores = softmax_loss(scores, y)
        losses.append(loss)
        _, grads = cnn_backward(dscores, cache)
        for p, g in zip(params, grads):
            p -= 0.05 * g

    scores, _ = cnn_forward(X, params, conv_param, pool_param)
    acc = (np.argmax(scores, axis=1) == y).mean()
    assert losses[-1] < losses[0] * 0.5
    assert acc > 0.9, f"accuracy only {acc:.2%}"
