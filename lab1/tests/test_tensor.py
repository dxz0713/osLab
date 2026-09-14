import numpy as np
import pytest

from src.engine import Value
from src.gradient_check import numeric_gradient, rel_error
from src.tensor import Tensor, _unbroadcast, softmax_cross_entropy


def test_unbroadcast_removes_prepended_axes():
    """Axes that broadcasting added on the left are summed away entirely."""
    grad = np.ones((2, 5, 3))
    out = _unbroadcast(grad, (5, 3))
    assert out.shape == (5, 3)
    assert np.allclose(out, 2.0)


def test_unbroadcast_keeps_length_one_axes():
    """An axis that was length 1 stays length 1, it does not disappear."""
    grad = np.ones((4, 3))
    out = _unbroadcast(grad, (1, 3))
    assert out.shape == (1, 3)
    assert np.allclose(out, 4.0)


def test_unbroadcast_mixed_case():
    """Both rules at once: drop a prepended axis and collapse a 1-axis."""
    grad = np.ones((6, 4, 3))
    out = _unbroadcast(grad, (4, 1))
    assert out.shape == (4, 1)
    assert np.allclose(out, 18.0)


def test_unbroadcast_is_identity_on_matching_shape():
    rng = np.random.RandomState(11)
    grad = rng.randn(3, 7)
    out = _unbroadcast(grad, (3, 7))
    assert out.shape == (3, 7)
    assert np.allclose(out, grad)


def test_add_backward_general_broadcast():
    """(3, 1) + (1, 4) broadcasts both operands; both gradients must fit."""
    a = Tensor(np.zeros((3, 1)))
    b = Tensor(np.zeros((1, 4)))
    out = a + b
    assert out.data.shape == (3, 4)
    out.backward()
    assert a.grad.shape == (3, 1) and np.allclose(a.grad, 4.0)
    assert b.grad.shape == (1, 4) and np.allclose(b.grad, 3.0)


def test_add_forward_and_broadcast():
    a = Tensor(np.ones((4, 3)))
    b = Tensor(np.array([1.0, 2.0, 3.0]))
    out = a + b
    assert out.data.shape == (4, 3)
    assert np.allclose(out.data, np.array([2.0, 3.0, 4.0]))


def test_add_backward_sums_over_broadcast_axis():
    """A (C,) bias added to an (N, C) tensor gets N contributions back."""
    a = Tensor(np.zeros((5, 3)))
    b = Tensor(np.zeros(3))
    (a + b).backward()
    assert a.grad.shape == (5, 3)
    assert b.grad.shape == (3,)
    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, 5.0)


def test_mul_forward_and_broadcast():
    rng = np.random.RandomState(12)
    A, b = rng.randn(4, 3), rng.randn(3)
    out = Tensor(A) * Tensor(b)
    assert out.data.shape == (4, 3)
    assert np.allclose(out.data, A * b)


def test_mul_backward():
    """Local derivative is the other operand, then the same unbroadcast."""
    rng = np.random.RandomState(13)
    A, b = rng.randn(4, 3), rng.randn(3)
    ta, tb = Tensor(A), Tensor(b)
    (ta * tb).backward()
    assert ta.grad.shape == (4, 3)
    assert tb.grad.shape == (3,)
    assert np.allclose(ta.grad, np.broadcast_to(b, (4, 3)))
    assert np.allclose(tb.grad, A.sum(axis=0))


def test_mul_and_add_share_unbroadcast():
    """Both operators must route a (C,) operand's gradient back to (C,)."""
    rng = np.random.RandomState(14)
    X = rng.randn(6, 5)
    for build in [lambda a, b: a + b, lambda a, b: a * b]:
        ta, tb = Tensor(X), Tensor(rng.randn(5))
        build(ta, tb).backward()
        assert ta.grad.shape == (6, 5)
        assert tb.grad.shape == (5,)


def test_matmul_forward():
    rng = np.random.RandomState(0)
    A, B = rng.randn(4, 3), rng.randn(3, 2)
    out = Tensor(A) @ Tensor(B)
    assert out.data.shape == (4, 2)
    assert np.allclose(out.data, A @ B)


def test_matmul_backward():
    rng = np.random.RandomState(1)
    A, B = rng.randn(4, 3), rng.randn(3, 2)
    ta, tb = Tensor(A), Tensor(B)
    (ta @ tb).backward()
    assert ta.grad.shape == A.shape
    assert tb.grad.shape == B.shape
    assert np.allclose(ta.grad, np.ones((4, 2)) @ B.T)
    assert np.allclose(tb.grad, A.T @ np.ones((4, 2)))


def test_relu_forward():
    x = Tensor(np.array([[-2.0, 0.0, 3.0]]))
    assert np.allclose(x.relu().data, [[0.0, 0.0, 3.0]])


def test_relu_backward():
    """Zero on the negative side, and zero at the origin by convention."""
    x = Tensor(np.array([[-2.0, 0.0, 3.0]]))
    x.relu().backward()
    assert np.allclose(x.grad, [[0.0, 0.0, 1.0]])


def test_gradient_accumulates_when_reused():
    """out = a + a must give d(out)/da = 2, which requires += instead of =."""
    a = Tensor(np.ones((2, 2)))
    (a + a).backward()
    assert np.allclose(a.grad, 2.0)


def test_topological_order_is_respected():
    """Two paths from a to the output; a wrong order loses one of them."""
    rng = np.random.RandomState(2)
    A, W = rng.randn(3, 4), rng.randn(4, 4)
    a, w = Tensor(A), Tensor(W)
    ((a @ w) + a).backward()
    expected = np.ones((3, 4)) @ W.T + np.ones((3, 4))
    assert np.allclose(a.grad, expected)


def test_cross_entropy_uniform_and_confident():
    N, C = 7, 5
    loss = softmax_cross_entropy(Tensor(np.zeros((N, C))), np.arange(N) % C)
    assert float(loss.data) == pytest.approx(np.log(C))

    confident = softmax_cross_entropy(Tensor(np.array([[50.0, 0.0, 0.0]])), np.array([0]))
    assert float(confident.data) < 1e-10


def test_cross_entropy_numerical_stability():
    """Naive exp() would overflow to inf on these logits."""
    for scale in [1e3, 1e4, -1e4]:
        scores = np.array([[scale, scale - 1.0, scale + 2.0], [0.0, 1.0, 2.0]])
        logits = Tensor(scores)
        loss = softmax_cross_entropy(logits, np.array([2, 0]))
        assert np.isfinite(float(loss.data))
        loss.backward()
        assert np.all(np.isfinite(logits.grad))


def test_log_softmax_avoids_underflow():
    """Computing probs first and then log() would give -inf here.

    exp(-900) underflows to 0.0, so log(softmax(s)) loses the answer
    entirely. Subtracting the row max is not enough on its own -- the log
    has to be folded into the same expression.
    """
    scores = np.array([[0.0, -900.0], [-900.0, 0.0]])
    loss = softmax_cross_entropy(Tensor(scores), np.array([1, 0]))
    assert np.isfinite(float(loss.data))
    assert float(loss.data) == pytest.approx(900.0, rel=1e-9)


def test_cross_entropy_gradient():
    rng = np.random.RandomState(7)
    scores = rng.randn(12, 4) * 2
    y = rng.randint(0, 4, size=12)

    logits = Tensor(scores)
    loss = softmax_cross_entropy(logits, y)
    loss.backward()
    assert np.allclose(logits.grad.sum(axis=1), 0.0, atol=1e-12)

    num = numeric_gradient(
        lambda: float(softmax_cross_entropy(Tensor(scores), y).data), scores
    )
    assert rel_error(logits.grad, num) < 1e-7


def test_full_graph_gradient():
    """A linear layer, a ReLU and a loss, checked end to end."""
    rng = np.random.RandomState(3)
    X, W, b = rng.randn(6, 4), rng.randn(4, 3), rng.randn(3)
    y = rng.randint(0, 3, size=6)

    tw, tb = Tensor(W), Tensor(b)
    loss = softmax_cross_entropy((Tensor(X) @ tw + tb).relu(), y)
    loss.backward()

    def f():
        return float(
            softmax_cross_entropy((Tensor(X) @ Tensor(W) + Tensor(b)).relu(), y).data
        )

    assert rel_error(tw.grad, numeric_gradient(f, W)) < 1e-6
    assert rel_error(tb.grad, numeric_gradient(f, b)) < 1e-6


def test_agrees_with_scalar_engine():
    """The same computation, one node per number vs one node per matrix.

    Both engines run the same chain rule, so the gradients must match.
    """
    rng = np.random.RandomState(4)
    X, W = rng.randn(3, 2), rng.randn(2, 4)

    tx, tw = Tensor(X), Tensor(W)
    (tx @ tw).relu().backward()

    vx = [[Value(X[i, j]) for j in range(2)] for i in range(3)]
    vw = [[Value(W[j, k]) for k in range(4)] for j in range(2)]
    total = Value(0.0)
    for i in range(3):
        for k in range(4):
            acc = Value(0.0)
            for j in range(2):
                acc = acc + vx[i][j] * vw[j][k]
            total = total + acc.relu()
    total.backward()

    scalar_dx = np.array([[vx[i][j].grad for j in range(2)] for i in range(3)])
    scalar_dw = np.array([[vw[j][k].grad for k in range(4)] for j in range(2)])
    assert np.allclose(tx.grad, scalar_dx, atol=1e-9)
    assert np.allclose(tw.grad, scalar_dw, atol=1e-9)
