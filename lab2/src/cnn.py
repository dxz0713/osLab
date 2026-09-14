from __future__ import annotations

import numpy as np

from src.conv import conv_backward, conv_forward
from src.layers import affine_backward, affine_forward, relu_backward, relu_forward, softmax_loss
from src.pool import max_pool_backward, max_pool_forward
from src.init import he_conv


class ConvNet:
    """A small convolutional network for CIFAR-10.

    Architecture (the classic conv-pool block pattern):

        [conv - relu - pool] x num_blocks  ->  flatten  ->  affine -> scores

    Each conv keeps the spatial size unchanged (3x3 kernel, stride 1, pad 1);
    each pool halves it. Two blocks turn 32x32 into 8x8, which is then
    flattened and mapped to the 10 class scores by a single affine layer.

    The constructor only allocates parameters and records the architecture;
    shapes are verified lazily on the first forward pass. Params are stored
    as a plain dict of numpy arrays so that any optimizer from optim.py can
    be applied directly:

        for name, p in net.params.items():
            net.params[name], _ = adam(p, grads[name], configs[name])
    """

    def __init__(self, num_classes=10, num_blocks=2, base_filters=32, input_side=32, seed=0):
        rng = np.random.default_rng(seed)
        self.num_blocks = num_blocks
        self.input_side = input_side
        self.params = {}
        self.grads = {}

        in_ch = 3
        out_ch = base_filters
        for b in range(num_blocks):
            self.params[f"W{b}"] = he_conv(in_ch, out_ch, 3, rng)
            self.params[f"b{b}"] = np.zeros(out_ch)
            in_ch = out_ch
            out_ch *= 2

        # After num_blocks pools of stride 2 the map is input_side / 2**num_blocks.
        side = input_side // (2**num_blocks)
        self.params["Wfc"] = he_conv(in_ch * side * side, num_classes, 1, rng).reshape(
            in_ch * side * side, num_classes
        )
        self.params["bfc"] = np.zeros(num_classes)

    # ------------------------------------------------------------------ layers

    def _conv_relu_pool(self, x, idx):
        """One conv-relu-pool block. Returns (out, caches)."""
        W, b = self.params[f"W{idx}"], self.params[f"b{idx}"]

        conv_out, conv_cache = conv_forward(x, W, b, {"stride": 1, "pad": 1})
        relu_out, relu_cache = relu_forward(conv_out)
        pool_out, pool_cache = max_pool_forward(
            relu_out, {"pool_height": 2, "pool_width": 2, "stride": 2}
        )

        return pool_out, (conv_cache, relu_cache, pool_cache)
  

    def _conv_relu_pool_backward(self, dout, caches):
        conv_cache, relu_cache, pool_cache = caches

        drelu = max_pool_backward(dout, pool_cache)
        dconv = relu_backward(drelu, relu_cache)
        dx, dW, db = conv_backward(dconv, conv_cache)

        return dx, dW, db


    def _block_param_names(self, idx):
        return f"W{idx}", f"b{idx}"

    # ------------------------------------------------------------ fwd / bwd

    def loss(self, X, y=None):
        """Forward pass, and backward pass if labels are given.

        Inputs:
        - X: (N, 3, 32, 32) input images, already normalized
        - y: (N,) int labels; if None only scores are returned

        Returns:
        - scores (N, num_classes) if y is None
        - (loss, grads) otherwise, grads keyed like params
        """
        out = X
        caches = []
        for b in range(self.num_blocks):
            out, block_cache = self._conv_relu_pool(out, b)
            caches.append(block_cache)

        scores, fc_cache = affine_forward(
            out, self.params["Wfc"], self.params["bfc"]
        )
        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        _, grads = self._backward(dscores, caches, fc_cache)
        return loss, grads

    def _backward(self, dscores, caches, fc_cache):
        """Backward pass through the entire network.

        Args:
            dscores: Upstream gradient from the softmax loss layer,
                     shape (N, num_classes)
            caches: List of block caches from each conv block, length num_blocks
            fc_cache: Cache from the fully connected layer forward pass

        Returns:
            dout: Gradient with respect to the input X, shape (N, 3, H, W)
            grads: Dictionary mapping parameter names to their gradients.
                   Keys include:
                   - "Wfc": gradient for fully connected weight
                   - "bfc": gradient for fully connected bias
                   - "W{b}": gradient for conv block b weight
                   - "b{b}": gradient for conv block b bias
                   for b in range(num_blocks)
        """
        dout, dWfc, dbfc = affine_backward(dscores, fc_cache)
        grads = {"Wfc": dWfc, "bfc": dbfc}

        for b in range(self.num_blocks - 1, -1, -1):
            dout, dW, db = self._conv_relu_pool_backward(dout, caches[b])
            w_name, b_name = self._block_param_names(b)
            grads[w_name] = dW
            grads[b_name] = db

        return dout, grads

    def predict(self, X, batch_size=200):
        """Predict class labels for input images.

        Args:
            X: Input images of shape (N, 3, H, W), already normalized
            batch_size: Number of samples to process at once to bound memory

        Returns:
            preds: 1D array of shape (N,) containing predicted class indices
                   for each input image. Each value is an integer in [0, num_classes-1].
        """
        N = X.shape[0]
        preds = np.zeros(N, dtype=np.int64)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X[start:end]

            # Forward pass through the network
            out = X_batch
            for b in range(self.num_blocks):
                out, _ = self._conv_relu_pool(out, b)

            # Flatten and pass through the fully connected layer
            out_flat = out.reshape(out.shape[0], -1)
            scores, _ = affine_forward(out_flat, self.params["Wfc"], self.params["bfc"])

            # Get predicted class indices
            preds[start:end] = np.argmax(scores, axis=1)

        return preds
