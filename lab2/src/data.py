from __future__ import annotations

import os
import urllib.request
import zipfile

import numpy as np

CIFAR10_URL = (
    "https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com/"
    "ComputerVision/cifar10_mindspore.zip"
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _read_cifar10_batch(bin_path):
    """Read one CIFAR-10 binary batch file into (images, labels).

    CIFAR-10 binary record layout: 10000 records per batch, each record is
    3073 bytes = 1 label byte followed by 3072 pixel bytes (R/G/B three
    channel planes, each 32x32=1024 bytes, stored in row-major order).

    Returns:
    - X: (10000, 3, 32, 32) float64 in [0, 1]
    - y: (10000,) int64
    """
    data = np.fromfile(bin_path, dtype=np.uint8)
    data = data.reshape(-1, 3073)
    y = data[:, 0].astype(np.int64)
    X = data[:, 1:].reshape(-1, 3, 32, 32).astype(np.float64) / 255.0
    return X, y


def _locate_extracted_dir(root):
    """Return the extracted MindSpore dataset directory, or None if absent.

    Accepts both the documented layout and a few common alternatives, so a
    student who unzips the archive by hand into a slightly different folder
    still gets picked up automatically.
    """
    candidates = [
        os.path.join(root, "cifar10_mindspore"),
        os.path.join(root, "cifar10_mindspore", "cifar10_mindspore"),
    ]
    for d in candidates:
        train_dir = os.path.join(d, "data", "10-batches-bin")
        test_path = os.path.join(d, "data", "10-verify-bin", "test_batch.bin")
        if os.path.isdir(train_dir) and os.path.exists(test_path):
            return train_dir, test_path
    return None, None


def load_cifar10(root="data"):
    """Load CIFAR-10 into numpy arrays.

    On first use it looks for the extracted MindSpore dataset under
    ``<root>/cifar10_mindspore``; if missing, it downloads the archive from
    the Huawei OBS mirror and extracts it. The parsed arrays are then cached
    as a single ``cifar10.npz`` so subsequent runs start instantly.

    Returns:
    - X_train: (50000, 3, 32, 32) float64 in [0, 1]
    - y_train: (50000,) int64
    - X_test:  (10000, 3, 32, 32) float64 in [0, 1]
    - y_test:  (10000,) int64
    """
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, "cifar10.npz")

    if os.path.exists(cache):
        with np.load(cache) as z:
            return z["X_train"], z["y_train"], z["X_test"], z["y_test"]

    train_dir, test_path = _locate_extracted_dir(root)

    if train_dir is None:
        zip_path = os.path.join(root, "cifar10_mindspore.zip")
        if not os.path.exists(zip_path):
            print(f"downloading CIFAR-10 from {CIFAR10_URL} ...")
            urllib.request.urlretrieve(CIFAR10_URL, zip_path)
        extract_dir = os.path.join(root, "cifar10_mindspore")
        print(f"extracting to {extract_dir} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        train_dir, test_path = _locate_extracted_dir(root)

    if train_dir is None:
        raise FileNotFoundError(
            "Could not locate the CIFAR-10 batch files. Expected them under "
            f"{os.path.join(root, 'cifar10_mindspore', 'data', '10-batches-bin')}."
        )

    xs, ys = [], []
    for i in range(1, 6):
        X, y = _read_cifar10_batch(os.path.join(train_dir, f"data_batch_{i}.bin"))
        xs.append(X)
        ys.append(y)

    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = _read_cifar10_batch(test_path)

    np.savez_compressed(cache, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test


def get_mean_image(X_train):
    """Compute the mean training image for preprocessing (Task 2)."""
    return X_train.mean(axis=0)


def train_val_split(X, y, num_val=5000, seed=0):
    """Shuffle and hold out a validation set.

    Returns (X_train, y_train, X_val, y_val).
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(X.shape[0])
    val_idx, train_idx = idx[:num_val], idx[num_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def sample_batch(rng, X, y, batch_size):
    """One random minibatch of indices and the corresponding data."""
    idx = rng.choice(X.shape[0], size=batch_size, replace=False)
    return X[idx], y[idx]
