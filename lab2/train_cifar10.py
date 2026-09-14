"""Train a from-scratch CNN on CIFAR-10 (Task 3 run script).

Usage:
    python train_cifar10.py                                    # default settings
    python train_cifar10.py --epochs 3                # tweak hyperparameters



The script ends by:
  1. printing per-class accuracy on the test set;
  2. saving the training curves (loss / accuracy) to PNG files;
  3. (optional) saving the trained model and a batch of test samples with
     their predicted vs. true labels.
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.data import CIFAR10_CLASSES, load_cifar10, train_val_split
from src.train import TrainConfig, evaluate, predict, train


def parse_args():
    p = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 from scratch")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--optimizer", choices=["sgd", "sgd_momentum", "adam"], default="adam")
    p.add_argument("--num-val", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--save-model", action="store_true", help="Save trained model parameters to an .npz file")
    p.add_argument("--predict", type=str, default=None, help="Given an .npz file with X (and optional y), print predicted class (and true class) for each sample")
    p.add_argument("--model", type=str, default=None, help="Path to saved model .npz file to use with --predict")
    return p.parse_args()


# --------------------------------------------------------------------------- utils

def _save_curves(history, out_dir):
    matplotlib.use("Agg")
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # ---- Loss Curve ----
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], marker="o", label="train loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], marker="s", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"loss curve saved to {loss_path}")

    # ---- Accuracy Curve ----
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_acc"], marker="o", label="train acc")
    if "val_acc" in history:
        plt.plot(epochs, history["val_acc"], marker="s", label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"accuracy curve saved to {acc_path}")


def _report_test(net, X_test, y_test, out_dir):
    preds = predict(net, X_test)
    acc = float((preds == y_test).mean())
    test_loss, _ = evaluate(net, X_test, y_test)
    print("=" * 60)
    print(f"test loss = {test_loss:.4f}, test accuracy = {acc:.4f} ({acc * 100:.2f}%)")
    print("-" * 60)
    print("per-class accuracy:")
    for c in range(10):
        mask = y_test == c
        c_acc = float((preds[mask] == c).mean())
        print(f"  {CIFAR10_CLASSES[c]:>10s}: {c_acc * 100:5.2f}%  ({mask.sum():5d} samples)")
    print("=" * 60)

    # 保存预测结果，供助教核对。
    np.savez(
        os.path.join(out_dir, "test_predictions.npz"),
        y_true=y_test,
        y_pred=preds,
        classes=np.array(CIFAR10_CLASSES),
    )
    print(f"predictions saved to {os.path.join(out_dir, 'test_predictions.npz')}")


def _save_model(net, out_dir):
    """Export network parameters as .npz for reuse with --predict."""
    path = os.path.join(out_dir, "net.npz")
    np.savez(path, **net.params)
    print(f"model saved to {path}")
    return path


def _load_model(net, model_path):
    with np.load(model_path) as z:
        for name in net.params:
            if name not in z:
                raise KeyError(f"model file {model_path} missing param {name!r}")
            net.params[name] = z[name]
    print(f"model loaded from {model_path}")


def _predict_samples(net, samples_path):
    """Given an .npz file containing X (and optionally y), print the predicted class (and true class) for each sample."""
    with np.load(samples_path) as z:
        X = z["X"]
        y = z["y"] if "y" in z else None

    preds = predict(net, X)
    print("=" * 60)
    print(f"{'index':>5}  {'predicted':>10}  {'true':>10}")
    for i in range(X.shape[0]):
        pred_name = CIFAR10_CLASSES[preds[i]]
        true_name = CIFAR10_CLASSES[y[i]] if y is not None else "-"
        print(f"{i:>5}  {pred_name:>10}  {true_name:>10}")
    print("=" * 60)


# --------------------------------------------------------------------------- main

def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Prediction-only mode: load and reuse a pre-trained model ----
    if args.predict is not None:
        from src.cnn import ConvNet

        net = ConvNet(
            num_classes=10,
            num_blocks=args.num_blocks,
            base_filters=args.base_filters,
            seed=args.seed,
        )
        _load_model(net, args.model)
        _predict_samples(net, args.predict)
        return

    # ---- normal train pattern ----
    X_train_full, y_train_full, X_test, y_test = load_cifar10(args.data_root)
    X_tr, y_tr, X_val, y_val = train_val_split(
        X_train_full, y_train_full, num_val=args.num_val, seed=args.seed
    )

    # ====================================================================
    # 预处理方式 A：减均值（去中心化）
    # ====================================================================
    mean_img = X_tr.mean(axis=0)
    X_tr = X_tr - mean_img
    X_val = X_val - mean_img
    X_test = X_test - mean_img

    # ====================================================================
    # 预处理方式 B：按通道标准化（减均值 + 除以标准差）
    # ====================================================================
    # mean_img = X_tr.mean(axis=(0, 2, 3), keepdims=True)   
    # std_img = X_tr.std(axis=(0, 2, 3), keepdims=True)     
    # X_tr = (X_tr - mean_img) / std_img
    # X_val = (X_val - mean_img) / std_img
    # X_test = (X_test - mean_img) / std_img

    # ====================================================================
    # 预处理方式 C：仅缩放到 [0,1]
    # 注意：load_cifar10 内部已经做了 /255.0，数据本身就在 [0,1] 范围内。
    # ====================================================================
    # mean_img = 0  # 占位变量

    print(f"train: {X_tr.shape}, val: {X_val.shape}, test: {X_test.shape}")
    print(f"after centering: mean={X_tr.mean():+.4f}, std={X_tr.std():.4f}")
    print("class names:", CIFAR10_CLASSES)

    cfg = TrainConfig(
        num_blocks=args.num_blocks,
        base_filters=args.base_filters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        num_val=args.num_val,
        seed=args.seed,
    )
    print(f"config: {cfg}")
    # ---- train start ----
    start_time = time.time()

    net, history = train(X_tr, y_tr, cfg, X_val=X_val, y_val=y_val)

    # ---- train end ----
    total_time = time.time() - start_time
    print(f"Training {args.epochs} epochs completed, total time: {total_time:.2f}s ({total_time/60:.2f}min)")

    # ---- output result ----
    _save_curves(history, args.out_dir)
    _report_test(net, X_test, y_test, args.out_dir)

    if args.save_model:
        _save_model(net, args.out_dir)

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(X_test.shape[0], size=10, replace=False)
    np.savez(
        os.path.join(args.out_dir, "samples.npz"),
        X=(X_test[idx] + mean_img),  
        y=y_test[idx],
    )
    print(f"sample images saved to {os.path.join(args.out_dir, 'samples.npz')}")

    return net, history


if __name__ == "__main__":
    main()
  