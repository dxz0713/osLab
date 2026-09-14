import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Show CIFAR-10 test predictions")
parser.add_argument(
    "--predictions",
    default="outputs/test_predictions.npz",
    help="Path to test_predictions.npz",
)
args = parser.parse_args()

pred_data = np.load(args.predictions)
y_pred_all = pred_data['y_pred']   # (10000,)
classes = pred_data['classes']     


from src.data import load_cifar10
_, _, X_test, y_test = load_cifar10('data')

n_samples = 12
indices = np.random.choice(X_test.shape[0], size=n_samples, replace=False)


X_batch = X_test[indices]
y_batch = y_test[indices]
y_pred_batch = y_pred_all[indices]

correct_count = (y_pred_batch == y_batch).sum()
accuracy = correct_count / n_samples

n_cols = 4
n_rows = (n_samples + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

if n_rows == 1:
    axes = axes.reshape(1, -1)

for i in range(n_samples):
    row, col = i // n_cols, i % n_cols
    ax = axes[row, col]
    
    
    img = X_batch[i].transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    
    ax.imshow(img)
    
    
    true_label = classes[y_batch[i]]
    pred_label = classes[y_pred_batch[i]]
    
    
    is_correct = (y_pred_batch[i] == y_batch[i])
    color = 'green' if is_correct else 'red'
    
    
    ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', 
                 color=color, fontsize=10)
    ax.axis('off')

for i in range(n_samples, n_rows * n_cols):
    row, col = i // n_cols, i % n_cols
    axes[row, col].axis('off')

fig.suptitle(f'Random Samples ({n_samples} images)\nAccuracy: {correct_count}/{n_samples} = {accuracy:.2%}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = os.path.join(os.path.dirname(args.predictions), "sample_predictions.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"accuracy: {correct_count}/{n_samples} = {accuracy:.2%}")
print(f"figure saved to {output_path}")