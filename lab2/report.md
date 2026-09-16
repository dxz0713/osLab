# Task 3：CIFAR-10 CNN 正式训练与调参实验报告

姓名：__________  
学号：__________  
日期：__________

## 1. 实验环境与任务说明

本实验使用纯 NumPy 实现的 CNN，在 CIFAR-10 数据集上完成图像分类。训练脚本为
`train_cifar10.py`，默认使用两层卷积 block、Adam 优化器和减均值预处理。
训练过程中每 10 个 minibatch 打印一次当前 batch loss，便于确认训练过程正常进行。

建议先运行 1 个 epoch 验证完整流程：

```bash
uv run python train_cifar10.py --epochs 1 --out-dir outputs/smock
```

正式实验应为每组实验指定不同的 `--out-dir`，避免曲线和预测文件互相覆盖。

说明：README 中的“三组对照实验”指三类对比因素，不是要求每个配置重复运行
3 次。本文使用固定的 `--seed 0` 进行对比；除非特别说明，下面每个配置运行
1 次即可。三类对比因素分别为优化器、网络深度/宽度和输入预处理。

### 1.1 实验环境迁移说明

前期流程验证和基线实验在 Ubuntu 虚拟机中完成；第 3 节及之后的所有对照实验
均在 Windows 宿主机上完成。由于本项目使用纯 NumPy 实现，训练过程不使用 GPU
加速，迁移到宿主机后主要使用宿主机的 CPU 和内存资源。与虚拟机相比，宿主机
整体运行更流畅、卡顿更少；迁移前已经得到的 Ubuntu 实验数据保留不变。除非
另有说明，本文后续实验均指 Windows 宿主机上的运行结果。

不同平台的训练时间仅用于记录实际耗时。由于运行环境、CPU、内存、线程数和
数据加载方式不同，不能仅根据训练时间直接比较平台性能。

在 Windows PowerShell 中，命令可写成单行，或使用 PowerShell 的反引号（`）
换行。例如：

```powershell
uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 --num-blocks 2 --base-filters 32 --batch-size 128 --num-val 5000 --seed 0 --out-dir outputs/windows_baseline_adam
```

为避免覆盖 Ubuntu 阶段的输出文件，Windows 阶段应使用新的输出目录，例如
`outputs/windows_opt_sgd`、`outputs/windows_depth_3blocks` 和
`outputs/windows_preprocess_standard`。如果 Windows 阶段重新运行 baseline，
应在结果表中作为 Windows baseline 单独记录，不覆盖下方已有的 Ubuntu 数据。
因此各表中的 `time` 只用于记录实际耗时。

### 1.2 流程验证结果（1 epoch）

本次流程验证实际执行命令为：

```bash
uv run python train_cifar10.py --epochs 1 --out-dir outputs/smock
```

由于未显式指定其他参数，本次实际配置为：Adam，学习率 `1e-3`，`num_blocks=2`，
`base_filters=32`，batch size `128`，验证集大小 `5000`，随机种子 `0`，减均值预处理。

训练结果：

| epoch | train loss | val loss | train acc | val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.1642 | 1.1719 | 59.78% | 58.82% |

训练耗时约 **1225.41 秒（20.42 分钟）**。测试集结果为 loss `1.1997`，
accuracy **58.13%**。该结果用于确认流程正确，不作为最终三组对照实验结论。

逐类测试准确率如下：

| class | accuracy |
|---|---:|
| airplane | 62.80% |
| automobile | 84.80% |
| bird | 46.30% |
| cat | 34.40% |
| deer | 34.60% |
| dog | 52.00% |
| frog | 67.50% |
| horse | 72.00% |
| ship | 71.80% |
| truck | 55.10% |

已生成的流程验证产物：

- `outputs/smock/loss_curve.png`
- `outputs/smock/accuracy_curve.png`
- `outputs/smock/test_predictions.npz`
- `outputs/smock/samples.npz`
- `outputs/smock/sample_predictions.png`

## 2. 基线实验

### 2.1 配置与命令

```powershell
uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 --num-blocks 2 --base-filters 32 --batch-size 128 --num-val 5000 --seed 0 --out-dir outputs/baseline_adam
```

### 2.2 结果

| epoch | train loss | val loss | train acc | val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.1642 | 1.1719 | 59.78% | 58.82% |
| 2 | 0.9731 | 1.0226 | 66.69% | 64.50% |
| 3 | 0.8597 | 0.9468 | 70.84% | 66.78% |
| 4 | 0.7931 | 0.9094 | 73.16% | 68.98% |
| 5 | 0.7394 | 0.8906 | 75.23% | 68.76% |

测试 loss：0.9166  
测试准确率：**68.69%**  
训练时间：**7111.18 秒（118.52 分钟）**
输出图：`outputs/baseline_adam/loss_curve.png`、`outputs/baseline_adam/accuracy_curve.png`

逐类测试准确率如下：

| class | accuracy |
|---|---:|
| airplane | 75.80% |
| automobile | 78.60% |
| bird | 61.00% |
| cat | 50.50% |
| deer | 51.00% |
| dog | 60.80% |
| frog | 84.60% |
| horse | 68.70% |
| ship | 77.10% |
| truck | 78.80% |

其中 `frog` 的识别效果最好（84.60%），`cat` 和 `deer` 相对较弱（分别为
50.50% 和 51.00%）。训练集准确率持续上升，但验证集准确率在第 4 个 epoch
达到最高的 68.98% 后，第 5 个 epoch 小幅下降到 68.76%，说明模型已经出现
轻微过拟合迹象。

## 3. 对照实验一：优化器

固定网络结构、预处理、batch size、epoch 数和随机种子，仅改变优化器。SGD
需要使用较大的学习率；下面的 `0.05` 是起始建议值，可根据曲线调整。
本组比较 `sgd`、`sgd_momentum` 和 `adam` 三种优化器各 1 次。

```powershell
uv run python train_cifar10.py --epochs 5 --optimizer sgd --lr 0.05 --num-blocks 2 --base-filters 32 --batch-size 128 --num-val 5000 --seed 0 --out-dir outputs/opt_sgd

uv run python train_cifar10.py --epochs 5 --optimizer sgd_momentum --lr 0.01 --num-blocks 2 --base-filters 32 --batch-size 128 --num-val 5000 --seed 0 --out-dir outputs/opt_momentum

uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 0.001 --num-blocks 2 --base-filters 32 --batch-size 128 --num-val 5000 --seed 0 --out-dir outputs/opt_adam
```

| optimizer | lr | final train loss | best val acc | test acc | time |
|---|---:|---:|---:|---:|---:|
| sgd | 0.05 | 1.0204 | 61.72% | 60.91% | 10318.92 s（171.98 min） |
| sgd_momentum | 0.01 | 0.8525 | 66.14% | 65.67% | 7062.52 s（117.71 min） |
| adam | 0.001 | 0.7394 | 68.76% | 68.69% | 9372.16 s（156.20 min） |

SGD 的逐 epoch 结果如下：

| epoch | train loss | val loss | train acc | val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.6455 | 1.6507 | 43.10% | 42.60% |
| 2 | 1.2588 | 1.2707 | 55.70% | 55.00% |
| 3 | 1.1840 | 1.2002 | 59.11% | 57.94% |
| 4 | 1.1050 | 1.1378 | 62.03% | 60.12% |
| 5 | 1.0204 | 1.0748 | 64.29% | 61.72% |

测试集 loss 为 `1.0965`，测试准确率为 **60.91%**。逐类准确率为：

| class | accuracy |
|---|---:|
| airplane | 44.90% |
| automobile | 66.70% |
| bird | 51.50% |
| cat | 52.60% |
| deer | 44.10% |
| dog | 42.50% |
| frog | 85.00% |
| horse | 67.60% |
| ship | 88.90% |
| truck | 65.30% |

本次 SGD 训练集和验证集指标均持续改善，暂未出现明显震荡；其中 `ship` 和 `frog`
的分类效果最好，`dog`、`deer` 和 `airplane` 相对较弱。
输出文件为 `outputs/opt_sgd/loss_curve.png`、`outputs/opt_sgd/accuracy_curve.png`、
`outputs/opt_sgd/test_predictions.npz` 和 `outputs/opt_sgd/samples.npz`。

动量 SGD 的逐 epoch 结果如下：

| epoch | train loss | val loss | train acc | val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.2415 | 1.2431 | 56.71% | 55.90% |
| 2 | 1.0726 | 1.1025 | 63.18% | 61.14% |
| 3 | 0.9479 | 1.0076 | 67.82% | 64.84% |
| 4 | 0.8996 | 0.9827 | 69.30% | 65.80% |
| 5 | 0.8525 | 0.9666 | 70.85% | 66.14% |

测试集 loss 为 `0.9888`，测试准确率为 **65.67%**。逐类准确率为：

| class | accuracy |
|---|---:|
| airplane | 77.30% |
| automobile | 79.40% |
| bird | 59.30% |
| cat | 42.90% |
| deer | 37.90% |
| dog | 56.40% |
| frog | 89.30% |
| horse | 65.80% |
| ship | 75.20% |
| truck | 73.20% |

动量 SGD 的训练和验证指标均持续改善，验证准确率从 55.90% 提升到 66.14%，
测试准确率也高于普通 SGD。Adam 的训练损失下降更明显，最终训练损失为 0.7394，
测试准确率达到 68.69%，是三种优化器中最高的；其最佳验证准确率为 68.76%，
同样高于普通 SGD 和动量 SGD。
输出文件为 `outputs/opt_momentum/loss_curve.png`、`outputs/opt_momentum/accuracy_curve.png`、
`outputs/opt_momentum/test_predictions.npz` 和 `outputs/opt_momentum/samples.npz`。

Adam 的逐 epoch 结果如下：

| epoch | train loss | val loss | train acc | val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.2415 | 1.2431 | 56.71% | 55.90% |
| 2 | 1.0726 | 1.1025 | 63.18% | 61.14% |
| 3 | 0.9479 | 1.0076 | 67.82% | 64.84% |
| 4 | 0.8996 | 0.9827 | 69.30% | 65.80% |
| 5 | 0.7394 | 0.8906 | 75.23% | 68.76% |

测试集 loss 为 `0.9166`，测试准确率为 **68.69%**。逐类准确率为：

| class | accuracy |
|---|---:|
| airplane | 75.80% |
| automobile | 78.60% |
| bird | 61.00% |
| cat | 50.50% |
| deer | 51.00% |
| dog | 60.80% |
| frog | 84.60% |
| horse | 68.70% |
| ship | 77.10% |
| truck | 78.80% |

Adam 的训练和验证指标整体持续改善，验证准确率从 55.90% 提升到 68.76%，
测试准确率高于普通 SGD 和动量 SGD。第 5 个 epoch 的训练准确率达到 75.23%，
暂未观察到明显的验证集回落。其中 `frog` 的识别效果最好（84.60%），`cat` 和
`deer` 仍然相对较弱（分别为 50.50% 和 51.00%）。输出文件为
`outputs/opt_adam/loss_curve.png`、`outputs/opt_adam/accuracy_curve.png`、
`outputs/opt_adam/test_predictions.npz` 和 `outputs/opt_adam/samples.npz`。

## 4. 对照实验二：网络深度/宽度

固定优化器为 Adam 和学习率为 `1e-3`，在 baseline 基础上改变 block 数或基础
通道数。按照 README，本组至少完成一种变化即可；下面同时列出深度和宽度两种
建议配置，若资源有限可选择其中一种。每个配置运行 1 次，并建议记录单 epoch
用时。

```powershell
uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 --num-blocks 3 --base-filters 32 --batch-size 128 --num-val 5000 --seed 0 --out-dir outputs/depth_3blocks

uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 --num-blocks 2 --base-filters 48 --batch-size 128 --num-val 5000 --seed 0 --out-dir outputs/width_48
```

| model | num blocks | base filters | best val acc | test acc | time |
|---|---:|---:|---:|---:|---:|
| baseline | 2 | 32 | 68.98% | 68.69% | 7111.18 s（118.52 min） |
| deeper | 3 | 32 | 71.54% | 71.03% | 18776.81 s（312.95 min） |
| wider | 2 | 48 | 70.46% | 69.60% | 14064.98 s（234.42 min） |

### 4.1 Deeper 模型结果

`deeper` 配置使用 3 个 block、32 个基础通道，其他参数与 baseline 保持一致。
逐 epoch 结果如下：

| epoch | train loss | val loss | train acc | val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.1299 | 1.1481 | 59.84% | 59.12% |
| 2 | 0.8989 | 0.9612 | 69.15% | 66.66% |
| 3 | 0.7617 | 0.8712 | 74.12% | 70.06% |
| 4 | 0.6828 | 0.8307 | 76.76% | 71.32% |
| 5 | 0.6223 | 0.8220 | 78.82% | 71.54% |

测试集 loss 为 `0.8446`，测试准确率为 **71.03%**。逐类准确率如下：

| class | accuracy |
|---|---:|
| airplane | 80.10% |
| automobile | 80.30% |
| bird | 63.50% |
| cat | 39.00% |
| deer | 56.00% |
| dog | 61.60% |
| frog | 88.70% |
| horse | 71.20% |
| ship | 87.60% |
| truck | 82.30% |

与 2 个 block 的 baseline 相比，deeper 模型的测试准确率提高 `2.34` 个百分点，
最佳验证准确率提高 `2.56` 个百分点；但训练时间从 118.52 分钟增加到
312.95 分钟，约为 baseline 的 2.64 倍。增加网络深度提升了模型表现，但也带来
了明显的计算时间开销，是否值得取决于对准确率和训练时间的侧重。

### 4.2 Wider 模型结果

`wider` 配置使用 2 个 block、48 个基础通道，其他参数与 baseline 保持一致。
逐 epoch 结果如下：

| epoch | train loss | val loss | train acc | val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.0987 | 1.1227 | 61.84% | 60.58% |
| 2 | 0.8927 | 0.9687 | 69.77% | 66.36% |
| 3 | 0.7857 | 0.9133 | 73.34% | 68.68% |
| 4 | 0.7024 | 0.8701 | 76.04% | 69.90% |
| 5 | 0.6558 | 0.8630 | 77.98% | 70.46% |

测试集 loss 为 `0.8842`，测试准确率为 **69.60%**。逐类准确率如下：

| class | accuracy |
|---|---:|
| airplane | 80.50% |
| automobile | 80.40% |
| bird | 60.20% |
| cat | 45.00% |
| deer | 50.80% |
| dog | 68.70% |
| frog | 86.90% |
| horse | 66.90% |
| ship | 76.30% |
| truck | 80.30% |

与 baseline 相比，wider 模型的测试准确率提高 `0.91` 个百分点，最佳验证准确率
提高 `1.48` 个百分点；训练时间从 118.52 分钟增加到 234.42 分钟，约为
baseline 的 1.98 倍。增加基础通道数带来了小幅性能提升，但额外计算时间较多；
在本实验中，deeper 模型的准确率提升更明显，但训练耗时也最高。

## 5. 对照实验三：输入预处理

保持模型和优化器不变，修改 `train_cifar10.py` 中第 169--190 行的预处理代码。
本组比较以下三种预处理方式各 1 次，每次只启用一种方式。报告中的顺序与
`train_cifar10.py` 中的代码顺序一致：

1. **减均值（默认，方式 A）**：启用 `mean_img = X_tr.mean(axis=0)`，并将同一
   组训练集均值分别从训练、验证和测试数据中减去。
2. **按通道标准化（方式 B）**：启用按通道计算的均值和标准差代码，并使用同一
   组训练集统计量处理训练、验证和测试数据。
3. **仅缩放到 `[0, 1]`（方式 C）**：注释掉前两种预处理代码。由于
   `load_cifar10` 内部已经执行 `/255.0`，此时数据保持在 `[0, 1]` 范围内。

方式 A 的减均值结果直接复用第 2 节 baseline 实验，输出目录为
`outputs/baseline_adam`；其余两种方式使用独立的输出目录：

```powershell
uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 --num-blocks 2 --base-filters 32 --num-val 5000 --seed 0 --out-dir outputs/preprocess_standard

uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 --num-blocks 2 --base-filters 32 --num-val 5000 --seed 0 --out-dir outputs/preprocess_scale
```

| preprocessing | initial loss | epoch-5 train loss | best val acc | test acc |
|---|---:|---:|---:|---:|
| mean subtraction | 1.1642 | 0.7394 | 68.98% | 68.69% |
| channel standardization | 1.1297 | 0.6795 | 68.48% | 68.04% |
| `[0,1]` scaling | 1.1697 | 0.8252 | 66.14% | 67.15% |

### 5.1 按通道标准化结果

按通道标准化使用训练集的通道均值和标准差处理训练、验证和测试数据。逐 epoch
结果如下：

| epoch | train loss | val loss | train acc | val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.1297 | 1.1608 | 61.16% | 58.76% |
| 2 | 0.9234 | 1.0097 | 68.48% | 65.20% |
| 3 | 0.8002 | 0.9389 | 73.04% | 67.42% |
| 4 | 0.7296 | 0.9175 | 75.17% | 68.14% |
| 5 | 0.6795 | 0.9164 | 77.20% | 68.48% |

测试集 loss 为 `0.9455`，测试准确率为 **68.04%**。逐类准确率如下：

| class | accuracy |
|---|---:|
| airplane | 76.80% |
| automobile | 79.40% |
| bird | 62.30% |
| cat | 47.10% |
| deer | 55.60% |
| dog | 56.70% |
| frog | 83.00% |
| horse | 63.70% |
| ship | 80.00% |
| truck | 75.80% |

训练时间为 **8563.63 秒（142.73 分钟）**，输出文件为
`outputs/preprocess_standard/loss_curve.png`、
`outputs/preprocess_standard/accuracy_curve.png`、
`outputs/preprocess_standard/test_predictions.npz` 和
`outputs/preprocess_standard/samples.npz`。

分析：按通道标准化的初始训练 loss 为 `1.1297`，第 5 个 epoch 的训练 loss
下降到 `0.6795`，测试准确率为 `68.04%`。在本次实验中，其测试准确率比减均值
方式的 `68.69%` 低 `0.65` 个百分点，最佳验证准确率也低 `0.50` 个百分点；
仅凭本次单次运行结果，按通道标准化没有表现出优于减均值的泛化效果。

### 5.2 仅缩放到 `[0, 1]` 的结果

由于 `load_cifar10` 已经将像素值除以 `255.0`，方式 C 不再进行减均值或标准化，
直接使用 `[0, 1]` 范围内的输入。逐 epoch 结果如下：

| epoch | train loss | val loss | train acc | val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.1697 | 1.1840 | 59.96% | 58.64% |
| 2 | 1.0159 | 1.0729 | 65.35% | 62.28% |
| 3 | 0.9152 | 1.0051 | 69.06% | 64.60% |
| 4 | 0.8966 | 1.0039 | 69.58% | 65.48% |
| 5 | 0.8252 | 0.9693 | 71.96% | 66.14% |

测试集 loss 为 `0.9651`，测试准确率为 **67.15%**。逐类准确率如下：

| class | accuracy |
|---|---:|
| airplane | 72.70% |
| automobile | 81.00% |
| bird | 63.70% |
| cat | 39.50% |
| deer | 47.10% |
| dog | 64.10% |
| frog | 83.90% |
| horse | 66.40% |
| ship | 75.30% |
| truck | 77.80% |

训练时间为 **8895.92 秒（148.27 分钟）**，输出文件为
`outputs/preprocess_scale/loss_curve.png`、`outputs/preprocess_scale/accuracy_curve.png`、
`outputs/preprocess_scale/test_predictions.npz` 和 `outputs/preprocess_scale/samples.npz`。

与减均值方式相比，仅缩放到 `[0, 1]` 的测试准确率低 `1.54` 个百分点，
最佳验证准确率低 `2.84` 个百分点；与按通道标准化相比，测试准确率低
`0.89` 个百分点。就本次实验结果而言，减均值的整体泛化表现最好，按通道
标准化次之，仅缩放到 `[0, 1]` 的验证集和测试集表现相对较弱。

## 6. 预测结果与可视化

训练结束后，使用最终模型输出的预测文件运行：

```bash
uv run python show_predict_picture.py \
  --predictions outputs/baseline_adam/test_predictions.npz
```

程序会随机显示 12 张测试图片。结合多次运行结果，填写哪些类别通常较容易
识别、哪些类别容易混淆，并从类别外观相似度、目标姿态和背景变化等方面分析原因。

## 7. 思考题

### 7.1 哪些类别预测较好或较差？原因是什么？

正式基线结果中，`frog`、`truck`、`automobile` 和 `ship` 的准确率较高，
其中 `frog` 为 84.60%；`cat` 和 `deer` 较低，分别为 50.50% 和 51.00%。
这可能是因为青蛙、车辆和船的整体轮廓相对明显，而猫、鹿与其他动物类别在
低分辨率、姿态变化和背景干扰下更容易混淆。流程验证结果中也呈现了相同趋势：
`cat` 和 `deer` 是较难识别的类别。

### 7.2 训练准确率上升但验证准确率停滞或下降说明什么？

这通常说明模型开始过拟合训练集，泛化能力没有继续提升。我的实验中**出现了轻微
过拟合**：训练准确率从第 4 个 epoch 的 73.16% 继续上升到第 5 个 epoch 的
75.23%，但验证准确率从 68.98% 小幅下降到 68.76%。后续可以使用最佳验证准确率
对应的 checkpoint（第 4 个 epoch），也可以减少训练 epoch、加入数据增强或正则化，
并进一步调整学习率。

### 7.3 哪个因素影响最大？

根据三组实验的测试准确率填写：

| 因素 | 最好测试 acc | 最差测试 acc | 差值 |
|---|---:|---:|---:|
| 优化器 | 68.69% | 60.91% | 7.78 个百分点 |
| 深度/宽度 | 71.03% | 68.69% | 2.34 个百分点 |
| 预处理 | 68.69% | 67.15% | 1.54 个百分点 |

结论：在本实验设置和随机种子下，影响最大的是**优化器**，因为其测试准确率
变化为 **7.78 个百分点**。Adam 的测试准确率最高（68.69%），普通 SGD
最低（60.91%）；从训练曲线看，Adam 的 loss 下降更快、最终训练 loss 更低，
说明优化器选择对收敛速度和最终性能的影响最明显。

## 8. 最终模型与复现命令

最终提交的 `test_predictions.npz` 位于最终实验的输出目录。请将下面命令替换为
实际采用的全部参数，并确保命令中的 `--out-dir` 对应提交的预测文件。若最终
模型在 Windows 宿主机上训练，应填写 Windows 阶段实际执行的命令和输出目录；
下方命令仅保留为原 Ubuntu baseline 的复现命令示例。

```powershell
uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 --num-blocks 2 --base-filters 32 --batch-size 128 --num-val 5000 --seed 0 --data-root data --out-dir outputs/baseline_adam
```

最终测试准确率：**68.69%**（基线实验，Ubuntu 虚拟机）
流程验证测试准确率：58.13%
流程验证预测文件：`outputs/smock/test_predictions.npz`  
最终预测文件：`outputs/baseline_adam/test_predictions.npz`  
文件包含：`y_true`、`y_pred`、`classes`。

## 9. 提交前检查清单

- [x] 三类对照实验均有命令、表格、曲线和文字分析；每个配置至少运行 1 次。
- [x] 深度/宽度对比至少完成一种变化（deeper 和 wider 均已完成）。
- [x] 报告回答三个思考题，并使用实际数据支持结论。
- [x] 最终命令包含所有参数，且能生成提交的 `test_predictions.npz`。
- [x] 已说明实验平台划分；Windows 阶段使用独立的 `--out-dir`，未覆盖原有数据。
- [x] 已保留 `loss_curve.png`、`accuracy_curve.png` 和逐类准确率输出。
- [ ] 已运行 `uv run python show_predict_picture.py` 并检查可视化结果。
