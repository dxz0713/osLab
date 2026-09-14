# Task 3：CIFAR-10 CNN 正式训练与调参实验报告（初稿）

姓名：__________  
学号：__________  
日期：__________

## 1. 实验环境与任务说明

本实验使用纯 NumPy 实现的 CNN，在 CIFAR-10 数据集上完成图像分类。训练脚本为
`train_cifar10.py`，默认使用两层卷积 block、Adam 优化器和减均值预处理。
训练过程中每 10 个 minibatch 会打印一次当前 batch loss，便于确认训练仍在正常进行。

建议先运行 1 个 epoch 验证完整流程：

```bash
uv run python train_cifar10.py --epochs 1 --out-dir outputs/smock
```

正式实验应为每组实验指定不同的 `--out-dir`，避免曲线和预测文件互相覆盖。

### 1.1 流程验证结果（1 epoch）

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

```bash
uv run python train_cifar10.py \
  --epochs 5 --optimizer adam --lr 1e-3 \
  --num-blocks 2 --base-filters 32 \
  --batch-size 128 --num-val 5000 --seed 0 \
  --out-dir outputs/baseline_adam
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

```bash
uv run python train_cifar10.py --epochs 5 --optimizer sgd \
  --lr 0.05 --num-blocks 2 --base-filters 32 --batch-size 128 \
  --num-val 5000 --seed 0 --out-dir outputs/opt_sgd

uv run python train_cifar10.py --epochs 5 --optimizer sgd_momentum \
  --lr 0.01 --num-blocks 2 --base-filters 32 --batch-size 128 \
  --num-val 5000 --seed 0 --out-dir outputs/opt_momentum

uv run python train_cifar10.py --epochs 5 --optimizer adam \
  --lr 0.001 --num-blocks 2 --base-filters 32 --batch-size 128 \
  --num-val 5000 --seed 0 --out-dir outputs/opt_adam
```

| optimizer | lr | final train loss | best val acc | test acc | time |
|---|---:|---:|---:|---:|---:|
| sgd | 0.05 | 待填写 | 待填写 | 待填写 | 待填写 |
| sgd_momentum | 0.01 | 待填写 | 待填写 | 待填写 | 待填写 |
| adam | 0.001 | 待填写 | 待填写 | 待填写 | 待填写 |

分析：比较三张 loss 曲线的下降速度、曲线是否震荡，以及验证准确率的最终值。
填写实际结果后说明哪种优化器收敛最快、哪种泛化最好。

## 4. 对照实验二：网络深度/宽度

固定优化器为 Adam 和学习率为 `1e-3`，分别改变 block 数或基础通道数。
至少完成一种变化，建议同时记录单 epoch 用时。

```bash
uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 \
  --num-blocks 3 --base-filters 32 --batch-size 128 \
  --num-val 5000 --seed 0 --out-dir outputs/depth_3blocks

uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 \
  --num-blocks 2 --base-filters 48 --batch-size 128 \
  --num-val 5000 --seed 0 --out-dir outputs/width_48
```

| model | num blocks | base filters | best val acc | test acc | time |
|---|---:|---:|---:|---:|---:|
| baseline | 2 | 32 | 待填写 | 待填写 | 待填写 |
| deeper | 3 | 32 | 待填写 | 待填写 | 待填写 |
| wider | 2 | 48 | 待填写 | 待填写 | 待填写 |

分析：讨论增加容量是否提高准确率，以及额外计算时间是否值得。若训练或显存
明显变慢，也应在这里记录。

## 5. 对照实验三：输入预处理

保持模型和优化器不变，修改 `train_cifar10.py` 中第 169--190 行的预处理代码。
每次只启用一种方式：

1. **仅缩放到 `[0, 1]`**：注释掉减均值代码，保留 `mean_img = 0`。
2. **减均值（默认）**：启用 `mean_img = X_tr.mean(axis=0)` 及三次减法。
3. **按通道标准化**：启用通道均值和标准差代码，并用同一组训练集统计量处理
   训练、验证和测试数据。

每次运行前请使用不同输出目录，例如：

```bash
uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 \
  --num-blocks 2 --base-filters 32 --num-val 5000 --seed 0 \
  --out-dir outputs/preprocess_scale

uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 \
  --num-blocks 2 --base-filters 32 --num-val 5000 --seed 0 \
  --out-dir outputs/preprocess_center

uv run python train_cifar10.py --epochs 5 --optimizer adam --lr 1e-3 \
  --num-blocks 2 --base-filters 32 --num-val 5000 --seed 0 \
  --out-dir outputs/preprocess_standard
```

| preprocessing | initial loss | epoch-5 train loss | best val acc | test acc |
|---|---:|---:|---:|---:|
| `[0,1]` scaling | 待填写 | 待填写 | 待填写 | 待填写 |
| mean subtraction | 待填写 | 待填写 | 待填写 | 待填写 |
| channel standardization | 待填写 | 待填写 | 待填写 | 待填写 |

分析：比较初始 loss、前几个 epoch 的下降速度和最终验证/测试准确率，并解释
输入中心化或标准化对优化稳定性的影响。

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
| 优化器 | 待填写 | 待填写 | 待填写 |
| 深度/宽度 | 待填写 | 待填写 | 待填写 |
| 预处理 | 待填写 | 待填写 | 待填写 |

结论：在本实验设置和随机种子下，影响最大的是 __________，因为其测试准确率
变化为 __________ 个百分点，且曲线显示 __________。

## 8. 最终模型与复现命令

最终提交的 `test_predictions.npz` 位于最终实验的输出目录。请将下面命令替换为
实际采用的全部参数，并确保命令中的 `--out-dir` 对应提交的预测文件：

```bash
uv run python train_cifar10.py \
  --epochs 5 --optimizer adam --lr 1e-3 \
  --num-blocks 2 --base-filters 32 \
  --batch-size 128 --num-val 5000 --seed 0 \
  --data-root data --out-dir outputs/baseline_adam
```

最终测试准确率：**68.69%**  
流程验证测试准确率：58.13%  
流程验证预测文件：`outputs/smock/test_predictions.npz`  
最终预测文件：`outputs/baseline_adam/test_predictions.npz`  
文件包含：`y_true`、`y_pred`、`classes`。

## 9. 提交前检查清单

- [ ] 三组对照实验均有命令、表格、曲线和文字分析。
- [ ] 报告回答三个思考题，并使用实际数据支持结论。
- [ ] 最终命令包含所有参数，且能生成提交的 `test_predictions.npz`。
- [ ] 已保留 `loss_curve.png`、`accuracy_curve.png` 和逐类准确率输出。
- [ ] 已运行 `uv run python show_predict_picture.py` 并检查可视化结果。
