# 无标签信息最大化测试时适配：六项 EEG 数据实验

## 结论

我测试了一个输出层测试时适配方法：对每位 PhysioNet 测试被试的无标签预测单独校准类别 logit 偏置。六项准确率的等权平均由 **72.78%** 升到 **72.98%（+0.20 个百分点）**。验证集只支持在 PhysioNet task 1 和 task 3 启用适配；另外四项沿用适配前结果。

这次实验显示了两个小幅局部提升，但没有证据说明整体提升显著，也没有达到“所有数据集都提升”。PhysioNet task 1 的被试配对 bootstrap 区间跨 0；task 3 的区间下界为 0.00 pp。当前结果不应表述成显著改进。

## 方法

方法受测试时熵适配工作启发：在每个无标签目标域批次上，仅调整类别 logit 偏置 `b`，优化

`H(Y_hat | X) - beta * H(Y_hat) + ridge * ||b - mean(b)||^2`。

第一项鼓励单样本预测更确定，第二项避免整批都塌缩到同一类别，最后一项限制偏置变化。最终概率可与原概率按 `mix` 混合。验证集从五组预设强度及“不适配”中选择；只有配对验证准确率增益超过一个验证域标准误的预设才有资格被选中。PhysioNet 按被试（每被试 45 个试次）形成目标域批次。BCI 的已保存预测文件没有被试 ID，因此按整个已提供分区成批；验证选择没有启用适配。

该做法是受 T-TIME 启发的简化批次版，只适配输出偏置，不是原论文算法的复现。[T-TIME 论文](https://arxiv.org/abs/2412.07228)

## 测试结果

| 数据集 | 验证选择 | 原准确率 | 适配后 | 变化 | 相对15模型历史最佳 |
|---|---|---:|---:|---:|---:|
| PhysioNet task 1 | mi_full | 70.91% | 71.82% | +0.91 pp | -0.10 pp |
| PhysioNet task 2 | 不适配 | 75.35% | 75.35% | +0.00 pp | +1.11 pp |
| PhysioNet task 3 | mi_small_shift | 75.66% | 75.96% | +0.30 pp | +0.20 pp |
| PhysioNet task 4 | 不适配 | 65.15% | 65.15% | +0.00 pp | -1.11 pp |
| BCI IIIA | 不适配 | 65.31% | 65.31% | +0.00 pp | +6.12 pp |
| BCI IIIB | 不适配 | 84.29% | 84.29% | +0.00 pp | +1.90 pp |
| **六项等权平均** | — | **72.78%** | **72.98%** | **+0.20 pp** | **+1.35 pp** |

“相对15模型历史最佳”是每项任务分别与此前 15 个模型中最高的准确率比较。该表只比较点估计；基线没有逐试次预测，无法做历史最佳的配对检验。

### 被试配对不确定性

PhysioNet 的 95% 区间按 22 个测试被试进行 20,000 次配对聚类 bootstrap：

| 数据集 | 准确率变化 | 95% 聚类 bootstrap 区间 | 重采样变化 ≤ 0 的比例 |
|---|---:|---:|---:|
| PhysioNet task 1 | +0.91 pp | [-0.10, +1.92] pp | 4.99% |
| PhysioNet task 2 | +0.00 pp | [0.00, 0.00] pp | 100.00% |
| PhysioNet task 3 | +0.30 pp | [0.00, +0.61] pp | 4.29% |
| PhysioNet task 4 | +0.00 pp | [0.00, 0.00] pp | 100.00% |

BCI 两项回退到原结果，所以准确率变化恰为 0。预测文件未保留 BCI 被试 ID，不能诚实地给出被试聚类置信区间。以上区间没有针对六项任务及多轮探索作多重比较校正；测试集也已在前几轮中查看过，因此结果是探索性的，不能作为确认性显著性结论。

## 解释与限制

- 校准强度由验证数据选择，测试标签没有参与适配或选择；测试时只使用每个目标批次的无标签概率。
- task 1 与 task 3 的增益很小，平均只提高 0.20 pp；这次方法不足以改善所有数据集。
- 按每位测试被试的整批预测做校准属于 transductive 评估，需要部署时取得目标被试的一批无标签预测。它不是严格的逐试次、完全归纳式预测。
- 这轮在已有六项模型输出上做后处理，没有重训原始 EEG 模型。要判断适配是否有效，仍需预先固定新的一组外层测试被试，并在独立数据上复验。

## 代码与复现

- 实验代码：[information_maximization_tta.py](D:/programpack/eeg/information_maximization_tta.py)
- 依赖清单：[requirements_tta.txt](D:/programpack/eeg/requirements_tta.txt)
- 六项汇总：[comparison.csv](D:/programpack/result/new_method/information_maximization_tta/comparison.csv)
- 完整指标、验证候选和配对区间：[all_datasets.json](D:/programpack/result/new_method/information_maximization_tta/all_datasets.json)
- 各数据集概率与标签：[预测结果目录](D:/programpack/result/new_method/information_maximization_tta/)

在仓库根目录运行：

```powershell
python -m pip install -r eeg/requirements_tta.txt
python eeg/information_maximization_tta.py
```
