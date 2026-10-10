# 第二轮 EEG 实验报告

PhysioNet 第二轮同时计算运动区 ROI 的 TV-FB-Hybrid 与全 64 通道的多频带 CSP 对数方差/分时段功率；验证集选择两路概率融合权重。task 1 还纳入一份按原设置训练的 ShallowConvNet 融合候选，并由验证集决定是否选择。BCI IIIA 使用 TV-FB-Hybrid；BCI IIIB 训练三个 EEG-Deformer 随机种子，以 seed 42 为主模型，只在它与 TV-FB-Hybrid 之间按验证集选择融合权重；seed 43/44 用于稳定性诊断。测试集只做最终评估。

| 数据集 | 第二轮方法 | 测试准确率 | 95% 试次 bootstrap 区间 | 测试宏 F1 | 基线最佳 | 对最佳差值 | 可用基线均值 | 对均值差值 | 验证准确率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PhysioNet task 1 | full_channel_filterbank + ShallowConvNet; weight_right=0.55 | 0.706 | [0.677, 0.735] | 0.706 | 0.719（shallowconvnet） | -1.31 pp | 0.566 | +14.05 pp | 0.782 |
| PhysioNet task 2 | ROI_filterbank + all_channel_CSP_power; weight_right=0.25 | 0.618 | [0.588, 0.648] | 0.616 | 0.742（shallowconvnet） | -12.42 pp | 0.564 | +5.41 pp | 0.703 |
| PhysioNet task 3 | ROI_filterbank + all_channel_CSP_power; weight_right=0.50 | 0.737 | [0.711, 0.765] | 0.736 | 0.758（shallowconvnet） | -2.02 pp | 0.554 | +18.30 pp | 0.810 |
| PhysioNet task 4 | ROI_filterbank + all_channel_CSP_power; weight_right=0.60 | 0.610 | [0.579, 0.640] | 0.610 | 0.663（shallowconvnet） | -5.25 pp | 0.551 | +5.96 pp | 0.677 |
| BCI IIIA | TV-FB-Hybrid full existing feature selection | 0.653 | [0.510, 0.776] | 0.653 | 0.592（EEG-Conformer） | +6.12 pp | 0.364 | +28.86 pp | 0.776 |
| BCI IIIB | TV-FB-Hybrid + EEG-Deformer seed 42; weight_right=0.65 | 0.843 | [0.795, 0.890] | 0.843 | 0.824（EEG-Deformer） | +1.90 pp | 0.715 | +12.79 pp | 0.819 |

六项任务平均：第二轮方法 0.695；各任务基线最佳值均值 0.716；各任务可用基线行均值的平均 0.552。因此第二轮平均高于基线行均值约 14.23 个百分点，但低于每组最佳模型均值约 2.16 个百分点。

## 限制

基线 CSV 只保存汇总指标，没有原模型逐试次预测，因而不能对本实验和历史基线做配对显著性检验。每组只保留一个固定测试划分；PhysioNet 的试次还嵌套于被试，试次级置信区间会低估被试间不确定性。模型结构/融合方案都由验证集选择，测试集没有参与选择；结果不能据此声称跨划分或统计显著提升。

同一固定测试划分在多个实验轮次中被反复评估和查看，所以即使每轮超参数由验证集选择，整套实验仍是探索性结果；最终方案需要在新的外层划分或独立数据集复核。

task 1 另按原设置完整重训 ShallowConvNet；验证集在该融合（78.18%）与 ROI/全通道特征融合（77.37%）间选择前者，因此 70.61% 的测试结果进入主表。由于 CPU 训练成本过高，其余 PhysioNet 任务没有重训神经基线。task 1 诊断 JSON/NPZ 和权重仍保存在 physionet_task_1_shallow_blend_diagnostic* 文件。

置信区间按试次 bootstrap 计算；PhysioNet 的试次来自同一批被试，未按被试聚类，区间可能偏窄。

复现环境依赖见 [requirements_new_method.txt](D:/programpack/eeg/requirements_new_method.txt)。从仓库根目录完整重跑：`python -m pip install -r eeg/requirements_new_method.txt`，然后 `python eeg/second_round.py`。只用已有逐试次文件重建报告：`python eeg/second_round.py --reuse-results`。

完整代码：[second_round.py](D:/programpack/eeg/second_round.py)；原始特征方法：[new_method.py](D:/programpack/eeg/new_method.py)。模型权重、逐试次概率和 JSON 选择记录均在本目录。
