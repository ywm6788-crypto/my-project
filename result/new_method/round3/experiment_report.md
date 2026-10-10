# 六项 EEG 数据集综合优化报告

## 总体结果

本轮先在六个数据集统一测试无标签协方差对齐，再将按被试分组交叉验证的时间分段 FBCSP 迁移到 PhysioNet task 1、3、4；task 2 使用上一轮专项得到的结果。每项任务都只按验证集决定采用原方案、对齐方案或小比例融合。测试数据从未用于选择融合权重。

| 数据集 | 上轮最好 | 本轮验证 | 本轮测试 | 对上轮变化 | 对15模型最佳 |
|---|---:|---:|---:|---:|---:|
| PhysioNet task 1 | 70.61% | 81.62% | 70.91% [67.98%, 73.74%] | +0.30 pp | -1.01 pp |
| PhysioNet task 2 | 75.35% | 76.97% | 75.35% [72.63%, 78.08%] | +0.00 pp | +1.11 pp |
| PhysioNet task 3 | 73.74% | 81.41% | 75.66% [72.93%, 78.28%] | +1.92 pp | -0.10 pp |
| PhysioNet task 4 | 61.01% | 69.90% | 65.15% [62.22%, 68.18%] | +4.14 pp | -1.11 pp |
| BCI IIIA | 65.31% | 87.76% | 65.31% [51.02%, 77.55%] | +0.00 pp | +6.12 pp |
| BCI IIIB | 84.29% | 81.90% | 84.29% [79.05%, 89.05%] | +0.00 pp | +1.90 pp |

六项测试准确率的简单平均为 **72.78%**，上轮相同口径平均为 **71.72%**，提高 **+1.06 pp**。对照 15 模型表中每项任务的最佳值，平均差值为 **+1.15 pp**。三个 PhysioNet 任务获得测试提升：task 1 +0.30 pp、task 3 +1.92 pp、task 4 +4.14 pp；task 2 与两项 BCI 保持原结果。

最终审计对 22 名被试的配对聚类 bootstrap 分布做了精确枚举，替代本报告早期有限次随机重采样的近似值：task 4 相对第二轮方法的提升区间为 **+1.11 至 +7.47 pp**；task 2 相对第二轮特征方法为 **+9.39 至 +17.58 pp**。task 1 和 task 3 的区间都跨 0。因为测试划分此前已经查看过，这些比较仍然是探索性证据，不能当成独立确认。

## 新尝试

**无监督协方差对齐：** 对每个 PhysioNet 被试用其无标签 EEG 协方差估计收缩白化矩阵，再由原滤波器组分类器提取特征。BCI 文件没有保留被试 ID，因此将每个提供的数据分区视为一个无标签批次。验证集从 0%、25%、50%、75%、100% 的新方案概率权重中选择，与现有方案融合。

**被试分组时段 FBCSP：** 在 PhysioNet 训练集用 5 折 `GroupKFold` 按被试筛选 CSP 频带、早中晚时段和 SVM 正则强度，再与当前方案按验证结果融合。task 4 的验证集选择 25% 的 `classic_phases_plus_full_csp2`，测试准确率从 61.01% 到 65.15%。task 1 和 task 3 的分组验证没有支持加入该 CSP 候选，因此保留协方差对齐组合。

task 2 保留了前一轮的 **75% ShallowConvNet + 25% FBCSP** 专项方案（75.35%）。本轮统一协方差对齐在 task 2、task 4 和 BCI IIIB 的验证集未胜过原有方案，因此权重设为 0，避免测试退步。BCI IIIA 的新对齐候选使验证准确率由 77.55% 升到 87.76%，但测试仍为 65.31%，显示 49 个验证试次过少、验证选择的方差较大。

## 限制

本轮不能声称所有任务都超过 15 模型表的最佳值：PhysioNet task 1、3、4 仍略低于各自历史最佳，task 4 仍差 1.11 pp。历史基线没有逐试次预测，无法对其进行配对显著性检验；试次 bootstrap 区间也没有按被试聚类。同一固定测试集在前几轮已经被查看，因此本轮结果属于探索性结论，需要用新的外层被试划分或独立测试集复核。

对齐步骤没有使用任何测试标签，但它在每个测试被试上使用该被试整批无标签试次估计协方差；BCI 则使用整个无标签测试分区。部署时需要能取得目标被试/目标批次的一组无标签 EEG，这是 transductive 设定，应与逐试次、完全归纳式预测区分。

## 代码和结果

- 六项数据的结果表：[comparison.csv](D:/programpack/result/new_method/round3/comparison.csv)
- 完整结构化结果：[all_datasets.json](D:/programpack/result/new_method/round3/all_datasets.json)
- 协方差对齐实验：[domain_alignment_experiment.py](D:/programpack/eeg/domain_alignment_experiment.py)
- 被试分组 FBCSP 实验：[subject_cv_physionet_experiment.py](D:/programpack/eeg/subject_cv_physionet_experiment.py)
- 结果汇总脚本：[summarize_round3.py](D:/programpack/eeg/summarize_round3.py)
- 被试聚类 bootstrap 脚本：[cluster_bootstrap_all_physionet.py](D:/programpack/eeg/cluster_bootstrap_all_physionet.py)
- task 2 专项方法：[task2_experiment.py](D:/programpack/eeg/task2_experiment.py)

按仓库根目录运行：`python -m pip install -r eeg/requirements_new_method.txt`，然后运行 `python eeg/domain_alignment_experiment.py`、`python eeg/subject_cv_physionet_experiment.py`、`python eeg/cluster_bootstrap_all_physionet.py` 和 `python eeg/summarize_round3.py`。task 2 专项结果与预测文件位于 `result/new_method/task2_special/`。

执行环境为 Python 3.13、NumPy 2.5.3、SciPy 1.18.1、scikit-learn 1.9.1、MNE 1.13.2；task 2 神经网络重训使用 PyTorch 2.14.1 CPU。
