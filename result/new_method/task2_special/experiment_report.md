# PhysioNet task 2 专项实验报告

## 结果

| 方法 | 测试准确率 | 95% 区间 | 相对 15 模型最佳值 |
|---|---:|---:|---:|
| 原 15 模型表最佳：ShallowConvNet | 74.24%（735/990） | 原始 CSV 无逐试次区间 | — |
| 第二轮 TV-FB-Hybrid 特征融合 | 61.82%（612/990） | 试次 bootstrap：[58.79%, 64.85%] | −12.42 pp |
| 本轮验证集选定组合 | **75.35%（746/990）** | 试次 bootstrap：[72.63%, 78.08%]；精确被试聚类 bootstrap：[70.40%, 80.30%] | **+1.11 pp（多 11 个正确试次）** |

本轮组合在 495 个验证试次上的准确率为 76.97%。模型选择结果为 `narrow_mu_beta_full_csp2 + ShallowConvNet_seed42; right_weight=0.75`，即将 ShallowConvNet 与分组交叉验证选出的 FBCSP 概率按 75%/25% 融合。测试集混淆矩阵为 `[[395, 102], [142, 351]]`，宏 F1 为 75.30%，Cohen's κ 为 0.507。

相对上一轮 task 2 特征方法，本轮提高 13.54 个百分点。按 22 名测试被试进行配对聚类 bootstrap，差值的 95% 区间为 **+9.39 到 +17.58 pp**。这项区间比较的是两种特征/融合方法，不能替代对历史最佳 ShallowConvNet 的显著性检验。

## 方法与实验设置

新特征分支使用 4–40 Hz 原始带通信号和 15 个运动区通道，构造经典、窄带 Mu/Beta 与宽带 Mu/Beta 滤波器组，并提取完整试段或早、中、晚三个时间阶段的 CSP 对数方差。估计每个类别的协方差时先计算每个训练被试的均值，再让各被试等权，降低训练被试试次数量差异带来的偏重。

特征配置和线性 SVM 正则强度先用训练集内 5 折 `GroupKFold` 按被试筛选。三个排名靠前的配置再与同划分、原预处理（8–30 Hz、64 通道）下重训的 ShallowConvNet 进入验证集选择；验证集只搜索单模型和两模型间的 25%、50%、75% 粗粒度融合权重。测试集未用于配置或权重选择。训练集、验证集和测试集分别包含 76、11、22 名被试，试次数为 3,433、495、990。完整运行耗时约 23 分钟。

分组交叉验证中最佳 CSP 配置的平均被试准确率为 63.41%；三个最终入围 CSP 配置在原验证集的准确率为 64.24%–66.67%。本轮提升主要来自复现的 ShallowConvNet，CSP 分支作为 25% 的辅助概率输入。因此结果支持“验证选择的融合方案在该固定测试划分上优于历史记录”，不支持把收益归因于 CSP 单独优于神经网络。

## 统计解释

相对 CSV 中历史最佳值的观测差为 +1.11 pp，但原 CSV 只保存总体准确率和混淆矩阵，没有历史模型逐试次预测，无法做配对显著性检验；本轮被试聚类置信区间也覆盖 74.24%。所以目前**不能声称已显著击败历史最佳模型**。相对上一轮特征方法的配对被试 bootstrap 区间不跨 0，但同一固定测试集在此前实验中已经被查看过，整轮研究仍属于探索性结果。确认对最佳模型的提升需要独立测试集或新的外层被试划分，并保存两方法逐试次预测后再做配对检验。

## 复现与文件

从仓库根目录执行：

```powershell
python -m pip install -r eeg/requirements_new_method.txt
python eeg/task2_experiment.py
python eeg/analyze_task2_results.py
```

- 实验代码：[task2_experiment.py](D:/programpack/eeg/task2_experiment.py)
- 被试聚类分析代码：[analyze_task2_results.py](D:/programpack/eeg/analyze_task2_results.py)
- 结构化主结果：[physionet_task_2_special.json](D:/programpack/result/new_method/task2_special/physionet_task_2_special.json)
- 逐试次预测与概率：[physionet_task_2_special_predictions.npz](D:/programpack/result/new_method/task2_special/physionet_task_2_special_predictions.npz)
- 被试聚类比较：[clustered_comparison.json](D:/programpack/result/new_method/task2_special/clustered_comparison.json)
- ShallowConvNet 权重：[shallowconvnet_task2_seed42.pt](D:/programpack/result/new_method/task2_special/shallowconvnet_task2_seed42.pt)

试次级区间和原始配对分析使用 20,000 次重采样；最终审计又对经验被试聚类 bootstrap 分布做了精确枚举，故表中的被试级区间为精确值。试次级区间把试次近似看作独立样本，会低估被试间差异；被试聚类区间以测试被试为重采样单位。该固定拆分和该 15 模型表的比较均未建立跨数据划分的泛化结论。
