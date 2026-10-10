# EEG 实验结果

## 方法

TV-FB-Hybrid 将每个 trial 分成 5 个频带（8–30 Hz），并为每个频带提取三类特征：① 收缩空间协方差的切空间表示与对数频带功率；② 一对多 CSP 空间滤波后的对数方差；③ 分时段对数频带功率，双通道数据额外加入左右通道功率差。每个频带在验证集上选择特征分支和分类器（线性/RBF SVM，小样本低维特征还比较 KNN、随机森林和 MLP）；验证集再选择融合频带数。测试集只用于最终指标。PhysioNet 使用 FC/C/CP 区 15 个运动区电极，BCI IIIA/IIIB 使用提供的全部通道。

PhysioNet 的 epoch、过滤和被试划分与 eeg/dataset.py、eeg/train.py 一致。BCI IIIA/IIIB 使用现有 train/validation/test NPZ。所有对比使用 result 中原有的固定测试划分。

## 结果

| 数据集 | TV-FB-Hybrid | 95% 试次自助区间 | 15 模型最佳 | 对最佳差值 | 15 模型均值 | 对均值差值 |
|---|---:|---:|---:|---:|---:|---:|
| PhysioNet task 1 | 0.653 | [0.623, 0.683] | 0.719（shallowconvnet） | -6.67 个百分点 | 0.566 | +8.69 个百分点 |
| PhysioNet task 2 | 0.622 | [0.592, 0.653] | 0.742（shallowconvnet） | -12.02 个百分点 | 0.564 | +5.81 个百分点 |
| PhysioNet task 3 | 0.683 | [0.655, 0.712] | 0.758（shallowconvnet） | -7.47 个百分点 | 0.554 | +12.84 个百分点 |
| PhysioNet task 4 | 0.623 | [0.592, 0.653] | 0.663（shallowconvnet） | -3.94 个百分点 | 0.551 | +7.27 个百分点 |
| BCI IIIA | 0.653 | [0.510, 0.776] | 0.592（EEG-Conformer） | +6.12 个百分点 | 0.364 | +28.86 个百分点 |
| BCI IIIB | 0.719 | [0.657, 0.776] | 0.824（EEG-Deformer） | -10.48 个百分点 | 0.715 | +0.41 个百分点 |

六项任务的平均准确率：TV-FB-Hybrid 0.659，各组 15 模型最佳值的平均 0.716，各组 15 模型均值的平均 0.552。

## 统计说明

区间是新方法按测试试次重采样得到的 bootstrap 区间，并未按被试聚类。现有 15 模型 CSV 没有逐试次预测，因此无法计算 McNemar 配对检验；当前单次固定划分的结果也不足以宣称统计显著。和 15 模型均值相比，六项任务平均提升约 10.65 个百分点；和每组最佳模型相比，新方法只在 IIIA 超过最佳值，其他五项仍低于最佳模型。

## 复现

```powershell
python -m pip install -r eeg/requirements_new_method.txt
python eeg/new_method.py --all
```

各任务的超参数选择、混淆矩阵和逐试次预测保存在同目录 JSON/NPZ 文件中。
