# 黎曼几何迁移尝试：六项 EEG 实验

## 结论

本轮参考了两类已发表的 EEG 解码思路：SPD 协方差矩阵上的黎曼分类，以及跨被试的 Riemannian Alignment（RA）。切空间分类已出现在此前的模型集合中，因此本实验主要检验新增的被试级仿射不变对齐是否能在现有方法之上增加收益。[Barachant 等人的黎曼几何分类工作](https://pubmed.ncbi.nlm.nih.gov/22010143/)提出 MDRM 与切空间 LDA；[Zanini 等人的迁移学习工作](https://pubmed.ncbi.nlm.nih.gov/28841546/)通过仿射变换将不同被试/会话的协方差中心化。

实现采用每频带的 EEG trial 协方差、按训练被试等权估计的仿射不变参考均值、被试级部分/完全对齐，以及切空间向量上的 LinearSVC。超参数先由训练集内的五折被试分组交叉验证选择；候选还必须在验证集上比当前模型高出一个配对标准误才会替换它。

**候选方法在六项验证集上的准确率都低于当前方案，因此六项均未启用。最终测试结果保持不变：平均准确率 72.98%，变化 0.00 pp。此次没有获得准确率提升。**门控避免了把较差候选部署到测试集，但不代表候选方法有效。

## 结果

表中的“RA 候选测试”只用于最终诊断，未参与任何选择；“最终测试”是严格按验证门控选出的结果。

| 数据集 | 验证：当前方案 | 验证：RA 候选 | RA 候选测试诊断 | 验证是否采用 | 最终测试 |
|---|---:|---:|---:|---:|---:|
| PhysioNet task 1 | 82.63% | 71.31% | 69.70% | 否 | 71.82% |
| PhysioNet task 2 | 76.97% | 65.86% | 62.83% | 否 | 75.35% |
| PhysioNet task 3 | 81.82% | 78.59% | 73.54% | 否 | 75.96% |
| PhysioNet task 4 | 69.90% | 59.39% | 64.55% | 否 | 65.15% |
| BCI IIIA | 87.76% | 61.22% | 65.31% | 否 | 65.31% |
| BCI IIIB | 81.90% | 62.86% | 61.90% | 否 | 84.29% |
| **六项等权平均** | — | — | — | — | **72.98%** |

RA 候选的最佳配置由训练集交叉验证选出：PhysioNet 四项均为 `alpha=1.0, C=0.01`；BCI IIIA 为 `alpha=0.5, C=0.01`；BCI IIIB 为 `alpha=1.0, C=0.1`。候选在验证集上全面落后，测试诊断也没有显示可泛化的优势。

目前最终平均比 15 个历史基线模型各任务的最佳准确率平均高 1.35 pp；这是前面轮次已经取得的结果，本轮没有进一步提高。历史基线没有逐试次预测，无法对历史最佳值做配对显著性检验。所有固定测试划分也已在前几轮查看，因此汇总仍属探索性结果。

## 评估边界

- PhysioNet 用训练被试分组交叉验证；验证与测试目标域按被试分别使用无标签协方差做适配。
- BCI IIIA/IIIB 的已保存切分没有被试 ID，且原始划分按试次分层，因此只能把整个划分当作一个批次，不能声称做了被试级对齐。
- 测试标签只用于最终候选诊断和报告指标；超参数选择与是否采用候选完全由训练集交叉验证和验证集决定。
- 一个合理但尚未证实的解释是：仿射对齐移除了部分有判别作用的被试协方差结构，或当前切空间线性分类器与这些数据的特征分布不匹配。要区分原因，需要在新的外层被试划分上做消融实验。

## 代码与复现

- 实验代码：[riemannian_alignment_experiment.py](D:/programpack/eeg/riemannian_alignment_experiment.py)
- 依赖清单：[requirements_riemannian.txt](D:/programpack/eeg/requirements_riemannian.txt)
- 六项对照表：[comparison.csv](D:/programpack/result/new_method/riemannian_alignment/comparison.csv)
- 完整参数、验证结果、预测和不确定性：[all_datasets.json](D:/programpack/result/new_method/riemannian_alignment/all_datasets.json)
- 各数据集预测文件：[预测结果目录](D:/programpack/result/new_method/riemannian_alignment/)

在仓库根目录安装依赖并运行：

```powershell
python -m pip install -r eeg/requirements_riemannian.txt
python eeg/riemannian_alignment_experiment.py
```

实验环境：Python 3.13.5、NumPy 2.5.3、SciPy 1.18.1、scikit-learn 1.9.1、MNE 1.13.2。临时依赖安装在工作区内，实验完成后已清理；以上依赖清单可重新安装。
