# 基于持续同调的拓扑特征识别及其应用

> 浙江大学大学生创新训练项目（SRTP）· 2026.03 – 2027.05
> 本仓库是该项目的**代码、实验脚本、结果与阅读笔记**汇总，围绕一条主线：
> **持续同调（Persistent Homology）特征在形状数据与时间序列上的识别能力及其边界。**

## 项目简介

持续同调能够在多尺度上刻画点云、图像、网络等数据的拓扑不变量（连通分量、环、空腔），
并且对噪声与采样扰动具有稳定性。本项目以欧氏空间中的 PH 为起点，研究它向更复杂情形推广时的**适用条件**：

- 拓扑特征在什么条件下**优于**传统几何特征？采样规模、噪声、旋转等扰动如何影响其稳定性？
- 面对时间序列这类"没有形状"的数据，如何构造点云（延迟嵌入）并保持拓扑信息的可判别性？
- 拓扑特征的收益是普适的，还是依赖任务本身的结构？

## 方法

| 环节 | 做法 |
|---|---|
| 拓扑特征 | Vietoris–Rips 过滤 + 持续同调（`ripser`）；提取 WD/BD 持续度、top-k 持续度、持续环数、Betti 曲线 |
| 几何对照 | FPFH、法线直方图、凸包统计、PCA 主轴 |
| 时序嵌入 | Takens 延迟嵌入：互信息/自相关定延迟 $\tau$，假近邻法定嵌入维数 $m$ |
| 分类器 | SVM（RBF）、Random Forest、XGBoost/LightGBM（2A 冲刺）、轻量 MLP（1C） |
| 防泄漏 | 按物体 ID 的 GroupKFold（1C）、UCR 官方 train/test 划分（2A） |

## 结果选览

**① 合成点云：拓扑特征 vs 几何特征**（`results/exp1b_topo_vs_geo/`）

![拓扑 vs 几何](results/exp1b_topo_vs_geo/topo_vs_geo_comparison.png)

左上：分类准确率；右上：噪声鲁棒性；左下：旋转鲁棒性；右下：特征维度构成。
拓扑特征（25 维）在噪声与旋转扰动下明显更稳，几何特征（67 维）在干净数据上接近但抗扰动差。

**② ModelNet10：真实三维物体的拓扑 / 几何 / 融合特征**（`results/exp1c_modelnet10/`）

![ModelNet10](results/exp1c_modelnet10/modelnet10_comparison.png)

按物体 ID 做 GroupKFold（同一物体不跨训练/测试），**融合特征最好**。

**③ ECG200：真实时间序列的拓扑 / 时频 / 融合特征**（`results/exp2a_ecg200/`）

![ECG200](results/exp2a_ecg200/ecg200_comparison.png)

延迟嵌入 + PH 得到拓扑特征，与常规时频统计量对照；融合特征优于任何单一特征。
参数扫描（$(m,\tau)$ 网格上的 H1 持续环数）见 `results/exp2a_synth_scan/synth_param_scan_heatmaps.png`。

## 仓库结构

```
src/
  ph_pipeline.py                         持续同调核心工具（点云生成 / PH / 向量化 / 绘图）
  experiment_1a_param_scan.py            参数与计算复杂度扫描：子采样比例 × maxdim
  experiment_1a_threshold_sensitivity.py 绝对 WD 阈值的稳健性分析
  experiment_1b_topo_vs_geo.py           合成点云：拓扑特征 vs 几何特征
  experiment_1c_modelnet10.py            ModelNet10 真实物体分类（拓扑 / 几何 / 融合）
  experiment_2a_time_series.py           合成动力学时序生成器（周期 / 拟周期 / 混沌 / 噪声）
  experiment_2a_embedding.py             延迟嵌入 → PH 管道（m、τ 选取）
  experiment_2a_synth_scan.py            合成动力学批量参数扫描（(m,τ) 网格）
  experiment_2a_criterion.py             持续环判据优化（环数 / top-k / 分位数）
  experiment_2a_ecg200.py                ECG200 真实时序二分类实证
  experiment_2a_ecg200_boost.py          ECG200 特征扩展与集成冲刺
  pipeline_paper1_dna.py                 …论文 pipeline 复现（见下）
  pipeline_paper2_jet_tagging.py
  pipeline_paper3_topogat.py
  pipeline_paper4_3dphdl.py
  pipeline_paper5_filtration_learning.py
run_subtask_a.py / run_all_pipelines.py / run_stability.sh   批量运行入口

results/
  exp1a_param_scan/                      参数扫描结果与图（含阈值稳健性子目录）
  exp1b_topo_vs_geo/                     拓扑 vs 几何：汇总表、对比图、混淆矩阵、特征矩阵
  exp1c_modelnet10/                      ModelNet10 汇总表、对比图、特征矩阵
  exp2a_synth_scan/                      合成动力学参数扫描热力图与汇总表
  exp2a_criterion/                       判据优化：持续图缓存
  exp2a_ecg200/                          ECG200 汇总表、对比图、冲刺结果
  paper1..paper5_*.png                   pipeline 复现输出
  pipeline复现参考结果/                   同一批复现图的归档副本

docs/
  阶段2A_延迟嵌入参数选取.md              嵌入参数选取的设计说明
  CTDA阅读笔记理论部分整合.tex/.pdf       《Computational Topology for Data Analysis》阅读笔记
  CTDA第三章算法部分.tex/.pdf             第三章（算法部分）笔记
  CTDA理论/算法部分重点梳理（手写）.pdf     手写梳理
  项目进度以及后续计划2026.7.22.txt        项目进度记录

data/ecg200/                             UCR ECG200（公开数据，随仓库提供）
STAGE1_SUMMARY.md                        阶段 1 总结：三维场景拓扑识别
STAGE2A_SUMMARY.md                       阶段 2A 总结：时序拓扑表征
```

## 论文 pipeline 复现

项目起步阶段复现了 5 条公开的 TDA 应用流水线，用于校准自己的实现：

| 脚本 | 复现对象 |
|---|---|
| `pipeline_paper1_dna.py` | DNA 结构的持续同调分析（Persistent Homology: A Pedagogical Introduction） |
| `pipeline_paper2_jet_tagging.py` | 喷注 tagging 中的拓扑特征 |
| `pipeline_paper3_topogat.py` | TopoGAT：图注意力 + 拓扑特征选择（下图） |
| `pipeline_paper4_3dphdl.py` | 3D-PHDL：三维形状的持续同调描述子 |
| `pipeline_paper5_filtration_learning.py` | 自适应过滤学习（NeurIPS 2023, Adaptive Topological Feature via PH Filtration Learning） |

![TopoGAT 复现](results/paper3_topogat_selection.png)

输出图见 `results/paper*_*.png`。

## 数据说明

| 数据 | 来源 | 是否随仓库提供 |
|---|---|---|
| 合成点云（circle / disk / torus / sphere / porous_block） | `src/ph_pipeline.py` 内置生成器（固定随机种子） | 无需下载 |
| 合成动力学时序（周期 / 拟周期 / 混沌 / 纯噪声） | `src/experiment_2a_time_series.py` | 无需下载 |
| ECG200（UCR Time Series Classification Archive） | 公开数据集 | ✅ `data/ecg200/` |
| ModelNet10 | ModelNet 官方发布（10 类 CAD 模型，`.off`） | ❌ 体积大，自行下载到 `data/modelnet10/` |

## 复现

```bash
pip install -r requirements.txt

python src/experiment_1a_param_scan.py           # 参数扫描
python src/experiment_1a_threshold_sensitivity.py
python src/experiment_1b_topo_vs_geo.py          # 拓扑 vs 几何
python src/experiment_1c_modelnet10.py           # ModelNet10（需先准备数据）
python src/experiment_2a_synth_scan.py           # 合成动力学扫描
python src/experiment_2a_criterion.py            # 判据优化
python src/experiment_2a_ecg200.py               # ECG200 实证
```

所有生成过程使用固定随机种子；每个结果目录下 `*.csv` / `*.json` 为原始结果，`*.png` 为对应图。

## 阶段性认识

- **特征互补而非替代**：形状数据上拓扑特征对形变、噪声与旋转更稳健；几何特征对局部细节更敏感，
  二者融合通常优于任何单一特征（`results/exp1b_topo_vs_geo/`、`results/exp1c_modelnet10/`）。
- **参数存在"够用即可"的平台区**：过滤步长、子采样比例、嵌入参数 $(m,\tau)$ 超过某个阈值后，
  计算代价继续上升而判别力不再提升（`results/exp1a_param_scan/`、`results/exp2a_synth_scan/`）。
- **收益依赖任务本身的结构**：任务没有拓扑结构时，PH 更接近在描述噪声；
  时序上的适用边界由 `results/exp2a_*` 给出，判据（环数 / top-k / 分位数）的选择直接决定可用性。

## 许可

MIT License，见 [LICENSE](LICENSE)。
