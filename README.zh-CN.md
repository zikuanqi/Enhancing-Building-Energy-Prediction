[**English**](README.md) | [**中文**](README.zh-CN.md)

# KAN-Transformer 建筑能耗预测

[![CI](https://github.com/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach/actions/workflows/ci.yml/badge.svg)](https://github.com/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach/branch/main/graph/badge.svg)](https://codecov.io/gh/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **"基于KAN的Transformer学习网络用于建筑能耗预测"** 论文的官方代码复现 — 綦子宽，悉尼大学。

本仓库提供了论文中提出的 KAN-Transformer 架构的完整、可复现的 PyTorch 实现。该网络利用多变量时间序列——能耗负荷加上五个外生特征（温度、湿度、降水量、风速、入住率），对**四类建筑**（办公/住宅/商业/工业）进行多步逐小时能耗预测。

---

## 目录

- [亮点](#亮点)
- [架构总览](#架构总览)
- [数学公式](#数学公式)
- [项目结构](#项目结构)
- [安装指南](#安装指南)
- [数据格式](#数据格式)
- [快速开始](#快速开始)
- [配置系统](#配置系统)
- [训练细节](#训练细节)
- [评估与指标](#评估与指标)
- [对比方法（表1）](#对比方法表1)
- [模块实现细节](#模块实现细节)
- [可视化](#可视化)
- [测试](#测试)
- [CI/CD 与覆盖率](#cicd-与覆盖率)
- [引用](#引用)
- [许可证](#许可证)

---

## 亮点

| 特性 | 描述 |
|---|---|
| **多尺度时间分解** | 通过层次化移动平均将输入分解为趋势/季节/周/日/短期五个时间尺度流（第2.2节） |
| **动态Transformer（DyT）** | 学习门控残差 `Y = αX + βF(X)` 替代传统 Add & Norm，其中门控使用 LayerNorm + sigmoid（公式 4–6） |
| **无矩阵乘法稠密层** | 三值权重 `{-1, 0, +1}`，自适应阈值 + 直通估计器（公式 7–11） |
| **层次化 KAN 前馈网络** | B样条激活函数，128 → 256 → 128，每单元8个样条 × 16个基函数（公式 12–13） |
| **个性化输出头** | 共享主干上可选的逐建筑残差头，便于快速适配新建筑 |
| **10种对比方法** | 完整消融实验 + CNN-LSTM、LSTM-Attention 基线（表1复现） |
| **跨平台CI** | Linux / Windows / macOS × Python 3.10 / 3.11 / 3.12 = 9个并行任务 |
| **96% 测试覆盖率** | 59个单元测试覆盖所有模块，启用分支覆盖 |

---

## 架构总览

```
                        输入: (B, T, features)
                              │
                    ┌─────────▼──────────┐
                    │  多尺度时间分解     │
                    │  Multi-Scale       │
                    │  Decomposition     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  分量嵌入 +         │
                    │  跨尺度注意力融合   │
                    │  Component Embed   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  正弦位置编码       │
                    │  Positional Enc    │
                    └─────────┬──────────┘
                              │
               ┌──────────────▼──────────────┐
               │   KAN-Transformer Block ×N  │
               │  ┌────────────────────────┐ │
               │  │ DyT( 多头自注意力 )     │ │
               │  │ DyT( Multi-Head Attn ) │ │
               │  └───────────┬────────────┘ │
               │  ┌───────────▼────────────┐ │
               │  │ DyT( 层次化 KAN FFN )   │ │
               │  │ DyT( Hierarchical KAN )│ │
               │  └───────────┬────────────┘ │
               └──────────────┬──────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  均值池化           │
                    │  Mean Pooling      │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  无矩阵乘法稠密层   │
                    │  MatMul-Free Head  │
                    └─────────┬──────────┘
                              │
                    输出: (B, horizon, num_targets)
```

**数据流程：**

1. **多尺度时间分解** — 将原始输入序列通过层次化中心移动平均分解为五个时间分量（趋势、季节、周、日、短期）。
2. **分量嵌入** — 每个分量线性投影到 `d_model` 维，再通过跨分量注意力机制学习各时间尺度的重要性权重。
3. **位置编码** — 标准正弦编码注入时序位置信息。
4. **N × KAN-Transformer 块** — 每个块依次执行 DyT 包裹的多头自注意力和 DyT 包裹的层次化 KAN 前馈网络。
5. **无矩阵乘法投影头** — 时间维均值池化 → 三值权重稠密层，输出 `(horizon × num_targets)` 个预测值。

---

## 数学公式

### 目标函数（公式 1）

总损失为MSE加L2正则化：

```
L = (1/N) Σ ||y_pred - y_true||² + λ Σ ||θ||²
```

其中 λ 为权重衰减系数（默认 0.0001），通过 `AdamW` 优化器实现。

### 数据预处理（公式 2–3）

**线性插值**填补缺失值（公式 2）：

```
x_t = x_{t-k} + (t - (t-k)) / ((t+m) - (t-k)) · (x_{t+m} - x_{t-k})
```

**最小-最大归一化**（公式 3）：

```
x_norm = (x - x_min) / (x_max - x_min) ∈ [0, 1]
```

归一化范围仅在训练集上拟合，验证集/测试集复用训练范围，防止数据泄漏。

### 动态 Transformer 残差连接（公式 4–6）

用学习的门控替代传统的 Add & Norm：

```
Y_t = α_t · X_t + β_t · F_t(X_t)                    (4)
α_t = σ(W_α · LayerNorm(X_t) + b_α)                  (5)
β_t = σ(W_β · LayerNorm(F_t(X_t)) + b_β)             (6)
```

其中 σ 为 sigmoid 函数，F_t 为被包裹的子层（自注意力或 KAN-FFN）。

### 无矩阵乘法稠密层（公式 7–11）

将连续权重三值量化：

```
Ω_{j,i} ∈ {-1, 0, +1}
Q(M) = +1 如果 M > τ₊ ;  -1 如果 M < τ₋ ;  否则 0    (11)
```

其中 `τ± = ± α · std(M)` 为自适应阈值。前向传播使用累加形式（公式 10）：

```
Γ_i = Σ_{j: Ω=+1} v_j  -  Σ_{j: Ω=-1} v_j           (10)
```

梯度通过直通估计器（straight-through estimator）反向传播，梯度裁剪至 [-1, 1]。

### KAN 前馈网络（公式 12–13）

每个 KAN 单元对输入的 k 个线性投影施加 k 个学习的 B 样条激活函数：

```
z_i = Σ_{j=1..k} g_{i,j}(w_{i,j}^T x + b_{i,j})      (12)
g_{i,j}(s) = Σ_{l=1..L} c_{i,j,l} · B_l(s)            (13)
```

其中 `B_l(s)` 为均匀节点网格上的三次 B 样条基函数。附加 SiLU 残差连接辅助优化。层次化布局为 128 → 256 → 128，k=8 个样条，L=16 个基函数。

---

## 项目结构

```
KAN-Transformer-BECP/
├── .github/
│   └── workflows/
│       ├── ci.yml                # 跨平台CI：ruff + pytest + codecov
│       └── weekly.yml            # 每周依赖审计 + 回归检查
├── configs/
│   ├── default.yaml              # 完整训练配置（30轮，d_model=128）
│   ├── quick.yaml                # 快速测试配置（3轮，d_model=64）
│   └── transformer.yaml          # 原始Transformer基线配置
├── models/
│   ├── __init__.py               # 导出所有模型类
│   ├── dyt.py                    # 动态Transformer（DyT）层 — 公式 4–6
│   ├── matmul_free.py            # 三值无矩阵乘法稠密层 — 公式 7–11
│   ├── kan.py                    # KAN层 + 层次化KAN — 公式 12–13
│   ├── kan_transformer.py        # 完整的 proposed 网络
│   └── baselines.py              # 9种对比方法（表1）
├── utils/
│   ├── preprocessing.py          # 缺失值处理、异常检测、归一化、
│   │                             #   时间编码、滚动统计、度日法
│   ├── decomposition.py          # 多尺度时间分解
│   ├── data.py                   # EnergyDataset、DataLoader、合成数据生成器
│   ├── metrics.py                # MAPE / RMSE / MAE / R²
│   ├── visualize.py              # 图2（预测vs真实值）/ 图3（方法对比）
│   ├── callbacks.py              # 早停、结构化日志、种子固定
│   └── config.py                 # YAML加载器（深度合并）
├── experiments/
│   ├── compare.py                # 复现表1 — 训练全部10种方法
│   └── plot_figures.py           # 渲染图2/3 + 逐指标条形图
├── notebooks/
│   └── quickstart.ipynb          # 端到端 Jupyter 教程
├── tests/                        # 59个单元测试（96%覆盖率）
│   ├── test_smoke.py             # 对所有模型变体做前向+反向传播测试
│   ├── test_dyt.py               # DyT门控值范围、输出形状
│   ├── test_matmul_free.py       # 三值量化、公式10等价性
│   ├── test_kan.py               # B样条单位分割、KAN形状
│   ├── test_metrics.py           # 指标正确性（与已知值对比）
│   ├── test_preprocessing.py     # 插值、归一化、特征工程
│   ├── test_decomposition.py     # 分解重构恒等性
│   ├── test_callbacks.py         # 早停触发/重置、种子确定性、日志文件输出
│   ├── test_config.py            # YAML深度合并、点分路径获取、异常处理
│   ├── test_data.py              # 数据集形状、DataLoader、合成数据生成
│   └── test_visualize.py         # 绘图测试（使用matplotlib Agg后端）
├── train.py                      # 单模型训练入口
├── evaluate.py                   # 评估已保存的检查点
├── pyproject.toml                # 构建、lint、测试、覆盖率配置
├── requirements.txt              # pip依赖列表
├── LICENSE                       # MIT许可证
├── CHANGELOG.md                  # 变更日志
└── CONTRIBUTING.md               # 贡献指南
```

---

## 安装指南

### 前置要求

- Python ≥ 3.10
- pip（或 conda）

### 基础安装

```bash
git clone https://github.com/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach.git
cd Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach

# 可编辑安装（推荐用于开发）
pip install -e .
```

### 附加依赖

```bash
# 开发工具（pytest、pytest-cov、ruff）
pip install -e ".[dev]"

# Jupyter 笔记本支持（jupyter、seaborn）
pip install -e ".[notebook]"

# 全部安装
pip install -e ".[dev,notebook]"
```

### GPU 支持

默认 `pip install torch` 在某些平台只安装CPU版本。如需CUDA支持：

```bash
# 示例：CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 命令行工具

安装完成后，以下三个CLI命令可用：

| 命令 | 等价于 |
|---|---|
| `becp-train` | `python train.py` |
| `becp-evaluate` | `python evaluate.py` |
| `becp-compare` | `python experiments/compare.py` |

---

## 数据格式

### 合成数据（默认）

如果在配置路径下找不到CSV文件，系统会自动生成一个**合成数据集**，模拟论文的数据结构：

- **时长：** 2年逐小时数据（17,520个时间步）
- **4个负荷列：** `office`（办公）、`residential`（住宅）、`commercial`（商业）、`industrial`（工业）
- **5个外生特征：** `temperature`（温度）、`humidity`（湿度）、`precipitation`（降水）、`wind_speed`（风速）、`occupancy`（入住率）
- 具有真实的日/周/季节模式和可控噪声

这使得你无需获取真实数据即可立即运行所有实验。

### 真实数据

将CSV文件放在 `data/energy.csv`（或通过 `--data <路径>` 指定）。要求的格式：

| 列名 | 类型 | 描述 |
|---|---|---|
| `timestamp` | 日期时间 | 逐小时时间戳（解析为 `DatetimeIndex`） |
| `office` | 浮点数 | 办公建筑能耗（kWh） |
| `residential` | 浮点数 | 住宅建筑能耗 |
| `commercial` | 浮点数 | 商业建筑能耗 |
| `industrial` | 浮点数 | 工业建筑能耗 |
| `temperature` | 浮点数 | 室外温度（°C） |
| `humidity` | 浮点数 | 相对湿度（%） |
| `precipitation` | 浮点数 | 降水量（mm） |
| `wind_speed` | 浮点数 | 风速（m/s） |
| `occupancy` | 浮点数 | 入住率（0–1） |

预处理流水线自动处理：
- 缺失值线性插值（公式 2）
- 基于鲁棒z分数（MAD方法）的异常检测
- 最小-最大归一化，仅在训练集上拟合（公式 3）
- 特征工程：时间正弦/余弦编码、滚动统计（24h和168h窗口）、制热/制冷度日、入住率加权需求
- 时间顺序 70/10/20 训练/验证/测试分割

---

## 快速开始

### 1. 快速验证（CPU约2分钟）

```bash
python train.py --config configs/quick.yaml
```

使用 d_model=64 和 3 个 epoch —— 适合验证安装是否正确。

### 2. 完整 proposed 模型

```bash
python train.py --config configs/default.yaml
```

使用默认超参数训练 KAN-Transformer（d_model=128，30个epoch，余弦学习率调度）。

### 3. 复现表1

```bash
python experiments/compare.py --epochs 20
```

依次训练全部10种方法，计算每种方法的 MAPE / RMSE / MAE / R²，并将预测结果保存到 `predictions.npz`。

### 4. 生成图表

```bash
python experiments/plot_figures.py
```

读取 `predictions.npz` 并生成：
- **图2**：proposed 方法在每类建筑上的预测值 vs 真实值
- **图3**：所有方法在同一坐标轴上的对比
- **逐指标条形图**：各方法的并排指标比较

### 5. Jupyter 笔记本

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/quickstart.ipynb
```

交互式端到端教程：合成数据 → 预处理 → 训练 → 评估 → 可视化。

---

## 配置系统

项目使用**分层 YAML 配置**系统，具有深度合并语义：

```
configs/default.yaml  ←  你的覆盖YAML  ←  CLI参数
      (基础配置)          (深度合并)       (最高优先级)
```

### 默认配置

```yaml
experiment:
  name: kan_transformer_default
  seed: 42
  output_dir: checkpoints
  log_level: INFO

data:
  csv_path: data/energy.csv       # 文件不存在时自动使用合成数据
  window: 168                     # 1周的逐小时历史数据
  horizon: 24                     # 预测未来24小时
  batch_size: 64
  num_workers: 0
  rolling_windows: [24, 168]
  train_frac: 0.7
  val_frac: 0.1

model:
  name: proposed
  d_model: 128
  n_heads: 8
  num_layers: 3
  dropout: 0.1
  kan_hidden: [128, 256, 128]
  num_splines: 8
  num_basis: 16
  num_buildings: 1

training:
  epochs: 30
  lr: 0.001
  weight_decay: 0.0001            # 公式(1)中的 λ
  grad_clip: 5.0
  early_stopping:
    enabled: true
    patience: 8
    min_delta: 0.0001
  scheduler:
    name: cosine
    warmup_epochs: 2
```

### 创建自定义配置

只需指定要覆盖的键 — 其余全部从 `default.yaml` 继承：

```yaml
# configs/my_experiment.yaml
model:
  d_model: 256
  n_heads: 16
training:
  lr: 0.0005
  epochs: 50
```

```bash
python train.py --config configs/my_experiment.yaml
```

### CLI 覆盖

命令行参数优先级最高：

```bash
python train.py --config configs/default.yaml --epochs 50 --lr 5e-4 --batch_size 128
python train.py --resume checkpoints/proposed/checkpoint.pt
python experiments/compare.py --methods proposed transformer cnn_lstm
```

### 编程接口

```python
from utils.config import load_config, get

cfg = load_config("configs/default.yaml")
lr = get(cfg, "training.lr")                    # 0.001
patience = get(cfg, "training.early_stopping.patience")  # 8
missing = get(cfg, "some.nonexistent.key", default=42)   # 42
```

---

## 训练细节

### 优化器

- **AdamW**，权重衰减 λ = 0.0001（公式 1）
- 梯度裁剪，最大范数 5.0

### 学习率调度

- **余弦退火**（`CosineAnnealingLR`），覆盖整个训练周期
- 2个warmup epoch（可配置）

### 早停机制

- 监控验证集损失
- 耐心值（patience）：8个 epoch（可配置）
- 最小改善量（min_delta）：0.0001 — 小于此值的改善视为停滞

### 检查点保存

- **最佳模型**保存至 `checkpoints/<model>/best.pt`（最低验证损失）
- **最新模型**保存至 `checkpoints/<model>/checkpoint.pt`（每个epoch保存）
- 完整状态：模型权重 + 优化器状态 + 调度器状态 + epoch数 + 最佳验证损失

### 恢复训练

```bash
python train.py --resume checkpoints/proposed/checkpoint.pt
```

加载全部状态并从保存的 epoch 处继续训练。

### 结构化日志

每次训练同时输出到控制台和输出目录下的 `train.log` 文件。

### 确定性种子

`set_seed(42)` 为 Python `random`、NumPy 和 PyTorch（CPU + CUDA）设置种子，确保可复现性。

---

## 评估与指标

### 运行评估

```bash
python evaluate.py --model proposed --checkpoint checkpoints/proposed/best.pt
```

### 评估指标

所有指标在展平的预测数组上计算（所有建筑类型 × 所有预测步长）：

| 指标 | 公式 | 描述 |
|---|---|---|
| **MAPE** | `mean(\|y_true - y_pred\| / \|y_true\|)` | 平均绝对百分比误差 |
| **RMSE** | `sqrt(mean((y_true - y_pred)²))` | 均方根误差 |
| **MAE** | `mean(\|y_true - y_pred\|)` | 平均绝对误差 |
| **R²** | `1 - SS_res / SS_tot` | 决定系数 |

---

## 对比方法（表1）

本仓库实现了论文对比研究中的全部 10 种方法：

| 编号 | 方法 | KAN | DyT | 无矩阵乘法 | 模块 |
|---|---|---|---|---|---|
| 1 | **Proposed（KAN-Transformer）** | ✅ | ✅ | ✅ | `KANTransformer` |
| 2 | Transformer | ❌ | ❌ | ❌ | `PlainTransformer` |
| 3 | Transformer-KAN | ✅ | ❌ | ❌ | `TransformerKAN` |
| 4 | Transformer-DyT | ❌ | ✅ | ❌ | `TransformerDyT` |
| 5 | Transformer-MatMul-free | ❌ | ❌ | ✅ | `TransformerMatMulFree` |
| 6 | Transformer-KAN-MatMul-free | ✅ | ❌ | ✅ | `TransformerKANMatMulFree` |
| 7 | Transformer-KAN-DyT | ✅ | ✅ | ❌ | `TransformerKANDyT` |
| 8 | Transformer-DyT-MatMul-free | ❌ | ✅ | ✅ | `TransformerDyTMatMulFree` |
| 9 | CNN-LSTM | — | — | — | `CNNLSTM` |
| 10 | LSTM-Attention | — | — | — | `LSTMAttention` |

Transformer 变体共享一个 `_ConfigurableTransformer` 骨干网络，通过布尔标志（`use_kan`、`use_dyt`、`use_matmul_free`）控制，使每个组件的贡献可直接归因。

### 运行完整对比实验

```bash
# 全部10种方法
python experiments/compare.py --epochs 20

# 仅选择部分方法
python experiments/compare.py --methods proposed transformer cnn_lstm --epochs 20
```

结果保存至 `metrics.json`，预测保存至 `predictions.npz`。

---

## 模块实现细节

### `models/dyt.py` — 动态 Transformer 层

```python
class DyTLayer(nn.Module):
    """用动态门控残差包裹任意子层（公式 4–6）"""
    # α = sigmoid(W_α · LayerNorm(X) + b_α)
    # β = sigmoid(W_β · LayerNorm(F(X)) + b_β)
    # output = dropout(α * X + β * F(X))
```

- 门控 α 和 β **逐token逐通道**产生（d_model维）
- 两个门控通过 sigmoid 限制在 (0, 1) 范围内
- 门控后施加 Dropout

### `models/matmul_free.py` — 三值稠密层

```python
class MatMulFreeDense(nn.Module):
    """y = scale ⊙ (x @ Q(W).T) + bias"""
    # Q(W) 将连续权重量化为 {-1, 0, +1}
    # 自适应阈值：τ± = ± α · mean(|W|)
    # 反向传播：直通估计器，梯度裁剪至 [-1, 1]
```

- `ternary_accumulate()` 方法显式实现公式 10 用于验证
- `quantized_weight()` 导出可部署的三值矩阵
- 可学习的逐输出缩放因子补偿量化引起的幅度损失

### `models/kan.py` — KAN 层

```python
class KANLayer(nn.Module):
    """z_i = Σ_j g_{i,j}(w^T x + b)，其中 g 为学习的 B 样条"""
    # 每个输出单元 k=8 个样条函数
    # L=16 个三次 B 样条基函数，均匀节点网格
    # SiLU 残差连接保证优化稳定性
```

- `_b_spline_basis()` — Cox-de Boor 递归求值 B 样条基函数
- `HierarchicalKAN` — 三层 KANLayer 堆叠（128 → 256 → 128），带 Dropout

### `models/kan_transformer.py` — 完整 Proposed 网络

- `MultiScaleDecomposition` → `ComponentEmbedding`（含跨尺度注意力）→ `SinusoidalPositionalEncoding` → N × `KANTransformerBlock` → `MatMulFreeDense` 投影头
- 通过 `building_id` 参数支持可选的逐建筑个性化头
- 支持整数和批量张量形式的建筑ID

### `utils/preprocessing.py` — 数据管线

- `linear_interpolate_missing()` — 公式 2，通过 pandas 实现
- `detect_and_clean_anomalies()` — 鲁棒z分数（MAD），连续段→插值，孤立点→邻均值
- `minmax_normalize()` — 公式 3，可复用训练集拟合的范围
- `temporal_encoding()` — 小时、星期几、月份、季节的正弦/余弦对
- `rolling_statistics()` — 24h和168h窗口的滚动均值/方差/最大值
- `degree_days_and_demand()` — 制热度日(HDD)、制冷度日(CDD)、入住率加权需求
- `preprocess_energy_dataframe()` — 完整流水线 + 70/10/20 时序分割

### `utils/decomposition.py` — 多尺度分解

可微分 PyTorch 模块，使用 `F.conv1d` + 边缘复制填充：
- 层次化：依次剥离趋势 → 季节 → 周 → 日 → 短期残差
- 可配置周期（默认：日=24，周=168，季节=720，趋势=4320）

---

## 可视化

### 图2 — 预测值 vs 真实值

```python
from utils.visualize import plot_predictions_vs_true
plot_predictions_vs_true(y_true, y_pred, building_names=["office", "residential", ...])
```

### 图3 — 方法对比

```python
from utils.visualize import plot_method_comparison
plot_method_comparison(results_dict, building="office")
```

### 逐指标条形图

```python
from utils.visualize import plot_metric_bar
plot_metric_bar(metrics_dict, metric="MAPE")
```

或一次性生成所有图表：

```bash
python experiments/plot_figures.py
```

---

## 测试

### 运行全部测试

```bash
pytest tests/ -v
```

### 运行并查看覆盖率

```bash
pytest tests/ -v --cov=models --cov=utils --cov=experiments --cov-report=term
```

### 测试套件概览（59个测试）

| 文件 | 测试数 | 验证内容 |
|---|---|---|
| `test_smoke.py` | 3 | proposed、baseline、循环模型的前向+反向传播 |
| `test_dyt.py` | 3 | 门控值在(0,1)范围内、输出形状、Dropout效果 |
| `test_matmul_free.py` | 4 | 三值量化、公式10 ≡ 矩阵乘法等价性、梯度流 |
| `test_kan.py` | 4 | B样条单位分割、KAN输出形状、层次化KAN |
| `test_metrics.py` | 4 | MAPE/RMSE/MAE/R² 与已知值对比 |
| `test_preprocessing.py` | 5 | 插值正确性、归一化范围、特征工程 |
| `test_decomposition.py` | 3 | 各分量之和等于原始信号、输出形状、短序列处理 |
| `test_callbacks.py` | 8 | 早停触发/重置、种子确定性、日志文件输出 |
| `test_config.py` | 10 | YAML深度合并、点分路径获取、文件缺失异常、空文件覆盖 |
| `test_data.py` | 9 | 数据集形状、DataLoader批次、CSV加载/回退、确定性 |
| `test_visualize.py` | 6 | 绘图冒烟测试（均使用matplotlib Agg后端适配无头CI） |

### 覆盖率

当前覆盖率：**96%**（启用分支覆盖）。唯一排除的文件是CLI入口脚本（`experiments/compare.py`、`experiments/plot_figures.py`），它们通过端到端运行而非单元测试来验证。

---

## CI/CD 与覆盖率

### 持续集成

每次推送或拉取请求到 `main` 分支都会触发 **CI 流水线**（`.github/workflows/ci.yml`）：

1. **矩阵构建**：3个操作系统（Linux、Windows、macOS）× 3个Python版本（3.10、3.11、3.12）= **9个并行任务**
2. **代码检查**：`ruff check .`，规则集 E, F, I, B, UP, SIM
3. **测试**：`pytest tests/ -v`，带覆盖率收集
4. **覆盖率上传**：Codecov（仅ubuntu + Python 3.12）

### 每周审计

定时任务（`.github/workflows/weekly.yml`）每周一 UTC 06:00 运行：
- `pip-audit` 检查已知安全漏洞
- `pip list --outdated` 检查依赖版本过期
- 在最新依赖版本上运行完整测试套件
- 如有任何失败，自动创建 GitHub Issue

### 本地代码检查

```bash
ruff check .
ruff format .  # 自动格式化
```

---

## 引用

```bibtex
@inproceedings{qi2025kan,
  title     = {A KAN-based Transformer learning network for building
               energy consumption prediction},
  author    = {Qi, Zikuan},
  year      = {2025},
  note      = {International Conference paper, University of Sydney}
}
```

---

## 许可证

本项目采用 [MIT 许可证](LICENSE) — 可自由用于学术和商业目的，需保留版权声明。
