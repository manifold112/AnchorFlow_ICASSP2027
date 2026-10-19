<div align="center">

<img src="docs/assets/hero.svg" alt="AnchorFlow：面向跨设备声学场景分类的场景条件设备流建模" width="100%" />

### 超越设备不变性：面向跨设备声学场景分类的场景条件设备流建模

**Peihong Zhang · Yuxuan Liu · Yiqiang Cai · Yizhou Tan · Shengchen Li**  
西交利物浦大学

[**论文**](docs/paper.pdf) · [**方法详解**](docs/method.md) · [**实验结果**](docs/results.md) · [**接入指南**](docs/integration.md) · [**English**](README.md)

</div>

<br>

<img src="docs/assets/highlights.svg" alt="论文结果：未见设备平均提升 3.31 个百分点；未见麦克风响应提升 3.22 个百分点；不增加推理参数" width="100%" />

## 设备变化，也可以成为训练信号

不同录音设备会改变声学线索的呈现方式，而且这种变化与具体场景有关。单纯压低设备差异，可能同时削弱区分机场、地铁站、公园等场景所需的信息。

**AnchorFlow 在每个场景内部显式建模设备变化。** 方法为“场景 × 设备”维护分布锚点，在同场景设备条件之间构造解析路径，并保留样本在源分布中的标准化残差。训练时约束场景损失沿这些路径的变化，让预测器适应设备条件变化。

设备标签仅在训练阶段使用。推理仍由原有编码器和场景预测器完成。

| 设计 | 方法 | 作用 |
| :--- | :--- | :--- |
| 场景条件分布锚点 | 按场景与设备联合索引，维护一阶、二阶原始矩的 EMA | 允许设备变化因场景而异 |
| 保留残差的设备流 | 在线性插值的均值与尺度之间保留标准化残差 | 无需学习额外传输网络或数值 ODE 求解器 |
| 有限差分路径正则 | 约束相邻路径位置上的场景损失斜率平方 | 使用普通一阶反向传播 |
| 分离的路径几何 | 特征、锚点、路径均停止梯度 | 辅助分支更新共享预测器 |

## 方法总览

<img src="docs/assets/architecture.svg" alt="AnchorFlow 方法架构：分布锚点、设备路径、确定性共享预测头和有限差分路径正则" width="100%" />

源设备与目标设备属于同一场景。固定样本残差后，同时插值均值与标准差：

$$
r_i=\frac{\operatorname{sg}(h_i)-\mu_s}{\sigma_s},\qquad
z_i(\tau)=\mu_\tau+\sigma_\tau\odot r_i.
$$

对路径位置 $\tau$ 与 $\tau+\delta$ 计算场景损失的有限差分，并将其平方作为路径能量。辅助预测器共享原预测头参数，关闭随机掩码并固定 BatchNorm 统计量。

## 论文结果

下列数值来自随附论文 Table 1 与 Table 3，单位为准确率百分比，提升为百分点。完整的均值、标准差、兼容性比较与消融见 [实验结果](docs/results.md)。

| 基准 | 骨干 | 未见设备 Base | 未见设备 +AF | 提升 |
| :--- | :--- | ---: | ---: | ---: |
| DCASE 2020 | CP-Mobile | 66.57 | **69.70** | +3.13 |
| DCASE 2020 | BC-ResNet-ASC-8 | 64.34 | **67.78** | +3.44 |
| DCASE 2020 | TF-SepNet-64 | 64.14 | **67.85** | +3.71 |
| DCASE 2020 | PaSST-S | 74.34 | **77.02** | +2.68 |
| DCASE 2024 | CP-Mobile | 53.10 | **56.37** | +3.27 |
| DCASE 2024 | BC-ResNet-ASC-8 | 52.00 | **55.57** | +3.57 |
| DCASE 2024 | TF-SepNet-64 | 53.70 | **57.62** | +3.92 |
| DCASE 2024 | PaSST-S | 55.90 | **58.62** | +2.72 |
| CochlScene + MicIRP | 论文外部验证设置 | 65.22 | **68.44** | +3.22 |

## 快速开始

使用 Python 3.11 及以上版本，先安装适合当前硬件的 PyTorch，再安装项目：

```bash
python -m pip install -e .
```

将 1 秒 log-mel 片段保存为 NumPy 数组，并为每个片段记录场景、设备和录音标识。DCASE 2020 的同一条 10 秒录音需要对应十个片段，评估时先平均预测概率。

```bash
anchorflow validate-data --manifest data/train.csv \
  --protocol configs/protocols/dcase2024.yaml --source-only

anchorflow train --config configs/default.yaml \
  --protocol configs/protocols/dcase2024.yaml \
  --train-csv data/train.csv --val-csv data/val_source.csv \
  --feature-root data --output runs/tau22_af_seed0 --device cuda --seed 0

anchorflow evaluate --checkpoint runs/tau22_af_seed0/best.pt \
  --manifest data/eval.csv --feature-root data --device cuda

anchorflow export --checkpoint runs/tau22_af_seed0/best.pt \
  --output checkpoints/scene_model.pt
```

在已有模型中接入时，将网络拆分为输出二维特征的 `encoder` 与接收这些特征的 `head`。配置中的 `model.factory` 指向相应构造函数；直接接入示例见 [`examples/integrate.py`](examples/integrate.py)。

## 方法参数

| 参数 | 数值 |
| :--- | ---: |
| 正则权重 $\lambda_{\mathrm{AF}}$ | 0.3 |
| 有限差分步长 $\delta$ | 0.20 |
| 每个样本的目标设备数 $K$ | 1 |
| EMA 衰减 $\beta$ | 0.99 |
| 锚点最小样本量 | 32 |
| 方差下界 | 1e-6 |
| 数值稳定项 | 1e-5 |
| 预热 | 1 epoch |

保持基线与 AnchorFlow 的训练设置一致。训练、超参数选择和模型选择均使用源设备信息。

## 文档导航

- [数学定义与梯度路径](docs/method.md)
- [数据组织与设备划分](docs/data.md)
- [模型接入、梯度累积与分布式锚点](docs/integration.md)
- [论文实验、兼容性与消融](docs/results.md)
- [贡献规范](CONTRIBUTING.md)

代码采用 [MIT License](LICENSE)。论文与第三方数据保留各自权利。引用信息见 [英文首页](README.md#citation)。
