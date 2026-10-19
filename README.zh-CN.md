<div align="center">

<h1>AnchorFlow</h1>
<h3>Beyond Device Invariance: Scene-Conditioned Device Flow Modeling<br>for Cross-Device Acoustic Scene Classification</h3>

<p>
  <a href="docs/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square" alt="Paper PDF" /></a>
  <a href="docs/method.md"><img src="https://img.shields.io/badge/Method-Documentation-286a91?style=flat-square" alt="Method documentation" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-555555?style=flat-square" alt="MIT License" /></a>
</p>

[Peihong Zhang](https://orcid.org/0009-0002-9561-1412),
[Yuxuan Liu](https://orcid.org/0009-0007-1043-416X),
[Yiqiang Cai](https://orcid.org/0009-0004-7062-7850),
[Yizhou Tan](https://orcid.org/0000-0002-4103-8115),
[Shengchen Li](https://orcid.org/0000-0002-2488-298X)

Xi'an Jiaotong-Liverpool University

[English](README.md) / [简体中文](README.zh-CN.md)

</div>

[![AnchorFlow 方法总览](docs/figures/method.png)](docs/figures/method.png)

<p align="center"><em>场景条件分布锚点与保留残差的设备流。原图来自论文。</em></p>

> **概述** AnchorFlow 将同一场景内的设备差异作为训练信号，在不同设备分布之间构造解析路径，
> 并利用有限差分约束场景预测沿路径的损失变化。推理模型保持不变。

<p align="center">
  <a href="#快速开始">快速开始</a> &nbsp;·&nbsp;
  <a href="#实验结果">实验结果</a> &nbsp;·&nbsp;
  <a href="docs/integration.md">模型接入</a> &nbsp;·&nbsp;
  <a href="README.md#citation">引用</a>
</p>

## 快速开始

使用 Python 3.11 及以上版本，先安装适合当前硬件的 PyTorch：

```bash
git clone https://github.com/manifold112/AnchorFlow_ICASSP2027.git
cd AnchorFlow_ICASSP2027
python -m pip install -e .
```

按照[数据指南](docs/data.md)准备 1 秒 log-mel 特征及包含设备标签的 CSV 清单，然后启动训练：

```bash
anchorflow train --config configs/default.yaml \
  --protocol configs/protocols/dcase2024.yaml \
  --train-csv data/train.csv --val-csv data/val_source.csv \
  --feature-root data --output runs/tau22_af_seed0 --device cuda --seed 0
```

[配置文件](configs/default.yaml)包含论文使用的 AnchorFlow 参数。基线与 AnchorFlow
应采用相同的骨干、优化器和前处理。训练与模型选择使用 A-C 和 S1-S3，S4-S6 保留作未见设备评估。

<details>
<summary><b>评估与模型导出</b></summary>

```bash
anchorflow evaluate --checkpoint runs/tau22_af_seed0/best.pt \
  --manifest data/eval.csv --feature-root data --device cuda

anchorflow export --checkpoint runs/tau22_af_seed0/best.pt \
  --output checkpoints/scene_model.pt
```

TAU20 先平均同一条录音中十个 1 秒片段的预测概率，再生成录音级预测。
导出仅保留编码器与场景预测头。

</details>

<details>
<summary><b>接入已有分类器</b></summary>

将模型划分为输出 `[batch, feature_dim]` 特征的编码器，以及接收这些特征的场景预测头。
辅助分支共享预测头的可训练参数：

```python
from anchorflow import AnchorFlow

af = AnchorFlow(num_scenes=10, num_devices=6, feature_dim=128).to(device)
h = encoder(x)
base_loss = criterion(head(h), y)
flow = af(h, y, d, head=head, epoch=epoch)
loss = base_loss + af.config.weight * flow.energy
```

反向传播前暂存停止梯度的特征统计，在优化器成功更新后提交锚点。
完整示例见 [`examples/integrate.py`](examples/integrate.py)，
预测头状态、梯度累积和分布式锚点见[接入指南](docs/integration.md)。

</details>

## 实验结果

下表为论文 Table 1 中的**未见设备准确率（%）**，报告五次运行的均值与标准差。
TAU20 对应 DCASE 2020 Task 1A，TAU22 对应 DCASE 2024 Task 1 split100。

| 骨干 | TAU20 | TAU20 + AF | TAU22 | TAU22 + AF |
| :--- | ---: | ---: | ---: | ---: |
| CP-Mobile | 66.57 ± 0.57 | **69.70 ± 0.61** | 53.10 ± 0.58 | **56.37 ± 0.55** |
| BC-ResNet-ASC-8 | 64.34 ± 0.52 | **67.78 ± 0.55** | 52.00 ± 0.50 | **55.57 ± 0.57** |
| TF-SepNet-64 | 64.14 ± 0.52 | **67.85 ± 0.57** | 53.70 ± 0.49 | **57.62 ± 0.58** |
| PaSST-S | 74.34 ± 0.65 | **77.02 ± 0.60** | 55.90 ± 0.48 | **58.62 ± 0.59** |

八组设置下，未见设备准确率平均提升 **3.31 个百分点**。在 CochlScene + MicIRP 中，
未见麦克风响应准确率从 65.22 ± 0.60 提升至 **68.44 ± 0.56**。

[![跨场景设备偏移方向的余弦相似度](docs/figures/device-shifts.png)](docs/figures/device-shifts.png)

<p align="center"><em>跨场景设备偏移方向的相似度：左为基线，右为 AnchorFlow。原图来自论文 Figure 3。</em></p>

[完整结果与消融](docs/results.md) · [CSV 表格](benchmarks) · [方法与公式](docs/method.md)

引用信息见[英文首页](README.md#citation)。代码采用 [MIT License](LICENSE)，
论文与第三方数据保留各自权利。
