<div align="center">

# AnchorFlow

**超越设备不变性：面向跨设备声学场景分类的场景条件设备流建模**

Peihong Zhang, Yuxuan Liu, Yiqiang Cai, Yizhou Tan, Shengchen Li

西交利物浦大学

[论文](docs/paper.pdf) · [方法](docs/method.md) · [实验结果](docs/results.md) · [English](README.md)

</div>

AnchorFlow 是面向跨设备声学场景分类的训练机制。它在每个场景内部建模设备变化，
并约束场景预测器沿设备转换路径的损失变化。推理仍使用原有模型。

## 方法

[![AnchorFlow 方法总览](docs/figures/method.png)](docs/figures/method.png)

*论文图 2：场景与设备分布锚点、同场景设备配对、保留残差的设备路径，以及有限差分路径正则。*

方法为每个“场景 × 设备”维护高斯分布锚点，利用 EMA 更新特征的一阶与二阶原始矩。
同一场景下，源设备与目标设备的均值、标准差共同插值，样本的标准化残差保持不变。
共享预测头通过路径上场景损失的有限差分进行正则化。设备标签和辅助分支仅在训练阶段使用。

数学定义见[方法文档](docs/method.md)，模型拆分与更新顺序见[接入指南](docs/integration.md)。

## 实验结果

下表为论文 Table 1 中的**未见设备准确率（%）**，报告五次运行的均值与标准差。
TAU20 对应 DCASE 2020 Task 1A，TAU22 对应 DCASE 2024 Task 1 split100。

| 骨干 | TAU20 · Base | TAU20 · +AF | TAU22 · Base | TAU22 · +AF |
| :--- | ---: | ---: | ---: | ---: |
| CP-Mobile | 66.57 ± 0.57 | **69.70 ± 0.61** | 53.10 ± 0.58 | **56.37 ± 0.55** |
| BC-ResNet-ASC-8 | 64.34 ± 0.52 | **67.78 ± 0.55** | 52.00 ± 0.50 | **55.57 ± 0.57** |
| TF-SepNet-64 | 64.14 ± 0.52 | **67.85 ± 0.57** | 53.70 ± 0.49 | **57.62 ± 0.58** |
| PaSST-S | 74.34 ± 0.65 | **77.02 ± 0.60** | 55.90 ± 0.48 | **58.62 ± 0.59** |

八组设置的未见设备准确率平均提升 3.31 个百分点。在 CochlScene 与 MicIRP 的外部验证中，
未见麦克风响应准确率由 65.22 ± 0.60 提升至 68.44 ± 0.56。

[![不同场景之间设备偏移方向的余弦相似度](docs/figures/device-shifts.png)](docs/figures/device-shifts.png)

*论文图 3：基线与 AnchorFlow 的跨场景设备偏移方向余弦相似度。
非对角元素均值由 0.273 变为 0.362。*

[完整结果、兼容性比较与消融](docs/results.md) · [CSV 表格](benchmarks)

## 使用

安装 Python 3.11 及以上版本和适合当前硬件的 PyTorch，然后安装项目：

```bash
python -m pip install -e .
```

按[数据文档](docs/data.md)准备 1 秒 log-mel 特征和 CSV 清单。
训练与模型选择使用设备 **A、B、C、S1、S2、S3**，设备 **S4、S5、S6** 用于未见设备评估。

```bash
anchorflow train --config configs/default.yaml \
  --protocol configs/protocols/dcase2024.yaml \
  --train-csv data/train.csv --val-csv data/val_source.csv \
  --feature-root data --output runs/tau22_af_seed0 --device cuda --seed 0
```

[默认配置](configs/default.yaml)中的方法参数为 λ = 0.3、δ = 0.20、K = 1、EMA 衰减 0.99、
锚点最小样本量 32、预热 1 个 epoch。基线与 AnchorFlow 应采用相同的骨干、前处理和优化器配置。

<details>
<summary>评估与模型导出</summary>

```bash
anchorflow evaluate --checkpoint runs/tau22_af_seed0/best.pt \
  --manifest data/eval.csv --feature-root data --device cuda

anchorflow export --checkpoint runs/tau22_af_seed0/best.pt \
  --output checkpoints/scene_model.pt
```

DCASE 2020 先对同一条录音的十个片段平均预测概率，再生成录音级预测。
导出文件包含原有编码器与场景预测头。

</details>

已有模型接入示例见 [`examples/integrate.py`](examples/integrate.py)。
引用信息见[英文首页](README.md#citation)。代码采用 [MIT License](LICENSE)，
论文与第三方数据保留各自权利。
