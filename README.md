<div align="center">

# AnchorFlow

**Beyond Device Invariance: Scene-Conditioned Device Flow Modeling**  
**for Cross-Device Acoustic Scene Classification**

[Peihong Zhang](https://orcid.org/0009-0002-9561-1412),
[Yuxuan Liu](https://orcid.org/0009-0007-1043-416X),
[Yiqiang Cai](https://orcid.org/0009-0004-7062-7850),
[Yizhou Tan](https://orcid.org/0000-0002-4103-8115),
[Shengchen Li](https://orcid.org/0000-0002-2488-298X)

Xi'an Jiaotong-Liverpool University

[Paper](docs/paper.pdf) · [Method](docs/method.md) · [Results](docs/results.md) · [中文](README.zh-CN.md)

</div>

AnchorFlow is a training mechanism for cross-device acoustic scene classification.
It models device variation within each scene and regularizes the scene predictor
along the resulting device transitions. The inference model remains unchanged.

## Method

[![Overview of AnchorFlow](docs/figures/method.png)](docs/figures/method.png)

*Figure 2 from the paper. Scene-device anchors, same-scene device pairing,
residual-preserving trajectories, and finite-difference path regularization.*

For each scene-device pair, AnchorFlow maintains a diagonal-Gaussian anchor using
EMA feature moments. A source sample is transported toward another device anchor
in the same scene, keeping its standardized residual fixed. The shared scene head
is regularized through finite differences of the loss along this path. Device
labels and the auxiliary branch are used only during training.

See the [method notes](docs/method.md) for the equations and
[integration guide](docs/integration.md) for the encoder/head interface.

## Results

**Unseen-device accuracy (%)** on DCASE 2020 Task 1A (TAU20) and DCASE 2024
Task 1 split100 (TAU22). Values are from Table 1 of the paper, reported as
mean ± standard deviation over five runs.

| Backbone | TAU20 · Base | TAU20 · +AF | TAU22 · Base | TAU22 · +AF |
| :--- | ---: | ---: | ---: | ---: |
| CP-Mobile | 66.57 ± 0.57 | **69.70 ± 0.61** | 53.10 ± 0.58 | **56.37 ± 0.55** |
| BC-ResNet-ASC-8 | 64.34 ± 0.52 | **67.78 ± 0.55** | 52.00 ± 0.50 | **55.57 ± 0.57** |
| TF-SepNet-64 | 64.14 ± 0.52 | **67.85 ± 0.57** | 53.70 ± 0.49 | **57.62 ± 0.58** |
| PaSST-S | 74.34 ± 0.65 | **77.02 ± 0.60** | 55.90 ± 0.48 | **58.62 ± 0.59** |

The average unseen-device gain is 3.31 percentage points across these eight
settings. On CochlScene with MicIRP responses, unseen-response accuracy improves
from 65.22 ± 0.60 to 68.44 ± 0.56.

[![Cross-scene cosine similarity of device-shift directions](docs/figures/device-shifts.png)](docs/figures/device-shifts.png)

*Figure 3 from the paper. Cross-scene cosine similarity of device-shift directions
for the baseline and AnchorFlow. The mean off-diagonal similarity changes from
0.273 to 0.362.*

[Full results, comparisons and ablations](docs/results.md) · [CSV tables](benchmarks)

## Getting started

Install Python 3.11+ and a PyTorch build for your hardware, then install the package:

```bash
python -m pip install -e .
```

Prepare one-second log-mel features and device-aware CSV manifests following the
[data guide](docs/data.md). Training and validation use observed devices
**A, B, C, S1, S2, S3**; **S4, S5, S6** are reserved for unseen-device evaluation.

```bash
anchorflow train --config configs/default.yaml \
  --protocol configs/protocols/dcase2024.yaml \
  --train-csv data/train.csv --val-csv data/val_source.csv \
  --feature-root data --output runs/tau22_af_seed0 --device cuda --seed 0
```

The [default configuration](configs/default.yaml) uses the paper's AnchorFlow
parameters: λ = 0.3, δ = 0.20, K = 1, EMA decay = 0.99, minimum anchor count = 32,
and one warm-up epoch. Keep the optimizer, preprocessing and backbone recipe
matched between baseline and AnchorFlow runs.

<details>
<summary>Evaluation and model export</summary>

```bash
anchorflow evaluate --checkpoint runs/tau22_af_seed0/best.pt \
  --manifest data/eval.csv --feature-root data --device cuda

anchorflow export --checkpoint runs/tau22_af_seed0/best.pt \
  --output checkpoints/scene_model.pt
```

Evaluation reports overall, seen, unseen and per-device accuracy. TAU20 averages
ten segment probabilities before predicting the recording label. Export retains
the encoder and scene head.

</details>

For an existing ASC model, see [`examples/integrate.py`](examples/integrate.py).
The [integration guide](docs/integration.md) covers anchor update order, head
evaluation, gradient accumulation and distributed statistics.

## Citation

```bibtex
@unpublished{zhang_anchorflow,
  title  = {Beyond Device Invariance: Scene-Conditioned Device Flow Modeling
            for Cross-Device Acoustic Scene Classification},
  author = {Zhang, Peihong and Liu, Yuxuan and Cai, Yiqiang and
            Tan, Yizhou and Li, Shengchen},
  note   = {Research manuscript}
}
```

Code is released under the [MIT License](LICENSE). The paper and third-party
datasets retain their respective rights.
