# Results reported in the manuscript

[← Project](../README.md) · [Paper](paper.pdf) · [CSV tables](../benchmarks)

All benchmark values on this page are transcribed from the accompanying manuscript.
They are reported over five matched runs as mean ± standard deviation. Accuracy
values are percentages; changes are percentage points.

## Cross-device generalization

Table 1 evaluates four backbones on DCASE 2020 Task 1A and DCASE 2024 split100.
The complete per-backbone seen, unseen and overall statistics are in
[`paper_results.csv`](../benchmarks/paper_results.csv) and the project README.

| Protocol | Seen · Base / +AF | Unseen · Base / +AF | Overall · Base / +AF |
| :--- | ---: | ---: | ---: |
| DCASE 2020 average | 74.14 / **75.51** | 67.35 / **70.59** | 71.88 / **73.87** |
| DCASE 2024 average | 63.68 / **64.95** | 53.68 / **57.05** | 60.34 / **62.31** |

The paper reports consistent gains across all eight backbone-protocol pairs.
The average unseen-device improvement is 3.31 points; gains are systematically
larger on unseen devices than on observed devices.

## Compatibility with other methods

Table 2(a), using TF-SepNet-64. This compact view shows means; standard deviations
are retained in [`paper_compatibility.csv`](../benchmarks/paper_compatibility.csv).

| Method | TAU20 overall | TAU20 unseen | TAU22 overall | TAU22 unseen |
| :--- | ---: | ---: | ---: | ---: |
| Baseline | 70.32 | 64.14 | 61.10 | 53.70 |
| UADA | 70.69 | 66.87 | 61.57 | 56.51 |
| FBS + SDA | 71.31 | 67.54 | 62.13 | 57.34 |
| RFN | 71.77 | 68.99 | 62.77 | 58.90 |
| Freq-MS | 71.97 | 68.28 | 63.30 | 59.49 |
| DIR | 71.76 | 67.61 | 62.84 | 58.43 |
| ADIR | 71.91 | 68.04 | 63.06 | 59.05 |
| **AnchorFlow** | **72.63** | **67.85** | **63.42** | **57.62** |
| DIR + Freq-MS | 72.19 | 69.31 | 63.52 | 60.56 |
| AF + RFN | 73.00 | **70.24** | 63.78 | 60.62 |
| AF + Freq-MS | **73.16** | 70.12 | **64.02** | **61.16** |
| AF + DIR | 73.08 | 69.82 | 63.88 | 60.54 |

Among standalone methods in this comparison, AnchorFlow has the highest overall
accuracy on both protocols. RFN and Freq-MS have higher standalone unseen accuracy.
Adding AnchorFlow to normalization or augmentation methods improves the reported
overall and unseen results.

## Component ablations

Table 2(b), DCASE 2024 with TF-SepNet-64.

| Variant | Seen | Unseen | Overall |
| :--- | ---: | ---: | ---: |
| Baseline | 64.80 ± 0.38 | 53.70 ± 0.49 | 61.10 ± 0.24 |
| Scene-agnostic | 65.64 ± 0.39 | 55.78 ± 0.53 | 62.35 ± 0.28 |
| Mean-only | 65.94 ± 0.44 | 56.64 ± 0.59 | 62.84 ± 0.30 |
| Endpoint | 65.86 ± 0.46 | 56.44 ± 0.57 | 62.72 ± 0.32 |
| Non-detached | 63.55 ± 0.32 | 51.95 ± 0.46 | 59.68 ± 0.27 |
| Matched isotropic | 65.56 ± 0.43 | 55.66 ± 0.54 | 62.26 ± 0.31 |
| **AnchorFlow** | **66.32 ± 0.35** | **57.62 ± 0.58** | **63.42 ± 0.28** |

The mean-only variant is the strongest ablation. The non-detached variant falls
below the baseline, supporting the method's separation of path geometry from
auxiliary optimization.

## Geometry and external response shifts

Table 3(a) measures non-additive scene-device interaction on TAU22. The interaction
ratio is 0.01956 ± 0.00072 for the baseline and 0.02094 ± 0.00079 with AnchorFlow;
both have permutation-test p-values below 0.005. Figure 3 reports mean off-diagonal
cosine similarity of scene-dependent device-shift directions increasing from 0.273
to 0.362. These are diagnostics of scene-device geometry, not accuracy metrics.

Table 3(b), user-disjoint CochlScene with six observed and three held-out MicIRP
responses:

| Method | Seen responses | Unseen responses | Overall |
| :--- | ---: | ---: | ---: |
| Baseline | 76.13 ± 0.35 | 65.22 ± 0.60 | 72.49 ± 0.40 |
| **AnchorFlow** | **77.96 ± 0.39** | **68.44 ± 0.56** | **74.79 ± 0.38** |

## Reading the figures

The hero and architecture illustrations are schematics. The gain chart is generated
from the manuscript table values. Run `python tools/build_assets.py` to regenerate
the repository SVG assets after editing the structured tables.
