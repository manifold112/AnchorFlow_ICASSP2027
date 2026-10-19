# Device-aware data preparation

[← Project](../README.md) · [Integration](integration.md)

## Manifest contract

Each row describes one 1-second feature segment. Required columns are
`feature_path`, `scene`, `device`, and `recording_id`. A user-disjoint protocol also
requires `user_id`.

| Field | Meaning |
| :--- | :--- |
| `feature_path` | Relative path under `--feature-root` to a `.npy` log-mel array |
| `scene` | Exact scene string from the protocol vocabulary |
| `device` | Observed or evaluation device ID from the protocol |
| `recording_id` | Unique recording identity, shared by all segments of that recording |
| `user_id` | Recording contributor ID for user-disjoint evaluation |

Features are finite float32 arrays with shape `[mel_bins, frames]` or
`[1, mel_bins, frames]`. All tensors in a batch need the same shape. Arrays are
loaded with NumPy object deserialization disabled.

Feature paths stay inside the selected feature root. Repeated paths, unknown
labels, inconsistent recording labels and incorrect segment counts are rejected.
The trainer also checks for recording overlap between training and validation.
For user-disjoint protocols, users may not overlap either.

```text
data/
├── train.csv
├── val_source.csv
├── eval.csv
└── features/
    ├── airport_A_0001.npy
    └── ...
```

The `examples/manifest.csv` file illustrates the CSV schema.

## Protocols

| Protocol | Observed devices | Held-out devices | Evaluation unit |
| :--- | :--- | :--- | :--- |
| DCASE 2020 / TAU20 | A, B, C, S1, S2, S3 | S4, S5, S6 | Mean probability over ten 1-second segments |
| DCASE 2024 / TAU22 split100 | A, B, C, S1, S2, S3 | S4, S5, S6 | One 1-second segment |
| CochlScene + MicIRP | Six microphone responses | Three microphone responses | User-disjoint recording partitions |

Use the official recording partitions for each dataset. Build a source-only
validation partition for model selection; evaluation on held-out devices is a
separate step. If the underlying capture has multiple device recordings, use the
same canonical `recording_id` across those devices to prevent cross-partition leakage.

For CochlScene, supply a YAML protocol containing the dataset scene vocabulary and
the selected response IDs. Set `user_disjoint: true`. Keep the original contributor
identity when creating response-augmented recordings. See
[`configs/protocols/README.md`](../configs/protocols/README.md).

## Cache log-mel features from audio

Install audio dependencies and prepare an audio CSV with columns `audio_path`,
`scene`, `device`, `recording_id`, and optionally `user_id`:

```bash
python -m pip install -e '.[audio]'
python tools/prepare_audio.py --manifest data/audio_train.csv \
  --audio-root data/audio --output-root data \
  --output-manifest data/train.csv \
  --protocol configs/protocols/dcase2024.yaml \
  --sample-rate 32000 --n-mels 128 --n-fft 1024 --hop-length 320
```

This front end mixes channels to mono, resamples, splits into one-second segments,
and caches log mel-power values. Its arguments are configurable front-end choices.
Use the same cache for matched baseline and AnchorFlow experiments. The audio
backend dependencies must match the chosen PyTorch / torchaudio installation.

## Label-preserving inputs for anchor updates

The bank expects one scene and one device per feature. Use an unmixed source view
to update anchors if the base recipe uses mixup or combines recordings from
different device conditions. The auxiliary loss uses hard scene labels to sample
same-scene target anchors.

## Metrics

The evaluator averages segment probabilities within `(recording_id, device)`
groups before classification. It reports recording-weighted accuracy overall,
within seen and unseen device groups, and per device. An absent group has value
`null`, so source-only validation does not produce a misleading unseen score.
Keep the official benchmark aggregation convention when comparing external recipes.

Dataset audio, microphone impulse responses and cached features are managed
separately from the source repository under their respective access terms.
