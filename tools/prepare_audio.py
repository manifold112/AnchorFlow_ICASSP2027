"""Cache 1-second mono log-mel segments from a labeled audio CSV.

Input columns: audio_path, scene, device, recording_id, user_id (optional).
Front-end choices are explicit arguments so baseline/AF pairs can share caches.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    args = parser.parse_args()
    with open(args.protocol, encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    with open(args.manifest, encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    root = Path(args.audio_root).resolve()
    output = Path(args.output_root)
    (output / "features").mkdir(parents=True, exist_ok=True)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate, n_fft=args.n_fft,
        hop_length=args.hop_length, n_mels=args.n_mels)
    result, identities = [], set()
    for row in records:
        identity = (row["recording_id"], row["device"])
        if identity in identities:
            raise ValueError(f"duplicate recording/device: {identity}")
        identities.add(identity)
        audio_path = (root / row["audio_path"]).resolve()
        if not audio_path.is_relative_to(root):
            raise ValueError("audio_path must stay within audio-root")
        waveform, source_rate = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(waveform.mean(0), source_rate,
                                                   args.sample_rate)
        segments = protocol["segments_per_recording"]
        if waveform.numel() != segments * args.sample_rate:
            raise ValueError(f"{audio_path}: expected exactly {segments} second(s)")
        key = hashlib.sha256("\0".join(identity).encode()).hexdigest()[:20]
        for segment in range(segments):
            clip = waveform[segment * args.sample_rate:(segment + 1) * args.sample_rate]
            with torch.no_grad():
                feature = transform(clip).clamp_min(1e-10).log().numpy()
            relative = f"features/{key}_{segment:02d}.npy"
            np.save(output / relative, feature)
            result.append({"feature_path": relative, "scene": row["scene"],
                           "device": row["device"], "recording_id": row["recording_id"],
                           "user_id": row.get("user_id", "")})
    path = Path(args.output_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["feature_path", "scene", "device",
                                                   "recording_id", "user_id"])
        writer.writeheader()
        writer.writerows(result)
    print(f"Cached {len(result)} segments from {len(records)} recordings")


if __name__ == "__main__":
    main()
