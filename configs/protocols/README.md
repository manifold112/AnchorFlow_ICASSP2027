# Evaluation protocols

`dcase2020.yaml` groups ten 1-second segments into a single 10-second recording.
`dcase2024.yaml` evaluates individual 1-second segments in the split100 protocol.
Both distinguish observed devices A-C / S1-S3 from held-out devices S4-S6.

The CochlScene + MicIRP experiment uses a user-disjoint split and six observed /
three held-out microphone responses. Define a protocol YAML with the same schema,
using the actual CochlScene scene vocabulary and MicIRP response IDs from your
manifests, and set `user_disjoint: true`. Response identities are not enumerated
in the manuscript, so this configuration is supplied by the experiment owner.
