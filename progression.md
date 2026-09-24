# SpatialAD score progression

This records the measured image-level VAD results as the model evolved. The
main comparison is test classification AUROC and FPR@95TPR. Reconstruction
scores are diagnostic only.

## Main progression

| Stage | Main change                                                                             | Best validation AUROC | Test AUROC | Seen defects | Unseen defects | Test FPR@95TPR |
| ----- | --------------------------------------------------------------------------------------- | --------------------: | ---------: | -----------: | -------------: | -------------: |
| 1     | First patch SpatialAD: raw intensity, spatial patches, classifier + reconstruction loss |                 66.36 |      62.79 |        62.89 |          62.26 |          91.10 |
| 2     | Frozen normal-reference per-pixel mean/std z-score channel added                        |                 73.67 |      65.65 |        64.57 |          71.15 |          89.90 |
| 3     | Normal-reference z-score, z_clip=16, reconstruction loss disabled                       |                 67.19 |      67.96 |        67.46 |          70.49 |          87.30 |
| 4     | Normal-reference z-score, z_clip=8, reconstruction loss disabled                        |                   N/A |      68.70 |        68.16 |          71.42 |          85.80 |
| 5     | Reconstruction head and loss removed; BCE classifier baseline                           |                 73.01 |      69.21 |        68.72 |          71.73 |          87.30 |
| 6     | Shared encoder + two-direction reverse-InfoNCE embedding objective                      |                 71.35 |  **71.07** |    **71.05** |          71.20 |      **84.60** |

## Per-stage details

### 1. First patch SpatialAD

The original SpatialAD converted each image into 15x15 patches:

```text
255x255 image -> 17x17 = 289 spatial tokens
```

It used raw pixel intensities, spatial positional embeddings, a Transformer, an
image classifier, and a reconstruction branch.

```text
best validation AUROC: 66.36
test AUROC:            62.79
seen defects:          62.89
unseen defects:        62.26
FPR@95TPR:             91.10
reconstruction-only:   58.15
```

### 2. Normal-reference sensor statistics

Each pixel coordinate received a frozen reference distribution fitted from
2,000 normal images:

```text
z[j] = (x[j] - normal_mean[j]) / normal_std[j]
```

The model received both raw intensity and the reference z-score.

```text
best validation AUROC: 73.67 at epoch 20
test AUROC:            65.65
seen defects:          64.57
unseen defects:        71.15
FPR@95TPR:             89.90
reconstruction-only:   60.98
```

This established that per-coordinate statistical context was more useful than
raw intensity alone.

### 3. z-score clipping ablation

The z_clip=16 experiment allowed more extreme deviations through. With
reconstruction loss disabled:

```text
test AUROC:            67.96
seen defects:          67.46
unseen defects:        70.49
FPR@95TPR:             87.30
```

The later z_clip=8 run gave better unseen-defect robustness.

### 4. Reconstruction ablation with z_clip=8

The reconstruction objective was set to zero while retaining the diagnostic
decoder score.

```text
test AUROC:            68.70
seen defects:          68.16
unseen defects:        71.42
FPR@95TPR:             85.80
reconstruction-only:   66.67
```

Removing the reconstruction loss improved the supervised classifier, even
though the diagnostic reconstruction score remained correlated with anomalies.

### 5. Reconstruction-free BCE SpatialAD

The decoder and reconstruction loss were removed completely. SpatialAD became
a pure normal-reference spatial classifier.

```text
best validation AUROC: 73.01 at epoch 19
test AUROC:            69.21
seen defects:          68.72
unseen defects:        71.73
FPR@95TPR:             87.30
```

### 6. Shared spatial encoder with reverse-InfoNCE

The pooled spatial embedding was passed through a normalized projection head.
The objective used two directional terms:

```text
good anchor: cross-class similarity mass - good-good similarity mass
bad anchor:  cross-class similarity mass - bad-bad similarity mass
```

The classifier BCE term remained active as a stabilizer. This run used
batch-size=32, producing five large optimization batches per epoch and enough
good/bad examples for stable pair formation.

```text
best validation AUROC: 71.35 at epoch 29
test AUROC:            71.07
seen defects:          71.05
unseen defects:        71.20
FPR@95TPR:             84.60
```

## Net change from the first SpatialAD model

```text
test AUROC:       62.79 -> 71.07   (+8.28 points)
seen AUROC:       62.89 -> 71.05   (+8.16 points)
unseen AUROC:     62.26 -> 71.20   (+8.94 points)
FPR@95TPR:        91.10 -> 84.60   (-6.50 points)
```

The largest gains came from:

1. Modeling every pixel relative to its normal-reference distribution.
2. Training a shared embedding with the two-direction reverse-InfoNCE objective.

SOTA aucroc is 96.5
