# SpatialAD score progression

This records measured image-level VAD results. The primary metrics are test
classification AUROC and FPR@95TPR. Reconstruction and observer experiments
are diagnostic only.

## Main progression

| Stage | Main change | Best validation AUROC | Test AUROC | Seen defects | Unseen defects | Test FPR@95TPR |
|---|---|---:|---:|---:|---:|---:|
| 1 | Raw pixel SpatialAD with reconstruction branch | 66.36 | 62.79 | 62.89 | 62.26 | 91.10 |
| 2 | Pixel normal-reference z-score channel | 73.67 | 65.65 | 64.57 | 71.15 | 89.90 |
| 3 | z-score clipping and reconstruction ablations | N/A | 68.70 | 68.16 | 71.42 | 85.80 |
| 4 | Reconstruction-free supervised BCE SpatialAD | 73.01 | 69.21 | 68.72 | 71.73 | 87.30 |
| 5 | Reverse-InfoNCE embedding objective | 71.35 | 71.07 | 71.05 | 71.20 | 84.60 |
| 6 | Frozen ImageNet ResNet18 spatial features | N/A | **78.15** | **78.96** | **74.08** | **75.50** |
| 7 | Frozen DINOv2 CLS + classifier | 78.95 | 76.64 | 77.79 | 70.80 | 77.90 |
| 8 | DINOv2 + spatial Transformer, 224px, no z-map | 89.46 | 86.07 | 87.19 | 80.42 | 66.00 |
| 9 | DINOv2 + spatial Transformer + pixel z-map, 224px | 90.33 | 88.07 | 89.26 | 82.02 | 60.50 |
| 10 | DINOv2 + spatial Transformer + learned patch pooling | 90.16 | 87.65 | 89.02 | 80.73 | 59.50 |
| 11 | DINOv2 + spatial Transformer, 448px, z-map | **95.68** | 92.82 | 93.66 | **88.53** | **41.40** |
| 12 | DINOv2 + spatial Transformer, 448px, no z-map | 95.59 | **92.98** | **94.58** | 84.84 | 43.00 |
| 13 | 448px full model with BPR instead of BCE | 94.81 | 92.36 | 93.84 | 84.85 | 47.00 |

## Key conclusions

### Pretrained spatial features

Frozen DINOv2 with a global classifier reached 76.64 AUROC. Adding the spatial
Transformer raised performance to 86.07 at 224px, showing that spatial
relational modelling contributes substantially beyond the global DINO
representation.

### Normal-reference z-map

At 224px, adding the pixel-space z-map improved test AUROC from 86.07 to
88.07. At 448px, the overall AUROC difference became negligible:

```text
448px with z-map:    92.82
448px without z-map: 92.98
```

However, the z-map improved unseen-defect AUROC from 84.84 to 88.53 and
reduced unseen-defect FPR from 61.50% to 44.10% in the paired high-resolution
comparison. This suggests that the z-map may provide useful low-level
normality information for OOD defects even when it does not improve aggregate
AUROC.

### Patch pooling

Replacing mean pooling with learned patch-attention pooling did not improve
AUROC:

```text
mean pooling:       88.07 overall, 82.02 unseen
attention pooling:  87.65 overall, 80.73 unseen
```

Mean pooling remains the simpler and stronger default. Attention pooling gave
a small FPR improvement but did not improve ranking, especially on unseen
defects.

### Observation attention

Batch-wide observation attention was unstable because each image only saw the
other images in its current mini-batch rather than a persistent dataset-level
normal context. Its validation/test gap became very large. The gated residual
observer did not open meaningfully, so it did not provide evidence of a useful
observation module.

The next principled observation module is a fixed normal DINO feature memory
bank, evaluated independently from arbitrary test-batch composition.

### Loss ablation

At 448px, BPR and BCE were similar overall:

```text
BCE: 92.82 overall, 88.53 unseen, FPR 41.40
BPR: 92.36 overall, 84.85 unseen, FPR 47.00
```

BCE remains preferable because it preserved substantially better unseen-defect
performance and FPR. The main bottleneck is therefore not currently the choice
between BCE and BPR.

### Resolution is the dominant current bottleneck

Increasing DINO input resolution from 224px to 448px changed the patch grid
from 16x16 to 32x32:

```text
224px: 256 spatial tokens
448px: 1024 spatial tokens
```

This raised overall AUROC from 88.07 to 92.82 and unseen-defect AUROC from
82.02 to 88.53. The current strongest robust configuration is therefore the
448px DINOv2 SpatialAD model with the z-map and mean pooling.

## Current frontier

```text
Best aggregate AUROC:       92.98  (448px, no z-map)
Best unseen AUROC:           88.53  (448px, z-map)
Best FPR@95TPR:              41.40  (448px, z-map)
```

The next experiments should test multi-scale features, lightly fine-tuned
DINOv2 features, and a fixed normal-feature memory bank before introducing
additional reconstruction or contrastive objectives.

## Net change from the first SpatialAD model

Using the best aggregate AUROC:

```text
test AUROC:       62.79 -> 92.98   (+30.19 points)
seen AUROC:       62.89 -> 94.58   (+31.69 points)
unseen AUROC:     62.26 -> 88.53   (+26.27 points)
FPR@95TPR:        91.10 -> 41.40   (-49.70 points)
```
