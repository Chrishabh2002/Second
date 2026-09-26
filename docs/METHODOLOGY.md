# Methodology

This page describes how models are built, trained and scored. Every detail here matches the code in
[scd.py](../scd.py) and [train.py](../train.py). The paper behind each component, with links, is listed
in [REFERENCES.md](REFERENCES.md).

## 1. Task

Each input is a co-registered pair of aerial images: T1 (before) and T2 (after). For every pixel the
model predicts:

- a **change mask**: changed or unchanged (binary);
- a **land-cover class at T1** and a **land-cover class at T2**: one of *non-vegetated ground,
  tree, low vegetation, water, building, playground*.

The final SCD map for each date is `class × change`: unchanged pixels get label 0 (*no-change*),
changed pixels get their predicted class 1–6. This matches the SECOND label format, where semantic
labels exist only on changed pixels.

## 2. Data

| Item | Value |
|---|---|
| Source | SECOND public training release, 2,968 pairs, 512×512, 0.5–3 m resolution |
| Stored copy | 256×256: images resized with Lanczos, labels with nearest-neighbour ([prepare_data.py](../prepare_data.py)) |
| Label conversion | Official RGB colour map → class index 0–6; an unknown colour would be counted and mapped to 0 (none found) |
| Split | Random 70/10/20 with seed 0 → **2,077 train / 296 validation / 595 test**, fixed for every run |
| Normalisation | ImageNet mean/std |
| Augmentation (train only) | Horizontal flip, vertical flip, rot90 ×k, **temporal swap** (T1↔T2 with their labels), each at random, applied to the whole batch |

## 3. Architectures

All models use the same **ImageNet-pretrained ResNet-18** encoder, which gives features at strides
4, 8, 16 and 32 (64/128/256/512 channels). They also share the same **FPN-style decoder**: 1×1
lateral convs to 64 channels, top-down upsampling with addition, and a 3×3 conv-BN-ReLU after each
merge. Heads are 1×1 convs, bilinearly upsampled to the input size.

### Early Fusion (HRSCD strategy 2 style, [Daudt et al.](https://arxiv.org/abs/1810.08452)), 11.36M parameters
T1 and T2 are stacked as a 6-channel input. The first ResNet conv is widened by copying its
pretrained RGB filters and halving them. One decoder feeds three heads: change (1 channel),
semantic T1 (6), semantic T2 (6).

### Siamese SSCD (SSCD-l style, [Ding et al.](https://arxiv.org/abs/2108.06103)), 11.58M parameters
- A **shared** encoder processes T1 and T2 separately.
- A shared **semantic decoder** is applied to each date's features, followed by a shared semantic
  head.
- A separate **change decoder** takes the per-scale concatenation `[f1, f2]`
  (128/256/512/1024 channels) and feeds the change head.

### Bi-SRNet-lite ([Ding et al.](https://arxiv.org/abs/2108.06103)), 11.91M parameters
SSCD plus two ideas from Bi-SRNet:

1. **Cross-temporal attention** on the deepest features (stride 32). Each date's features attend to
   the other date's: `a' = a + γ · Attn(Q(a), K(b), V(b))`, with query/key reduced 8×.
   `γ` starts at 0, so training begins from plain SSCD and learns how much cross-date context
   to use.
2. **Semantic consistency loss.** With `cos` = cosine similarity of the two softmax semantic maps,
   the loss is `mean[(1 − cos)·(1 − change) + cos·change]`. The two dates should agree where
   nothing changed and disagree where something changed.

The original Bi-SRNet also has Siamese reasoning blocks and a ResNet-34 backbone. They are left out
here so the backbone stays the same across all three models.

## 4. Losses

Total loss = `L_change + 0.5 · (L_sem(T1) + L_sem(T2)) [+ L_consistency for Bi-SRNet-lite]`.

The semantic loss is computed **only on changed pixels**. No-change pixels are set to ignore
index −1.

Class statistics come from the **training split only**: pixel counts of classes 1–6 over both
dates, and the change ratio `p`.

| Technique | Change branch | Semantic branch |
|---|---|---|
| CE (baseline) | BCE | CE |
| Weighted CE | BCE, `pos_weight = √((1−p)/p)` | CE, weights ∝ 1/√freq |
| Median-freq ([Eigen & Fergus](https://arxiv.org/abs/1411.4734)) | BCE + `pos_weight` | CE, weights = median(freq)/freq |
| Class-balanced ([Cui et al.](https://arxiv.org/abs/1901.05555)) | BCE + `pos_weight` | CE, weights = (1−β)/(1−β^n), β = 1 − 1/mean(n) |
| Focal ([Lin et al.](https://arxiv.org/abs/1708.02002)) | Focal BCE, γ = 2, α = 0.75 | Focal CE, γ = 2 |
| CE + Dice ([V-Net](https://arxiv.org/abs/1606.04797)) | BCE + soft Dice | CE + multi-class soft Dice |
| OHEM ([Shrivastava et al.](https://arxiv.org/abs/1604.03540)) | Mean of the hardest 25% of pixel BCE values | Mean of the hardest 25% of pixel CE values |
| WCE + Dice | BCE + `pos_weight` + Dice | Weighted CE + Dice |
| … + rare sampling (after [LVIS](https://arxiv.org/abs/1908.03195) repeat-factor sampling) | same as the loss | same as the loss; training images are drawn with probability ∝ max over classes present of 1/√freq |

All class-weight vectors are normalised to mean 1. With `p ≈ 0.20`, the change `pos_weight` is ≈ 2.0.

## 5. Training

| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | 5e-4 for decoders/heads, 1e-4 (0.2×) for the pretrained encoder |
| Schedule | OneCycle, 10% warm-up, one scheduler step per iteration |
| Batch size / epochs | 8 / 20 (259 iterations per epoch) |
| Device | Apple M4 GPU through PyTorch MPS |
| Model selection | After each epoch, evaluate on validation; keep the checkpoint with the best **validation SeK** |
| Reporting | The chosen checkpoint is evaluated **once** on the test split |
| Seed | 0 for every run (same split, same initialisation) |

## 6. Metrics

This project uses the **official SECOND metrics** ([Yang et al.](https://arxiv.org/abs/2010.05687)), the same definitions as the evaluation code of
SECOND, Bi-SRNet and SCanNet. Predicted and true SCD maps of both dates are accumulated into one
7×7 confusion matrix (rows = prediction, columns = truth, class 0 = no-change).

| Metric | Definition |
|---|---|
| **OA** | Overall pixel accuracy over the 7 classes |
| **mIoU** | Mean of the *binary* IoUs: no-change IoU and change IoU |
| **SeK** (primary) | `κ̂ · e^(IoU_change − 1)`. `κ̂` is Cohen's kappa of the confusion matrix with the true-negative cell (no-change → no-change) set to 0, so the large no-change class cannot inflate it |
| **Fscd** | Harmonic mean of precision and recall of *correct semantic labels on changed pixels* |
| Per-class IoU | IoU of each of the 7 classes in the SCD map |

The change threshold is 0.5 on the sigmoid output.

SeK is chosen as the primary metric because it punishes both missed changes and wrong classes, and
because OA stays high even for a model that predicts almost no change. In our runs, OA barely moves
(84–86%) while SeK varies from 8 to 15.
