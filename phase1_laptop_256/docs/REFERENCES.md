# References: every model, technique and metric used

Each row lists the paper a component comes from, a link to that paper, where the component is
implemented in this repository, and how this project's version differs from the original. The
numbered list at the end gives the full citations.

## Dataset and benchmark

| Component | Source | Link | In this repo | Our use / difference |
|---|---|---|---|---|
| **SECOND dataset** (2,968 public pairs, 6 land-cover classes) | Yang et al. [1] | [arXiv:2010.05687](https://arxiv.org/abs/2010.05687) · [dataset page](https://captain-whu.github.io/SCD/) | [prepare_data.py](../prepare_data.py) | Public training release only, downsampled 512 → 256 px, split 70/10/20 (seed 0) |
| **SeK, Fscd, mIoU, OA** (official SECOND metrics) | Yang et al. [1], same code as Bi-SRNet [3] and SCanNet [4] | [arXiv:2010.05687](https://arxiv.org/abs/2010.05687) | [`scd_metrics`](../scd.py#L329) | Re-implemented to match the official evaluation code |
| **Cohen's kappa** (inside SeK) | Cohen [17] | [DOI:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104) | [`cal_kappa`](../scd.py#L321) | Computed with the no-change → no-change cell set to 0, as SeK requires |

## Architectures

| Component | Source | Link | In this repo | Our use / difference |
|---|---|---|---|---|
| **Early Fusion** (6-channel input, one network, three heads) | HRSCD strategy 2, Daudt et al. [2] | [arXiv:1810.08452](https://arxiv.org/abs/1810.08452) | [`EarlyFusion`](../scd.py#L117) | Uses our shared ResNet-18 + FPN decoder instead of the paper's network |
| **SSCD** (Siamese encoder, semantic decoders + change decoder) | SSCD-l, Ding et al. [3] | [arXiv:2108.06103](https://arxiv.org/abs/2108.06103) | [`SSCD`](../scd.py#L156) | ResNet-18 instead of ResNet-34; FPN decoders; 256 px |
| **Bi-SRNet-lite** (SSCD + cross-temporal attention + consistency loss) | Bi-SRNet, Ding et al. [3] | [arXiv:2108.06103](https://arxiv.org/abs/2108.06103) | [`CrossTemporalAttn`](../scd.py#L134), [`semantic_consistency_loss`](../scd.py#L308) | Keeps the cross-temporal reasoning (Cot-SR) and the consistency loss; leaves out the Siamese reasoning blocks and uses a gated attention |
| **ResNet-18 encoder** (ImageNet-pretrained) | He et al. [5] | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | [`Encoder`](../scd.py#L74) | torchvision weights; the first conv is widened to 6 channels for Early Fusion |
| **FPN-style decoder** | Lin et al. [6] | [arXiv:1612.03144](https://arxiv.org/abs/1612.03144) | [`Decoder`](../scd.py#L101) | Light version: 64 channels, fused at stride 4 |
| **Attention (query/key/value)** | Vaswani et al. [7]; non-local blocks, Wang et al. [8] | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) · [arXiv:1711.07971](https://arxiv.org/abs/1711.07971) | [`CrossTemporalAttn.attend`](../scd.py#L144) | Cross-attention between the two dates on stride-32 features |
| **Zero-initialised residual gate γ** | SAGAN, Zhang et al. [9] | [arXiv:1805.08318](https://arxiv.org/abs/1805.08318) | [`self.g`](../scd.py#L142) | Training starts from plain SSCD and learns how much attention to add |

### Published models used only for comparison (not re-implemented)

| Model | Source | Link |
|---|---|---|
| HRSCD strategies 2 and 4 | Daudt et al. [2] | [arXiv:1810.08452](https://arxiv.org/abs/1810.08452) |
| SSCD-l, Bi-SRNet | Ding et al. [3] | [arXiv:2108.06103](https://arxiv.org/abs/2108.06103) |
| SCanNet | Ding et al. [4] | [arXiv:2212.05245](https://arxiv.org/abs/2212.05245) |
| ChangeMamba | Chen et al. [10] | [arXiv:2404.03425](https://arxiv.org/abs/2404.03425) |
| Table of published SECOND numbers | Mamba-FCS [11] | [arXiv:2508.08232](https://arxiv.org/abs/2508.08232) |

## Class-imbalance techniques

| Technique | Source | Link | In this repo | Our use / difference |
|---|---|---|---|---|
| **Cross-entropy / BCE** (baseline) | Standard | — | [`SCDLoss`](../scd.py#L258) | Semantic CE only on changed pixels, as in Bi-SRNet [3] |
| **Weighted CE, inverse-sqrt frequency** | Common practice; square-root frequency re-balancing is also used by Mahajan et al. [12] | [arXiv:1805.00932](https://arxiv.org/abs/1805.00932) | [`sem_weights("inv_sqrt")`](../scd.py#L198) | Weights ∝ 1/√freq, normalised to mean 1; BCE `pos_weight = √(neg/pos)` on the change branch |
| **Median-frequency balancing** | Eigen & Fergus [13] | [arXiv:1411.4734](https://arxiv.org/abs/1411.4734) | [`sem_weights("median")`](../scd.py#L200) | Weight = median(freq) / freq |
| **Class-balanced loss (effective number)** | Cui et al. [14] | [arXiv:1901.05555](https://arxiv.org/abs/1901.05555) | [`sem_weights("effective")`](../scd.py#L202) | β = 1 − 1/mean(pixel count) |
| **Focal loss** | Lin et al. [15] | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | [`focal_ce`](../scd.py#L211), [`focal_bce`](../scd.py#L220) | γ = 2; α = 0.75 on the change branch |
| **Dice loss** | V-Net, Milletari et al. [16] | [arXiv:1606.04797](https://arxiv.org/abs/1606.04797) | [`dice_multiclass`](../scd.py#L230), [`dice_binary`](../scd.py#L239) | Soft Dice added to CE/BCE on both branches |
| **OHEM** (online hard example mining) | Shrivastava et al. [18] | [arXiv:1604.03540](https://arxiv.org/abs/1604.03540) | [`ohem_ce`](../scd.py#L244), [`ohem_bce`](../scd.py#L253) | Pixel-level: mean of the hardest 25% of pixel losses (the paper mines RoIs) |
| **Rare-class oversampling** | Inspired by repeat-factor sampling, LVIS, Gupta et al. [19] | [arXiv:1908.03195](https://arxiv.org/abs/1908.03195) | [train.py](../train.py#L58) | Image probability ∝ max over classes present of 1/√freq |
| **Semantic consistency loss** | Bi-SRNet, Ding et al. [3] | [arXiv:2108.06103](https://arxiv.org/abs/2108.06103) | [`semantic_consistency_loss`](../scd.py#L308) | Cosine similarity of the two softmax maps; agree if unchanged, differ if changed |
| **SeK-inspired loss** (planned comparison, see [ROADMAP](ROADMAP.md)) | Mamba-FCS [11] | [arXiv:2508.08232](https://arxiv.org/abs/2508.08232) | not yet | — |

## Training and evaluation

| Component | Source | Link | In this repo |
|---|---|---|---|
| **AdamW** optimiser | Loshchilov & Hutter [20] | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | [train.py](../train.py#L73) |
| **One-cycle learning-rate schedule** | Smith & Topin [21] | [arXiv:1708.07120](https://arxiv.org/abs/1708.07120) | [train.py](../train.py#L78) |
| **Augmentation**: flips, rot90, temporal swap | Standard for change detection | — | [`to_batch`](../scd.py#L53) |
| **K-fold cross-validation** | Standard; see Kohavi [22] | [IJCAI 1995](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf) | [train.py](../train.py) `--folds`, [run_cv.sh](../run_cv.sh) |
| **PyTorch / torchvision** | Paszke et al. [23] | [arXiv:1912.01703](https://arxiv.org/abs/1912.01703) | whole project |

## Bibliography

1. K. Yang, G.-S. Xia, Z. Liu, B. Du, W. Yang, M. Pelillo, L. Zhang. *Asymmetric Siamese Networks for Semantic Change Detection in Aerial Images.* IEEE TGRS, 2022. [arXiv:2010.05687](https://arxiv.org/abs/2010.05687)
2. R. Caye Daudt, B. Le Saux, A. Boulch, Y. Gousseau. *Multitask Learning for Large-scale Semantic Change Detection.* Computer Vision and Image Understanding, 2019. [arXiv:1810.08452](https://arxiv.org/abs/1810.08452)
3. L. Ding, H. Guo, S. Liu, L. Mou, J. Zhang, L. Bruzzone. *Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images.* IEEE TGRS, 2022. [arXiv:2108.06103](https://arxiv.org/abs/2108.06103)
4. L. Ding, J. Zhang, K. Zhang, H. Guo, B. Liu, L. Bruzzone. *Joint Spatio-Temporal Modeling for the Semantic Change Detection in Remote Sensing Images.* IEEE TGRS, 2024. [arXiv:2212.05245](https://arxiv.org/abs/2212.05245) · [DOI:10.1109/TGRS.2024.3362795](https://doi.org/10.1109/TGRS.2024.3362795)
5. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
6. T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, S. Belongie. *Feature Pyramid Networks for Object Detection.* CVPR, 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
7. A. Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
8. X. Wang, R. Girshick, A. Gupta, K. He. *Non-local Neural Networks.* CVPR, 2018. [arXiv:1711.07971](https://arxiv.org/abs/1711.07971)
9. H. Zhang, I. Goodfellow, D. Metaxas, A. Odena. *Self-Attention Generative Adversarial Networks.* ICML, 2019. [arXiv:1805.08318](https://arxiv.org/abs/1805.08318)
10. H. Chen, J. Song, C. Han, J. Xia, N. Yokoya. *ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model.* IEEE TGRS, 2024. [arXiv:2404.03425](https://arxiv.org/abs/2404.03425)
11. *Mamba-FCS: Joint Spatio-Frequency Feature Fusion, Change-Guided Attention, and SeK Loss for Enhanced Semantic Change Detection in Remote Sensing.* 2025. [arXiv:2508.08232](https://arxiv.org/abs/2508.08232)
12. D. Mahajan et al. *Exploring the Limits of Weakly Supervised Pretraining.* ECCV, 2018. [arXiv:1805.00932](https://arxiv.org/abs/1805.00932)
13. D. Eigen, R. Fergus. *Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture.* ICCV, 2015. [arXiv:1411.4734](https://arxiv.org/abs/1411.4734)
14. Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie. *Class-Balanced Loss Based on Effective Number of Samples.* CVPR, 2019. [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)
15. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. *Focal Loss for Dense Object Detection.* ICCV, 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
16. F. Milletari, N. Navab, S.-A. Ahmadi. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV, 2016. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
17. J. Cohen. *A Coefficient of Agreement for Nominal Scales.* Educational and Psychological Measurement, 1960. [DOI:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104)
18. A. Shrivastava, A. Gupta, R. Girshick. *Training Region-based Object Detectors with Online Hard Example Mining.* CVPR, 2016. [arXiv:1604.03540](https://arxiv.org/abs/1604.03540)
19. A. Gupta, P. Dollár, R. Girshick. *LVIS: A Dataset for Large Vocabulary Instance Segmentation.* CVPR, 2019. [arXiv:1908.03195](https://arxiv.org/abs/1908.03195)
20. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR, 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
21. L. N. Smith, N. Topin. *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.* 2017. [arXiv:1708.07120](https://arxiv.org/abs/1708.07120)
22. R. Kohavi. *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI, 1995. [PDF](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf)
23. A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS, 2019. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)

> Bibliographic details for [1]–[4] and [10] were checked against their arXiv pages. The other
> entries are standard citations; confirm the exact venue and year in your reference manager
> before submitting a paper.
