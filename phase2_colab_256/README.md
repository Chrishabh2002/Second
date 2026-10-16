# Phase 2: Colab study, 256×256

Semantic change detection on SECOND at **256×256** (down-sampled from 512 because of GPU memory). This run was done on Google Colab.

- **Notebook:** [`notebook/SECOND_SCD_256.ipynb`](notebook/SECOND_SCD_256.ipynb). Open it in Colab with a GPU runtime and run all cells. It mounts Google Drive and downloads SECOND automatically.
- **Plain-Python export:** [`src/second_scd_256.py`](src/second_scd_256.py), for reading and diffing.
- **Results:** [`results/`](results/): `SECOND_SCD_256_Results.xlsx` and one CSV per sheet (Results, TP/FP/FN/TN, per-class, fit check).

## Setup

ResNet-18 (ImageNet), small FPN decoder (64 channels), AdamW with lr 3e-4, cosine schedule, 20 epochs, batch 8, mixed precision. Random 70 / 10 / 20 split with seed 0. Phase 3 uses the same split.

- **Experiment 1:** Early Fusion, SSCD and Bi-SRNet, each with CE loss.
- **Experiment 2:** CE, WCE, Focal, CE + Dice and WCE + Dice on the best model (Bi-SRNet).

## Results (test set)

| Model | Loss | SeK | Fscd | mIoU | OA |
|---|---|---|---|---|---|
| Early Fusion | CE | 9.41 | 47.90 | 64.79 | 84.57 |
| SSCD | CE | 12.41 | 51.94 | 66.47 | 85.68 |
| Bi-SRNet | CE | 12.87 | 52.63 | 67.07 | 85.44 |
| Bi-SRNet | WCE | 13.25 | 52.19 | 67.47 | 85.57 |
| Bi-SRNet | Focal | 12.78 | 51.96 | 66.92 | 85.62 |
| Bi-SRNet | CE + Dice | 13.43 | 52.99 | 67.72 | 85.28 |
| **Bi-SRNet** | **WCE + Dice** | **13.92** | **53.38** | 67.60 | 84.68 |

The train-vs-validation SeK gap was 3–10 points, which is mild overfitting. Phase 3 fixes this. See the [main README](../README.md) for the 256→512 comparison.
