# Phase 3: Kaggle study, 512×512

Semantic change detection on SECOND at the **native 512×512 resolution**, trained on one Kaggle T4 GPU.

- **Final notebook (v4):** [`notebooks/SECOND_SCD_512.ipynb`](notebooks/SECOND_SCD_512.ipynb). It runs end to end on Kaggle or Colab: download, training of 8 runs, TTA, Excel output and charts. If a session ends, it resumes automatically.
- **Earlier versions:** [`notebooks/drafts/`](notebooks/drafts/). v2 added lazy image loading. v3 switched to a single GPU and added crash diagnostics. v1 is not kept.
- **Plain-Python export:** [`src/second_scd_512.py`](src/second_scd_512.py).
- **Notebook generator:** [`tools/build_512_notebook.py`](tools/build_512_notebook.py). Run it from the repository root.
- **Results:** [`results/`](results/): `SECOND_SCD_512_Results.xlsx` and CSVs (Results, TP/FP/FN/TN, per-class, fit check, 256 vs 512, vs papers).
- **Logs:** [`logs/`](logs/): the Kaggle log of the v2 kernel crash, and the v3 training session that ran out of RAM. The v4 notebook fixes both problems.

## Results (test set)

| Model | Loss | SeK | Fscd | mIoU | OA |
|---|---|---|---|---|---|
| Early Fusion | CE | 13.61 | 53.15 | 67.95 | 85.72 |
| SSCD | CE | 18.19 | 58.05 | 70.28 | 87.24 |
| Bi-SRNet | CE | 18.46 | 58.28 | 70.74 | 87.13 |
| Bi-SRNet | WCE | 18.79 | 57.89 | 71.33 | 87.17 |
| Bi-SRNet | Focal | 18.45 | 58.19 | 70.73 | 87.26 |
| Bi-SRNet | CE + Dice | 19.15 | 58.76 | 71.37 | 86.79 |
| Bi-SRNet | WCE + Dice | 18.71 | 58.16 | 71.05 | 86.79 |
| Bi-SRNet-DF (proposed) | CE + Dice | 20.38 | 59.62 | 72.25 | 86.81 |
| **Bi-SRNet-DF (proposed) + TTA** | **CE + Dice** | **21.00** | **60.22** | **72.62** | 87.08 |

See the [main README](../README.md) for the method, the 256→512 comparison, the fit check and the comparison with published papers.
