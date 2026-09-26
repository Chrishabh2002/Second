# Project summary

**Semantic change detection on SECOND: a controlled study of architectures and class imbalance**

This page summarises the project in a few minutes of reading. Details: [README](../README.md) ·
[Methodology](METHODOLOGY.md) · [Results](RESULTS.md) · [Diagnostics](DIAGNOSTICS.md) ·
[References](REFERENCES.md) · [Roadmap](ROADMAP.md).

## In one paragraph

Semantic change detection (SCD) finds where two aerial images of the same place differ, and what
each changed pixel was before and after. On the SECOND dataset only 20% of pixels change, and the
largest land-cover class is 59× bigger than the smallest. I compared three SCD architectures under
one fixed protocol, benchmarked nine class-imbalance techniques on the best one, analysed the
errors, and checked stability with 3-fold cross-validation. The best combination, Bi-SRNet-lite with
weighted cross-entropy + Dice, raised the primary metric SeK from 7.97 (Early Fusion baseline) to
15.38 (+93%). The main remaining error is missed changes, which motivates the next step: a
transition-aware loss.

## What was done

| Step | What | Why |
|---|---|---|
| Data | Streamed SECOND (2,968 pairs) into a 256 px copy; fixed 70/10/20 split | Run everything on one 16 GB laptop |
| Imbalance analysis | Class shares, 31 from → to transition types, change ratio | Understand the problem before choosing methods |
| Phase A | Early Fusion [HRSCD-str.2], SSCD [SSCD-l], Bi-SRNet-lite [Bi-SRNet]; same backbone, split, seed, schedule | Isolate the effect of the architecture |
| Phase B | Weighted CE, median-frequency, class-balanced, focal, Dice, OHEM, rare-class sampling, and combinations | Find which imbalance technique helps, under the same model |
| Error analysis | Correct / wrong class / missed / false alarm for every run | Find what kind of error remains |
| Cross-validation | 3 folds × 12 configurations, train vs validation curves | Check stability, over- and underfitting |

Every component is referenced in [REFERENCES.md](REFERENCES.md).

## Main results (test split, 595 pairs)

| Configuration | SeK | Fscd | mIoU |
|---|---:|---:|---:|
| Early Fusion + CE | 7.97 | 46.12 | 63.71 |
| SSCD + CE | 13.59 | 52.48 | 68.02 |
| Bi-SRNet-lite + CE | 14.24 | 53.36 | 68.65 |
| **Bi-SRNet-lite + WCE + Dice** | **15.38** | **53.88** | **68.97** |

1. **Architecture matters most:** +79% SeK from Early Fusion to Bi-SRNet-lite.
2. **Imbalance handling adds +8% on top.** 6 of 9 techniques beat plain CE. Focal loss and
   sampling alone did not help, and OHEM failed at its default setting.
3. **Rare classes gain the most:** water IoU 0.0 → 23.0, playground IoU 9.7 → 32.9.
4. **Missed changes are the largest error** (38% of changed pixels with CE, 28% with the best
   technique). Imbalance techniques raise change recall (62% → 72%) but also false alarms
   (5.4% → 9.0%).
5. **Cross-validation** confirms the architecture ranking and, so far, that WCE + Dice beats CE in
   all 3 folds. The runs are still in progress; see [DIAGNOSTICS.md](DIAGNOSTICS.md) for the
   current state.

## Honest limitations

- **Lower than published scores** (Bi-SRNet reports 23.22 SeK). The setup here uses 256 px (¼ of
  the pixels), ResNet-18 and 20 epochs, so it is not like-for-like.
- **Nothing new yet:** the models and losses all come from earlier papers. The contribution so far
  is the controlled comparison and the error analysis.
- **One seed for the main runs;** only 3 CV folds. Differences under ~0.5 SeK are not reliable.

## Next steps

Short version of the [roadmap](ROADMAP.md):

1. Finish cross-validation and add seeds.
2. Tune the change threshold and train longer.
3. **Transition-aware loss:** weight pixels by the rarity of their from → to change, and test it
   against CE, WCE + Dice and the SeK loss of Mamba-FCS.
4. Confirm at 512 px with a larger backbone on a GPU.
5. Add a second dataset (Landsat-SCD).
6. Write the paper.

## Likely questions

**Why SeK and not accuracy?** 80% of pixels are unchanged, so a model that predicts "no change"
everywhere already gets high OA. SeK removes that effect and rewards correct change *and* correct
class.

**Why is Bi-SRNet-lite "lite"?** It keeps the cross-temporal attention and consistency loss of
Bi-SRNet but not its Siamese reasoning blocks, so all three models share the same ResNet-18 and
the comparison stays fair.

**Is the gain from WCE + Dice real?** On the main split it is +1.14 SeK. In cross-validation it has
won every fold so far. More seeds will make this firmer.

**What would make it publishable?** A method that fixes the main error (missed changes on rare
transitions), shown to work at the published setting and on a second dataset.
