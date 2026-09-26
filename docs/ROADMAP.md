# Roadmap: from this study to a publishable paper

## Where the project stands

| Done | Evidence |
|---|---|
| Controlled comparison of 3 architectures (Early Fusion → SSCD → Bi-SRNet-lite) | [README](../README.md#phase-a-architecture-plain-ce), SeK 7.97 → 13.59 → 14.24 |
| 9 class-imbalance techniques on the best architecture | [README](../README.md#phase-b-class-imbalance-techniques-on-bi-srnet-lite), best WCE + Dice, SeK 15.38 |
| Error breakdown of every run | [README](../README.md#model-diagnostics-errors-cross-validation-and-fit): missed changes are the largest error in 11 of 12 runs |
| 3-fold cross-validation + train/val curves | [DIAGNOSTICS](DIAGNOSTICS.md), in progress (`run_cv.sh`) |
| Documentation, figures, references | README, [METHODOLOGY](METHODOLOGY.md), [RESULTS](RESULTS.md), [REFERENCES](REFERENCES.md) |

**What is missing for a paper.** None of the models or losses is new. The absolute scores are
below published ones because of the smaller setup (256 px, ResNet-18, 20 epochs). Most results rest
on one seed. The steps below fix these gaps, most important first.

## Step 1: finish and strengthen the statistics (running now, then ~1 day)

- **Why.** Several techniques differ by less than 0.5 SeK. Without repeated runs, a reviewer can
  say those gaps are noise.
- **How.** Let `run_cv.sh` finish all 36 folds. For the top 3 configurations and the CE baseline,
  add 2 more seeds of the main split (`--seed 1`, `--seed 2`). Report mean ± std, and a paired test
  (paired t-test across folds × seeds) for "technique vs CE".
- **Done when.** Every claim in the README carries a mean ± std and a fold/seed win count.

## Step 2: cheap improvements suggested by the error analysis (1–2 days)

| Idea | Why | Cost |
|---|---|---|
| **Tune the change threshold** on the validation split (now fixed at 0.5) | Missed changes are the biggest error. A lower threshold trades a few false alarms for fewer misses, and SeK rewards found changes | No retraining; re-evaluate saved checkpoints |
| **Train longer** (40 epochs) for WCE + Dice | In all 3 CV folds its best epoch was the last one (20), so it was still improving | 1 run ≈ 45 min |
| **Fix or drop OHEM** | With keep = 25% it marked 60% of unchanged pixels as changed. Try keep = 50–70%, or only apply OHEM to the semantic branch | 2–3 runs |

## Step 3: the new method, a transition-aware loss (main contribution, 1–2 weeks)

**Idea.** Current re-weighting looks at one class at a time. The real imbalance is in the 30 change
types ("from → to" pairs). *non-veg ground → building* is 24.9% of changed pixels, while
*playground → water* is 0.001% ([transition matrix](figures/transition_matrix.png)). The loss
should weight each pixel by the rarity of its **transition**, not of its class.

**Design.**
1. For each training pixel that changed from class *a* to class *b*, weight = (frequency of a → b)^(−½),
   normalised to mean 1. Clip very large weights so single pixels cannot dominate.
2. Apply the weight to the semantic loss of **both** dates. Also apply it to the positive pixels of
   the change loss, which directly attacks the missed-change error.
3. Optional: a small 30-way "transition head" trained with the same weights.

**Must beat.** Plain CE, WCE + Dice (our best so far), and the SeK-inspired loss of Mamba-FCS
([arXiv:2508.08232](https://arxiv.org/abs/2508.08232)), which we re-implement. Compare with
FD-ProtoSCD ([Remote Sensing 2026](https://www.mdpi.com/2072-4292/18/17/2957)), which also
targets imbalanced transitions, but with prototype learning instead of loss weighting.

**Measure.** SeK, Fscd, and a new **mean per-transition recall** (average recall over the 30 change
types), which shows whether rare transitions improve. Use 3 seeds.

**Before starting.** Do a proper literature search (IEEE Xplore, Scopus, Google Scholar,
2022–2026, keywords: *semantic change detection* + *class imbalance*, *long-tailed*, *transition*,
*from-to*) to confirm this exact loss has not been published.

## Step 4: match the published protocol (needs a GPU, ~1 week)

- **Why.** Reviewers will compare with Bi-SRNet (23.22 SeK) and SCanNet (23.94 SeK). Our 256 px /
  ResNet-18 / 20-epoch setup cannot show that the method helps state-of-the-art models.
- **How.** 512 px, ResNet-34, 50+ epochs, on Google Colab / Kaggle / a university GPU. Run CE,
  WCE + Dice, SeK loss and the transition-aware loss on at least Bi-SRNet-lite. Also plug the new
  loss into one public SOTA codebase (e.g. SCanNet), since a loss is easy to transfer.
- **Done when.** Our Bi-SRNet-lite + CE is within ~1 SeK of the published Bi-SRNet, and the new
  loss adds a consistent gain on top.

## Step 5: a second dataset (~1 week)

Show that the gain is not specific to SECOND. **Landsat-SCD** is the most common second benchmark
(Mamba-FCS reports on both). Its class and transition statistics are different, which makes it a
good test for a transition-aware loss.

## Step 6: write the paper (2–3 weeks)

| Section | Content |
|---|---|
| Introduction | Transition imbalance in SCD; missed changes dominate the error |
| Related work | SCD architectures [REFERENCES](REFERENCES.md); imbalance losses; SeK loss; FD-ProtoSCD |
| Method | Transition-aware loss |
| Experiments | Controlled benchmark (this repo) + SOTA setting + second dataset; CV and seeds |
| Analysis | Error breakdown, per-transition recall, precision–recall trade-off |

**Possible venues.** IEEE Geoscience and Remote Sensing Letters (short letter), IGARSS (conference),
MDPI *Remote Sensing*, IEEE JSTARS. Pick with your supervisor.

## Step 7: release

- Fix the git history (see [ENGINEERING_NOTES](ENGINEERING_NOTES.md#5-a-368-mb-file-blocked-git-push)) and
  tag a release that matches the paper.
- Keep `requirements.txt`, the fixed split and `make_docs.py`, so every table can be regenerated.
- If AI tools were used in the work, check your institution's and the venue's disclosure rules.

## Timeline (approximate)

| Week | Work |
|---|---|
| 1 | Finish CV + extra seeds; threshold tuning; longer training; literature search |
| 2–3 | Transition-aware loss: implement, test on the 256 px setup, 3 seeds |
| 4 | GPU runs at 512 px; plug the loss into a SOTA codebase |
| 5 | Landsat-SCD |
| 6–8 | Writing, supervisor feedback, submission |
