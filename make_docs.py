"""Build README.md, docs/*.md and docs/figures/*.png from results/*.json.

Every number in the generated documents is read from the run records, so re-running
this script after new experiments finish keeps the documentation in sync:
    .venv/bin/python make_docs.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import scd

R = os.path.join(scd.ROOT, "results")
D = os.path.join(scd.ROOT, "docs")
FIG = os.path.join(D, "figures")
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------- data
runs = {os.path.basename(p)[:-5]: json.load(open(p))
        for p in sorted(glob.glob(os.path.join(R, "[AB]_*.json")))}
imb = json.load(open(os.path.join(R, "imbalance_stats.json")))
best_arch = open(os.path.join(R, "best_arch.txt")).read().split()[-1]

ARCH = {"early_fusion": "Early Fusion (HRSCD-str.2-style)",
        "sscd": "Siamese SSCD (SSCD-l-style)",
        "bisrnet": "Bi-SRNet-lite (SSCD + cross-temporal attention + consistency loss)"}
ARCH_SHORT = {"early_fusion": "Early Fusion", "sscd": "SSCD", "bisrnet": "Bi-SRNet-lite"}
TECH = {  # (loss, sampler) -> (short label, description)
    ("ce", "uniform"): ("CE (baseline)", "Plain BCE (change) + CE (semantic), no balancing"),
    ("wce", "uniform"): ("Weighted CE", "Inverse-sqrt-frequency class weights + BCE pos_weight"),
    ("median", "uniform"): ("Median-freq", "Median-frequency balancing (Eigen & Fergus) + BCE pos_weight"),
    ("cb", "uniform"): ("Class-balanced", "Effective-number class-balanced loss (Cui et al., 2019) + BCE pos_weight"),
    ("focal", "uniform"): ("Focal", "Focal loss, gamma=2 (alpha=0.75 on the change branch)"),
    ("dice", "uniform"): ("CE + Dice", "CE/BCE plus soft Dice on both branches"),
    ("ohem", "uniform"): ("OHEM", "Online hard-example mining: mean of the hardest 25% of pixels"),
    ("ce", "rare"): ("CE + rare sampling", "Plain CE with rare-class image oversampling"),
    ("combo", "uniform"): ("WCE + Dice", "Weighted CE + Dice (both branches)"),
    ("combo", "rare"): ("WCE + Dice + rare sampling", "Weighted CE + Dice with rare-class oversampling"),
}
TECH_REF = {
    ("wce", "uniform"): "common practice; see [REFERENCES](REFERENCES.md)",
    ("median", "uniform"): "[Eigen & Fergus](https://arxiv.org/abs/1411.4734)",
    ("cb", "uniform"): "[Cui et al.](https://arxiv.org/abs/1901.05555)",
    ("focal", "uniform"): "[Lin et al.](https://arxiv.org/abs/1708.02002)",
    ("dice", "uniform"): "[Milletari et al.](https://arxiv.org/abs/1606.04797)",
    ("ohem", "uniform"): "[Shrivastava et al.](https://arxiv.org/abs/1604.03540)",
    ("ce", "rare"): "[Gupta et al. (LVIS)](https://arxiv.org/abs/1908.03195)",
    ("combo", "uniform"): "[Milletari et al.](https://arxiv.org/abs/1606.04797) + weighted CE",
    ("combo", "rare"): "as above + [LVIS](https://arxiv.org/abs/1908.03195)",
}
PLANNED_B = ["wce", "median", "cb", "focal", "dice", "ohem", "ce_rare", "combo", "combo_rare"]
SEM = scd.CLASS_NAMES[1:]
PARAMS = {"early_fusion": 11.36, "sscd": 11.58, "bisrnet": 11.91}  # millions, counted from the models

# Published results on SECOND (512x512, official protocol). Not directly comparable: see README.
LIT = [  # method, OA, mIoU, SeK, Fscd
    ("HRSCD-str.2", 85.49, 64.43, 10.69, 49.22),
    ("HRSCD-str.4", 86.62, 71.15, 18.80, 58.21),
    ("SSCD-l", 87.19, 72.60, 21.86, 61.22),
    ("Bi-SRNet", 87.84, 73.41, 23.22, 62.61),
    ("SCanNet", 87.86, 73.42, 23.94, 63.66),
    ("ChangeMamba", 88.12, 73.68, 24.11, 64.03),
]
LIT_OF = {"early_fusion": "HRSCD-str.2", "sscd": "SSCD-l", "bisrnet": "Bi-SRNet"}


def label(r):
    a = r["args"]
    return TECH[(a["loss"], a["sampler"])][0]


def info(tag):
    r = runs[tag]
    t = r["test"]
    return dict(tag=tag, model=r["args"]["model"], label=label(r), OA=t["OA"], mIoU=t["mIoU"],
                SeK=t["SeK"], Fscd=t["Fscd"], IoU_c=t["IoU_c"], pc=t["per_class_IoU"],
                val=max(h["SeK"] for h in r["history"]), ep=r["best_epoch"],
                mins=r["train_minutes"], hist=r["history"], conf=np.array(r["confusion"]))


A = [info(t) for t in runs if t.startswith("A_")]
A.sort(key=lambda x: ["early_fusion", "sscd", "bisrnet"].index(x["model"]))
base = next(x for x in A if x["model"] == best_arch)
B = sorted([info(t) for t in runs if t.startswith("B_")], key=lambda x: -x["SeK"])
BB = [base] + B                               # phase-B table incl. its CE baseline
ALL = A + B
best = max(ALL, key=lambda x: x["SeK"])
ef = next(x for x in A if x["model"] == "early_fusion")
done_b = {t.split("_", 2)[2] for t in runs if t.startswith("B_")}
pending = [k for k in PLANNED_B if k not in done_b]
rel = lambda new, old: (new - old) / old * 100

# ----------------------------------------------------------------------------- style
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2", "#e34948"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "font.family": "DejaVu Sans"})
from matplotlib.colors import LinearSegmentedColormap
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQ)


def clean(ax, grid_axis="x"):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y" if grid_axis == "x" else "x", visible=False)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def hbar(ax, labels, vals, colors, fmt="{:.2f}", ref=None):
    y = np.arange(len(labels))
    ax.barh(y, vals, 0.62, color=colors, edgecolor=SURF, linewidth=1.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, fmt.format(v), va="center", fontsize=9, color=INK2)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1)
    clean(ax)


# ----------------------------------------------------------------------------- figures
# 1. class imbalance: share of changed pixels per class, T1 vs T2
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(SEM))
t1 = [imb["T1_class_pct_of_changed"][c] for c in SEM]
t2 = [imb["T2_class_pct_of_changed"][c] for c in SEM]
ax.bar(x - 0.2, t1, 0.38, color=BLUE, label="Before (T1)", edgecolor=SURF, linewidth=1.5)
ax.bar(x + 0.2, t2, 0.38, color=ORANGE, label="After (T2)", edgecolor=SURF, linewidth=1.5)
for i in range(len(SEM)):
    ax.text(x[i] - 0.2, t1[i] + 0.8, f"{t1[i]:.1f}", ha="center", fontsize=8, color=INK2)
    ax.text(x[i] + 0.2, t2[i] + 0.8, f"{t2[i]:.1f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, SEM)
ax.set_ylabel("% of changed pixels")
ax.set_title(f"Semantic classes are {imb['semantic_imbalance_ratio_max_over_min']:.0f}x imbalanced "
             f"(and only {imb['change_pixel_ratio']*100:.1f}% of pixels change)")
ax.legend(loc="upper center", ncol=2)
ax.set_ylim(0, max(t1 + t2) * 1.22)
clean(ax, "y")
ax.spines["left"].set_visible(False)
save(fig, "class_distribution.png")

# 1b. the SeK "journey" for the summary: baseline -> architectures -> best technique
_ef = next(x for x in A if x["model"] == "early_fusion")
_ss = next(x for x in A if x["model"] == "sscd")
steps = [("Early Fusion + CE\n(starting point)", _ef["SeK"], GRAY), ("SSCD + CE", _ss["SeK"], GRAY),
         (f"{ARCH_SHORT[best_arch]} + CE", base["SeK"], GRAY),
         (f"{ARCH_SHORT[best_arch]} +\n{best['label']} (best)", best["SeK"], BLUE)]
fig, ax = plt.subplots(figsize=(8.5, 3.9))
xs = np.arange(len(steps))
ax.bar(xs, [v for _, v, _ in steps], 0.55, color=[c for _, _, c in steps], edgecolor=SURF, linewidth=1.5)
for i, (_, v, _) in enumerate(steps):
    ax.text(i, v + 0.35, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold", color=INK)
    if i:
        ax.text(i, v / 2, f"{(v - steps[0][1]) / steps[0][1] * 100:+.0f}%\nvs start", ha="center",
                va="center", fontsize=9, color="white" if steps[i][2] == BLUE else INK)
ax.set_xticks(xs, [n for n, _, _ in steps])
ax.set_ylim(0, max(v for _, v, _ in steps) * 1.18)
ax.set_ylabel("Test SeK (%)")
ax.set_title("How the score improved, step by step (test SeK, higher is better)")
clean(ax, "y")
ax.spines["left"].set_visible(False)
save(fig, "sek_journey.png")

# 2. from -> to transition matrix
M = np.zeros((6, 6))
for k, v in imb["transitions_pct"].items():
    a, b = k.split(" -> ")
    M[SEM.index(a), SEM.index(b)] = v
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(np.log10(M + 1e-3), cmap=SEQ_CMAP, vmin=-3, vmax=np.log10(M.max()))
for i in range(6):
    for j in range(6):
        v = M[i, j]
        ax.text(j, i, "-" if v == 0 else (f"{v:.1f}" if v >= 0.1 else f"{v:.2f}"), ha="center",
                va="center", fontsize=8, color="white" if v > 3 else INK)
ax.set_xticks(range(6), SEM, rotation=30, ha="right")
ax.set_yticks(range(6), SEM)
ax.set_xlabel("After (T2)")
ax.set_ylabel("Before (T1)")
ax.set_title("From → to transitions (% of changed pixels; colour on log scale)", fontsize=11)
ax.grid(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "transition_matrix.png")

# 3. architecture comparison: small multiples, one panel per metric
mets = [("SeK", "SeK (primary)"), ("Fscd", "Fscd"), ("mIoU", "mIoU"), ("OA", "OA")]
fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
for ax, (k, t) in zip(axes, mets):
    vals = [x[k] for x in A]
    cols = [BLUE if x["model"] == best_arch else GRAY for x in A]
    hbar(ax, [ARCH_SHORT[x["model"]] for x in A], vals, cols)
    ax.set_title(t, fontsize=11)
    ax.set_xlim(0, max(vals) * 1.18)
fig.suptitle("Phase A: architecture comparison (plain CE, test split)", x=0.01, ha="left",
             fontweight="bold", y=1.04)
save(fig, "architecture_comparison.png")

# 4. imbalance techniques: SeK per technique, CE baseline as reference line
fig, ax = plt.subplots(figsize=(8, 0.42 * len(BB) + 1.3))
vals = [x["SeK"] for x in BB]
cols = [BLUE if x is best else (INK2 if x is base else GRAY) for x in BB]
hbar(ax, [x["label"] for x in BB], vals, cols, ref=base["SeK"])
ax.set_xlim(0, max(vals) * 1.15)
ax.set_xlabel("Test SeK (%)")
ax.set_title(f"Phase B: class-imbalance techniques on {ARCH_SHORT[best_arch]}")
save(fig, "imbalance_techniques.png")

# 5. relative change vs CE baseline (diverging bars) for SeK / Fscd / change IoU
BOK = [x for x in B if x["val"] >= 0.5 * base["val"]]  # runs that trained; failures in the tables
fig, axes = plt.subplots(1, 3, figsize=(12, 0.42 * len(BOK) + 1.4), sharey=True)
for ax, k in zip(axes, ["SeK", "Fscd", "IoU_c"]):
    d = [rel(x[k], base[k]) for x in BOK]
    y = np.arange(len(BOK))
    ax.barh(y, d, 0.62, color=[BLUE if v >= 0 else RED for v in d], edgecolor=SURF, linewidth=1.5)
    ax.axvline(0, color=INK2, lw=1)
    lim = max(abs(v) for v in d) * 1.45 + 0.5
    ax.set_xlim(-lim, lim)
    for i, v in enumerate(d):
        ax.text(v + (lim * 0.03 if v >= 0 else -lim * 0.03), i, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [x["label"] for x in BOK])
    ax.invert_yaxis()
    ax.set_title({"SeK": "SeK", "Fscd": "Fscd", "IoU_c": "Change IoU"}[k], fontsize=11)
    clean(ax)
fig.suptitle("Relative change vs. the CE baseline (blue = better, red = worse)"
             + ("" if len(BOK) == len(B) else "; runs that failed to train are in the tables only"),
             x=0.01, fontsize=11,
             ha="left", fontweight="bold", y=1.03)
save(fig, "relative_gain_vs_ce.png")

# 6. per-class IoU heatmap
rows = A + B
P = np.array([x["pc"] for x in rows])
fig, ax = plt.subplots(figsize=(9, 0.4 * len(rows) + 1.6))
ax.imshow(P, cmap=SEQ_CMAP, vmin=0, vmax=90, aspect="auto")
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        ax.text(j, i, f"{P[i, j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if P[i, j] > 45 else INK)
ax.set_xticks(range(7), scd.CLASS_NAMES, rotation=25, ha="right")
ax.set_yticks(range(len(rows)),
              [f"{ARCH_SHORT[x['model']]} | {x['label']}" for x in rows])
ax.axhline(len(A) - 0.5, color=SURF, lw=4)
ax.set_title("Per-class IoU on the test split (%)")
ax.grid(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "per_class_iou.png")

# 7. validation curves (architectures + best technique)
fig, ax = plt.subplots(figsize=(8, 3.8))
curves = [(ARCH_SHORT[x["model"]] + " + CE", x, c) for x, c in zip(A, [GRAY, ORANGE, BLUE])]
if best is not base and best in B:
    curves.append((f"{ARCH_SHORT[best_arch]} + {best['label']}", best, AQUA))
for name, x, c in curves:
    e = [h["epoch"] for h in x["hist"]]
    s = [h["SeK"] for h in x["hist"]]
    ax.plot(e, s, color=c, lw=2, label=name)
top = max(curves, key=lambda t: max(h["SeK"] for h in t[1]["hist"]))[1]["hist"]
bi = max(range(len(top)), key=lambda i: top[i]["SeK"])
ax.annotate(f"best {top[bi]['SeK']:.1f}", (top[bi]["epoch"], top[bi]["SeK"]), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(0, len(top) + 1, 2))
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation SeK (%)")
ax.set_title("Validation SeK during training")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", visible=False)
save(fig, "training_curves.png")

# 8. ours vs published (reference only)
fig, ax = plt.subplots(figsize=(8, 3.6))
names = [m[0] for m in LIT]
ax.barh(np.arange(len(LIT)), [m[3] for m in LIT], 0.62, color=GRAY, edgecolor=SURF, linewidth=1.5,
        label="Published (512 px, full protocol)")
ours = [(f"Ours: {ARCH_SHORT[x['model']]} (256 px)", x["SeK"]) for x in A]
if best in B:
    ours.append((f"Ours: best ({best['label']})", best["SeK"]))
yo = np.arange(len(LIT), len(LIT) + len(ours))
ax.barh(yo, [v for _, v in ours], 0.62, color=BLUE, edgecolor=SURF, linewidth=1.5,
        label="Ours (256 px, ResNet-18, 20 epochs, laptop)")
ax.set_yticks(list(range(len(LIT))) + list(yo), names + [n for n, _ in ours])
ax.invert_yaxis()
for i, v in enumerate([m[3] for m in LIT] + [v for _, v in ours]):
    ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_xlabel("SeK (%)")
ax.set_title("SeK: our runs vs. published results (reference, not like-for-like)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, fontsize=8.5)
clean(ax)
save(fig, "literature_comparison.png")

# 9. confusion matrix of the best run (row-normalised by ground truth)
C = best["conf"].T.astype(float)  # stored rows=pred, cols=GT -> transpose to rows=GT
Cn = C / C.sum(1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(6.6, 5.6))
ax.imshow(Cn, cmap=SEQ_CMAP, vmin=0, vmax=100)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{Cn[i, j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if Cn[i, j] > 50 else INK)
ax.set_xticks(range(7), scd.CLASS_NAMES, rotation=30, ha="right")
ax.set_yticks(range(7), scd.CLASS_NAMES)
ax.set_xlabel("Predicted")
ax.set_ylabel("Ground truth")
ax.set_title(f"Confusion matrix, best run ({best['label']}), % of each true class")
ax.grid(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "confusion_best.png")

# 10. qualitative predictions, if a checkpoint of a finished run is on disk
PALETTE = np.array([[255, 255, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [0, 0, 255],
                    [128, 0, 0], [255, 0, 0]], np.uint8)
ckpts = [x for x in sorted(ALL, key=lambda x: -x["SeK"])
         if os.path.exists(os.path.join(scd.ROOT, "checkpoints", f"{x['tag']}.pt"))]
qual = ckpts[0] if ckpts else None
if qual:
    import torch
    names, im1, im2, l1, l2 = scd.load_all()
    _, _, te = scd.split_indices(len(names), seed=0)
    # pick test pairs that show several classes and a fair amount of change
    score = [(len(np.unique(np.concatenate([l1[i].ravel(), l2[i].ravel()]))), (l1[i] > 0).mean(), i)
             for i in te[:200]]
    picks = [i for _, _, i in sorted(score, key=lambda s: (-s[0], -s[1]))
             if 0.15 < (l1[i] > 0).mean() < 0.6][:4]
    net = scd.build_model(qual["model"])
    net.load_state_dict(torch.load(os.path.join(scd.ROOT, "checkpoints", f"{qual['tag']}.pt"),
                                   map_location="cpu"))
    net.eval()
    x1, x2, y1, y2 = scd.to_batch(im1, im2, l1, l2, np.array(picks), "cpu")
    with torch.no_grad():
        c, s1, s2, _ = net(x1, x2)
    ch = (torch.sigmoid(c.squeeze(1)) > 0.5).long()
    p1, p2 = ((s1.argmax(1) + 1) * ch).numpy(), ((s2.argmax(1) + 1) * ch).numpy()
    cols = ["Image T1", "Image T2", "Label T1", "Label T2", "Prediction T1", "Prediction T2"]
    fig, axes = plt.subplots(len(picks), 6, figsize=(13, 2.3 * len(picks)))
    for r, i in enumerate(picks):
        for cidx, img in enumerate([im1[i], im2[i], PALETTE[l1[i]], PALETTE[l2[i]],
                                    PALETTE[p1[r]], PALETTE[p2[r]]]):
            ax = axes[r, cidx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_color(GRID)
            if r == 0:
                ax.set_title(cols[cidx], fontsize=10, loc="center")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[k] / 255, ec=MUTED, lw=0.5) for k in range(7)]
    fig.legend(handles, scd.CLASS_NAMES, loc="lower center", ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Test-set predictions: {ARCH_SHORT[qual['model']]} + {qual['label']} "
                 f"(test SeK {qual['SeK']:.2f})", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "qualitative.png")

fig_md = lambda f, alt: f"![{alt}](docs/figures/{f})"
# ----------------------------------------------------------------------------- diagnostics
# (a) error breakdown of every main run, from its test confusion matrix
def err_breakdown(conf):
    C = conf.T.astype(float)  # stored rows=pred, cols=truth -> rows=truth, cols=pred
    chg = C[1:].sum()
    ok = np.trace(C[1:, 1:])
    return {"correct": ok / chg * 100, "wrong_class": (C[1:, 1:].sum() - ok) / chg * 100,
            "missed": C[1:, 0].sum() / chg * 100, "false_alarm": C[0, 1:].sum() / C[0].sum() * 100,
            "precision": C[1:, 1:].sum() / C[:, 1:].sum() * 100, "recall": C[1:, 1:].sum() / chg * 100}


for x in ALL:
    x["err"] = err_breakdown(x["conf"])
name_of = lambda x: f"{ARCH_SHORT[x['model']]} + {x['label']}"
ER = A + B
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.4 * len(ER) + 1.6), sharey=True,
                             gridspec_kw={"width_ratios": [3, 1.2]})
y = np.arange(len(ER))
left = np.zeros(len(ER))
for k, c, lab in (("correct", BLUE, "Correct class"), ("wrong_class", AQUA, "Changed, wrong class"),
                  ("missed", ORANGE, "Missed (predicted no-change)")):
    v = np.array([x["err"][k] for x in ER])
    a1.barh(y, v, 0.62, left=left, color=c, edgecolor=SURF, linewidth=1.5, label=lab)
    for i in range(len(ER)):
        if v[i] > 7:
            a1.text(left[i] + v[i] / 2, i, f"{v[i]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if c == BLUE else INK)
    left += v
a1.set_yticks(y, [name_of(x) for x in ER])
a1.invert_yaxis()
a1.set_xlim(0, 100)
a1.set_title("What happens to truly changed pixels (test)", fontsize=11)
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8.5)
clean(a1)
fa = [x["err"]["false_alarm"] for x in ER]
a2.barh(y, fa, 0.62, color=INK2, edgecolor=SURF, linewidth=1.5)
for i, v in enumerate(fa):
    a2.text(v + max(fa) * 0.02, i, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
a2.set_xlim(0, max(fa) * 1.3)
a2.set_title("False alarms (% of unchanged)", fontsize=11)
clean(a2)
fig.suptitle("Error breakdown of every model and technique", x=0.01, ha="left", fontweight="bold",
             y=1.02)
save(fig, "error_breakdown.png")

# (b) cross-validation runs (results/cv/<tag>_f<k>.json)
CV = {}
for pth in sorted(glob.glob(os.path.join(R, "cv", "*_f[0-9].json"))):
    r = json.load(open(pth))
    CV.setdefault(r["tag"].rsplit("_f", 1)[0], []).append(r)
CV_ORDER = ["A_early_fusion_ce", "A_sscd_ce", "A_bisrnet_ce", "B_bisrnet_combo", "B_bisrnet_dice",
            "B_bisrnet_wce", "B_bisrnet_median", "B_bisrnet_cb", "B_bisrnet_focal",
            "B_bisrnet_ce_rare", "B_bisrnet_combo_rare", "B_bisrnet_ohem"]


def slope(v):
    return float(np.polyfit(np.arange(len(v)), v, 1)[0])


def cv_summary(tag, rs):
    rs = sorted(rs, key=lambda r: r["args"]["fold"])
    a = rs[0]["args"]
    H = lambda k: np.array([[h[k] for h in r["history"]] for r in rs])  # folds x epochs
    tr_s, va_s, tr_l, va_l = H("train_SeK"), H("SeK"), H("train_eval_loss"), H("val_loss")
    be = [r["best_epoch"] - 1 for r in rs]
    f = np.arange(len(rs))
    gap_best = tr_s[f, be] - va_s[f, be]
    gap_end = tr_s[:, -1] - va_s[:, -1]
    loss_rise = (va_l[:, -1] - va_l.min(1)) / np.abs(va_l.min(1)) * 100
    tail = [slope(v[-5:]) for v in va_s]
    g, lr, sl = float(gap_end.mean()), float(loss_rise.mean()), float(np.mean(tail))
    if lr > 10 and g > 3:
        diag = "Overfitting"
    elif sl > 0.15 and g < 2:
        diag = "Underfitting (still improving)"
    elif g > 3 or lr > 10:
        diag = "Mild overfitting"
    elif tr_s[:, -1].mean() < 5:
        diag = "Failed to learn"
    else:
        diag = "Good fit"
    if va_s.max() < 5:
        diag = "Failed to learn"
    return dict(tag=tag, model=a["model"], label=TECH[(a["loss"], a["sampler"])][0], k=a["folds"],
                n=len(rs), test=np.array([r["test"]["SeK"] for r in rs]),
                test_fscd=np.array([r["test"]["Fscd"] for r in rs]),
                val=va_s[f, be], train=tr_s[f, be], gap_best=gap_best, gap_end=gap_end,
                loss_rise=loss_rise, tail=np.array(tail), diag=diag, tr_s=tr_s, va_s=va_s,
                tr_l=tr_l, va_l=va_l, n_train=rs[0]["n_train"], n_val=rs[0]["n_val"])


cvs = [cv_summary(t, CV[t]) for t in CV_ORDER if t in CV and len(CV[t]) == CV[t][0]["args"]["folds"]]
cv_partial = sorted(set(CV) - {c["tag"] for c in cvs})
cvd = {c["tag"]: c for c in cvs}
cname = lambda c: f"{ARCH_SHORT[c['model']]} + {c['label']}"
pm = lambda v: f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def paired(c, ref):
    d = c["test"] - ref["test"]
    return d, int((d > 0).sum())


if cvs:
    order = sorted(cvs, key=lambda c: -c["test"].mean())
    # CV scores: mean ± std with each fold as a dot
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(order) + 1.4))
    for i, c in enumerate(order):
        m, s = c["test"].mean(), c["test"].std(ddof=1)
        ax.plot([m - s, m + s], [i, i], color=BLUE, lw=2, solid_capstyle="round")
        ax.scatter(c["test"], [i] * len(c["test"]), s=22, color=GRAY, zorder=3, edgecolor=SURF)
        ax.scatter([m], [i], s=60, color=BLUE, zorder=4, edgecolor=SURF, linewidth=2)
        ax.text(max(c["test"].max(), m + s) + 0.3, i, f"{m:.2f} ± {s:.2f}", va="center",
                fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(order)), [cname(c) for c in order])
    ax.invert_yaxis()
    ax.set_xlabel("Test SeK (%) — dot = one fold, large dot = mean, line = ± 1 std")
    ax.set_title(f"{order[0]['k']}-fold cross-validation: test SeK of each fold's model")
    ax.set_xlim(min(0, min(c["test"].min() for c in order) - 1),
                max(c["test"].max() for c in order) * 1.25)
    clean(ax)
    save(fig, "cv_scores.png")

    # learning curves: train vs val SeK and loss, mean over folds, one panel per config
    for key, fname, ylab in (("SeK", "learning_curves_sek.png", "SeK (%)"),
                             ("loss", "learning_curves_loss.png", "loss")):
        n = len(cvs)
        cols = 4
        rows_n = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(13, 2.6 * rows_n + 0.6), squeeze=False)
        for ax, c in zip(axes.ravel(), cvs):
            tr_v, va_v = (c["tr_s"], c["va_s"]) if key == "SeK" else (c["tr_l"], c["va_l"])
            e = np.arange(1, tr_v.shape[1] + 1)
            for v, col, lab in ((tr_v, ORANGE, "train"), (va_v, BLUE, "validation")):
                mu, sd = v.mean(0), v.std(0)
                ax.fill_between(e, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
                ax.plot(e, mu, color=col, lw=2, label=lab)
            ax.set_title(f"{cname(c)}\n{c['diag']}", fontsize=9, loc="left")
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", visible=False)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        axes[0, 0].legend(fontsize=8, loc="lower right" if key == "SeK" else "upper right")
        fig.suptitle(f"Train vs. validation {ylab} per epoch (mean of {cvs[0]['k']} folds, band = ± 1 std)",
                     x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        save(fig, fname)

    # generalisation gap at the selected epoch
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(cvs) + 1.3))
    g = [c["gap_best"].mean() for c in cvs]
    gs = [c["gap_best"].std(ddof=1) for c in cvs]
    ax.barh(range(len(cvs)), g, 0.62, xerr=gs, color=BLUE, edgecolor=SURF, linewidth=1.5,
            error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
    for i, v in enumerate(g):
        ax.text(max(v, 0) + gs[i] + 0.15, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(cvs)), [cname(c) for c in cvs])
    ax.invert_yaxis()
    ax.set_xlabel("Train SeK − validation SeK at the selected epoch (points)")
    ax.set_title("Generalisation gap (bigger = more overfitting)")
    clean(ax)
    save(fig, "generalization_gap.png")


def cv_table():
    out = ["| Model / technique | Val SeK | **Test SeK** | Test Fscd | Train SeK | Gap (train−val) | "
           "Val-loss rise | Diagnosis | Beats reference in |",
           "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for c in sorted(cvs, key=lambda c: -c["test"].mean()):
        ref = cvd.get("A_early_fusion_ce" if c["tag"].startswith("A_") else "A_bisrnet_ce")
        if ref is None or ref is c:
            win = "— (reference)"
        else:
            d, w = paired(c, ref)
            win = f"{w}/{len(d)} folds ({d.mean():+.2f})"
        out.append(f"| {cname(c)} | {pm(c['val'])} | **{pm(c['test'])}** | {pm(c['test_fscd'])} | "
                   f"{pm(c['train'])} | {pm(c['gap_best'])} | {c['loss_rise'].mean():.1f}% | "
                   f"{c['diag']} | {win} |")
    return "\n".join(out)


def err_table():
    out = ["| Model / technique | Correct | Wrong class | Missed change | False alarm | "
           "Change precision | Change recall |", "|---|---:|---:|---:|---:|---:|---:|"]
    for x in ER:
        e = x["err"]
        out.append(f"| {name_of(x)} | {e['correct']:.1f}% | {e['wrong_class']:.1f}% | "
                   f"{e['missed']:.1f}% | {e['false_alarm']:.1f}% | {e['precision']:.1f}% | "
                   f"{e['recall']:.1f}% |")
    return "\n".join(out)


n_miss_big = sum(x["err"]["missed"] > x["err"]["wrong_class"] for x in ER)
if cvs:
    cv_best = max(cvs, key=lambda c: c["test"].mean())
    ref_ce = cvd.get("A_bisrnet_ce")
    verdict = [] if len(cvs) < 2 else [f"- **Best by cross-validation:** {cname(cv_best)}, test SeK {pm(cv_best['test'])} "
               f"over {cv_best['k']} folds."]
    if ref_ce is not None and cv_best is not ref_ce and cv_best["model"] == "bisrnet":
        d, w = paired(cv_best, ref_ce)
        verdict.append(f"- Against plain CE on the same model it wins in **{w} of {len(d)} folds** "
                       f"(mean {d.mean():+.2f} SeK, std {d.std(ddof=1):.2f}). "
                       + ("The gain is consistent across folds." if w == len(d) else
                          "The gain is not consistent across folds, so treat it as uncertain."))
    counts_d = {}
    for c in cvs:
        counts_d[c["diag"]] = counts_d.get(c["diag"], 0) + 1
    verdict.append("- Fit diagnosis: " + ", ".join(f"{v}× {k.lower()}" for k, v in counts_d.items())
                   + ".")
    cv_md = f"""### Cross-validation ({cvs[0]['k']}-fold)

The {cvs[0]['n_train'] + cvs[0]['n_val']:,} train+validation pairs are split into {cvs[0]['k']}
folds ({cvs[0]['n_train']:,} train / {cvs[0]['n_val']:,} validation per fold). The 595-pair test split is
never used for training or model selection. Each fold's best-validation model is scored on it.
Because each fold trains on fewer pairs than the main runs (2,077), CV scores are a little lower;
they are used to measure **stability and ranking**, not to replace the main-table numbers.

""" + ("\n".join(verdict)) + f"""

{cv_table()}

{fig_md("cv_scores.png", "Cross-validation scores")}

### Overfitting and underfitting

Each epoch, the model is also scored on a fixed 296-pair subset of its own training data (no
augmentation), next to the validation fold. Rules used for the diagnosis (averaged over folds):

| Diagnosis | Rule |
|---|---|
| Overfitting | validation loss ends >10% above its minimum **and** train SeK − val SeK > 3 at the last epoch |
| Mild overfitting | only one of the two conditions above |
| Underfitting (still improving) | val SeK still rising (> 0.15 per epoch over the last 5 epochs) and gap < 2 |
| Failed to learn | best val SeK < 5 |
| Good fit | none of the above |

Losses of different techniques are defined differently, so compare train and validation loss
**within** a panel, not across panels.

{fig_md("learning_curves_sek.png", "Train vs validation SeK")}

{fig_md("learning_curves_loss.png", "Train vs validation loss")}

{fig_md("generalization_gap.png", "Generalisation gap")}
"""
    if cv_partial:
        cv_md += f"\n*Cross-validation still running for: {', '.join(cv_partial)}.*\n"
else:
    cv_md = "### Cross-validation\n\n*Cross-validation runs (`run_cv.sh`) are in progress.*\n"

diag_md = f"""## Model diagnostics: errors, cross-validation and fit

### Error breakdown (main runs, test split)

Every changed pixel ends up in one of three buckets: correct class, wrong class, or missed (the
model says *no-change*). Unchanged pixels that the model marks as changed are false alarms.

{err_table()}

{fig_md("error_breakdown.png", "Error breakdown")}

- Missed changes are a bigger error than wrong classes in **{n_miss_big} of {len(ER)}** runs.
- Imbalance handling finds more changes but also raises false alarms. Going from plain CE to {best['label']} on
  {ARCH_SHORT[best_arch]}, change recall rises from {base['err']['recall']:.1f}% to
  {best['err']['recall']:.1f}%, but change precision falls from {base['err']['precision']:.1f}% to
  {best['err']['precision']:.1f}%, and false alarms go from {base['err']['false_alarm']:.1f}% to
  {best['err']['false_alarm']:.1f}% of unchanged pixels. The net effect on SeK is still positive
  ({base['SeK']:.2f} → {best['SeK']:.2f}).
- OHEM shows the extreme end of this trade: {[x for x in ER if x['label'] == 'OHEM'][0]['err']['false_alarm'] if any(x['label'] == 'OHEM' for x in ER) else 0:.0f}% false alarms.

{cv_md}"""


# ----------------------------------------------------------------------------- tables
def tbl(rows, first="Run"):
    out = [f"| {first} | OA | mIoU | **SeK** | Fscd | Change IoU | val SeK | best ep | train min |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        name = x["label"] if first == "Technique" else ARCH[x["model"]]
        b = "**" if x is (best if first == "Technique" else base) else ""
        out.append(f"| {b}{name}{b} | {x['OA']:.2f} | {x['mIoU']:.2f} | {b}{x['SeK']:.2f}{b} | "
                   f"{x['Fscd']:.2f} | {x['IoU_c']:.2f} | {x['val']:.2f} | {x['ep']} | {x['mins']} |")
    return "\n".join(out)


def pc_tbl(rows):
    out = ["| Run | " + " | ".join(scd.CLASS_NAMES) + " |", "|---|" + "---:|" * 7]
    for x in rows:
        out.append(f"| {ARCH_SHORT[x['model']]} + {x['label']} | "
                   + " | ".join(f"{v:.1f}" for v in x["pc"]) + " |")
    return "\n".join(out)


def delta_tbl():
    out = ["| Technique | ΔSeK | ΔFscd | ΔmIoU | ΔOA | ΔChange IoU | Δwater IoU | Δplayground IoU |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in B:
        f = lambda k: f"{rel(x[k], base[k]):+.1f}%"
        fp = lambda j: f"{x['pc'][j] - base['pc'][j]:+.1f} pt"
        out.append(f"| {x['label']} | {f('SeK')} | {f('Fscd')} | {f('mIoU')} | {f('OA')} | "
                   f"{f('IoU_c')} | {fp(4)} | {fp(6)} |")
    return "\n".join(out)


def lit_tbl():
    out = ["| Method | Setting | OA | mIoU | SeK | Fscd |", "|---|---|---:|---:|---:|---:|"]
    for m in LIT:
        out.append(f"| {m[0]} | published, 512 px | {m[1]:.2f} | {m[2]:.2f} | {m[3]:.2f} | {m[4]:.2f} |")
    for x in A + ([best] if best in B else []):
        n = ARCH_SHORT[x["model"]] + ("" if x in A else f" + {x['label']}")
        out.append(f"| **Ours: {n}** | 256 px, R-18, 20 ep | {x['OA']:.2f} | {x['mIoU']:.2f} | "
                   f"**{x['SeK']:.2f}** | {x['Fscd']:.2f} |")
    return "\n".join(out)


litd = {m[0]: m for m in LIT}
reach = {x["model"]: x["SeK"] / litd[LIT_OF[x["model"]]][3] * 100 for x in A}
sscd = next(x for x in A if x["model"] == "sscd")
oa_drop = sum(x["OA"] < base["OA"] for x in B)
failed = [x for x in B if x["val"] < 0.5 * base["val"]]
failed_md = "".join(
    f"> **{x['label']} did not train properly** (best validation SeK {x['val']:.2f}, test OA "
    f"{x['OA']:.1f}%). Its default setting was not tuned for this task, so this run shows that "
    f"setting failing here, not that the technique cannot work.\n\n" for x in failed)
better_b = [x for x in B if x["SeK"] > base["SeK"]]
worse_b = [x for x in B if x["SeK"] <= base["SeK"]]
_Cb = best["conf"].T.astype(float)
miss = float(np.mean(_Cb[1:, 0] / _Cb[1:].sum(1)) * 100)
rare_gain = [(x, x["pc"][4] - ef["pc"][4], x["pc"][6] - ef["pc"][6]) for x in [best]]
status = ("All planned experiments are complete." if not pending else
          f"**Status:** {len(B)}/{len(PLANNED_B)} phase-B runs complete; still running: "
          + ", ".join(f"`{p}`" for p in pending) + ". Re-run `make_docs.py` when they finish.")
fig_md = lambda f, alt: f"![{alt}](docs/figures/{f})"
qual_md = (fig_md("qualitative.png", "Qualitative predictions") + "\n\n"
           f"*Predictions from `{qual['tag']}` on test pairs. Colours follow the official SECOND legend.*"
           if qual else "*Qualitative figure will be generated once a finished run's checkpoint is on disk.*")

# ----------------------------------------------------------------------------- README
readme = f"""# Semantic Change Detection on SECOND: a controlled study of architectures and class imbalance

This project trains and evaluates **semantic change detection (SCD)** models on the
[SECOND](https://captain-whu.github.io/SCD/) aerial dataset. Each model answers two questions for
every pixel of a before/after image pair: *did this pixel change?* and *if so, what was it before
and what is it now?*

The project has two phases, both run under one fixed protocol so every difference in the results
comes from the thing being tested:

1. **Phase A: architecture.** Three SCD designs share the same backbone, decoder, data split,
   schedule and seed.
2. **Phase B: class imbalance.** Nine imbalance techniques are tested on the winning
   architecture.

Everything runs end to end on a 16 GB Apple M4 laptop.

**Documentation:** [Excel results workbook](results/SECOND_SCD_Results.xlsx) · [Project summary](docs/PROJECT_SUMMARY.md) ·
[Methodology](docs/METHODOLOGY.md) · [Full results](docs/RESULTS.md) ·
[Diagnostics and cross-validation](docs/DIAGNOSTICS.md) · [References](docs/REFERENCES.md) ·
[Roadmap](docs/ROADMAP.md) · [Engineering notes](docs/ENGINEERING_NOTES.md)

{status}

## Results at a glance

| | Test SeK | vs. Early Fusion baseline |
|---|---:|---:|
| Early Fusion + CE (starting point) | {ef['SeK']:.2f} | — |
| SSCD + CE | {sscd['SeK']:.2f} | **{rel(sscd['SeK'], ef['SeK']):+.1f}%** |
| {ARCH_SHORT[best_arch]} + CE (selected architecture) | {base['SeK']:.2f} | **{rel(base['SeK'], ef['SeK']):+.1f}%** |
| **Best overall: {ARCH_SHORT[best['model']]} + {best['label']}** | **{best['SeK']:.2f}** | **{rel(best['SeK'], ef['SeK']):+.1f}%** |

- **Architecture matters most.** Replacing early fusion with a Siamese design raised SeK by
  {rel(sscd['SeK'], ef['SeK']):.0f}%. Adding cross-temporal attention and a semantic-consistency
  loss (Bi-SRNet-lite) raised it by another {rel(base['SeK'], sscd['SeK']):.1f}%, with only
  {PARAMS['bisrnet'] - PARAMS['sscd']:.2f}M extra parameters.
- **Imbalance handling gives a smaller gain.** The best technique, *{best['label']}*, improves SeK by
  {rel(best['SeK'], base['SeK']):+.1f}% over plain CE on the same architecture.
  {len(better_b)} of {len(B)} techniques beat the CE baseline on SeK. {oa_drop} of {len(B)} lower
  overall accuracy (OA): they trade some no-change pixels for more detected change.
- **Rare classes gain the most.** From the Early Fusion baseline to the best run, *water* IoU goes
  from {ef['pc'][4]:.1f} to {best['pc'][4]:.1f} and *playground* IoU from {ef['pc'][6]:.1f} to
  {best['pc'][6]:.1f}.
- **Most errors are missed changes, not wrong classes.** In the best run, {miss:.0f}% of changed
  pixels (averaged over the 6 classes) are predicted as *no-change*. Mixing up two land-cover
  classes is much rarer ([confusion matrix](#training-curves-and-error-analysis)). The change branch
  has more room to improve than the semantic branch.
- **The published ranking holds at lower cost.** Our runs keep the published order
  (HRSCD-str.2 < SSCD-l < Bi-SRNet) and reach {reach['early_fusion']:.0f}% / {reach['sscd']:.0f}% /
  {reach['bisrnet']:.0f}% of the published SeK. They use ¼ of the pixels (256 px instead of 512 px),
  a ResNet-18 backbone, 20 epochs and about 22 minutes of training per model on a laptop.

{fig_md("imbalance_techniques.png", "SeK by imbalance technique")}

## What is different about this project

Most SCD papers propose a new network and report one number against earlier networks. This project
is a **controlled comparison** instead:

| | Typical SCD paper | This project |
|---|---|---|
| Comparison | Often against numbers reported in other papers (different code, schedules and sometimes splits) | 3 architectures, one fixed protocol: same backbone, decoder, split, seed, schedule and metric code |
| Class imbalance | Usually one loss choice, rarely compared with alternatives | Imbalance measured first (4:1 change ratio, {imb['semantic_imbalance_ratio_max_over_min']:.0f}× between classes, 31 transition types), then {len(PLANNED_B)} techniques benchmarked |
| Balancing scope | Varies | Both branches: BCE `pos_weight`/focal/Dice/OHEM on change, class weights/Dice/focal/OHEM on semantics, plus image-level rare-class oversampling |
| Hardware | GPU training at 512 px, usually for many more epochs | One 16 GB Apple M4 laptop (MPS), 256 px, 20 epochs, ~22 min/run |
| Data handling | Download, unpack, train | 2.2 GB RAR streamed from Google Drive into a 256 px copy without writing the archive to disk; lazy PNG loading keeps RAM low |
| Reproducibility | Varies | Fixed split (seed 0), one command runs everything; docs are generated from the result files |

Design choices used in every run (standard ideas, applied consistently rather than claimed as new):

- **Temporal-swap augmentation:** T1 and T2 (and their labels) are swapped at random, so the model
  sees every change in both directions (e.g. tree → building and building → tree).
- **Cross-temporal attention with a zero-initialised gate:** training starts exactly from the SSCD
  model and learns how much attention to add.
- **Change-aware semantic loss:** semantic loss is computed only on changed pixels, as in
  Bi-SRNet, because SECOND defines semantic labels only there.
- **Model selection on a separate validation split:** checkpoints are chosen by validation SeK and
  reported once on a separate test split, so no test-set tuning.

## Pipeline

```mermaid
flowchart TD
    A[(SECOND_train_set.rar<br/>2.2 GB, Google Drive)] -->|gdown stream| B[bsdtar<br/>RAR → tar stream]
    B -->|prepare_data.py| C[Resize 512→256<br/>RGB labels → class index]
    C --> D[(data/SECOND_256<br/>2,968 pairs)]
    D --> E[analyze_imbalance.py<br/>class and transition statistics]
    D --> F[Split 70/10/20<br/>2,077 / 296 / 595, seed 0]
    F --> G[Phase A<br/>3 architectures × plain CE]
    G --> H{{Best validation SeK}}
    H --> I[Phase B<br/>{len(PLANNED_B)} imbalance techniques]
    I --> J[Test-split evaluation<br/>OA · mIoU · SeK · Fscd]
    E --> K[make_docs.py<br/>figures + README]
    J --> K
```

### Experiment protocol

```mermaid
flowchart TD
    S[Start run: model, loss, sampler] --> T[Train one epoch<br/>AdamW + OneCycle, bs 8<br/>flip / rot90 / temporal swap]
    T --> V[Evaluate on validation split]
    V --> Q{{Val SeK > best?}}
    Q -- yes --> K[Save checkpoint]
    Q -- no --> N
    K --> N{{Epoch < 20?}}
    N -- yes --> T
    N -- no --> L[Load best checkpoint]
    L --> X[Evaluate once on test split]
    X --> R[(results/tag.json<br/>metrics, confusion, history)]
```

## Models

All three share an ImageNet-pretrained **ResNet-18** encoder and an FPN-style decoder (64 channels,
stride 4).

```mermaid
flowchart LR
    subgraph EF[Early Fusion ~ HRSCD-str.2 · {PARAMS['early_fusion']}M]
      direction LR
      e1[T1 ⊕ T2<br/>6-channel input] --> e2[ResNet-18] --> e3[FPN decoder] --> e4[change head<br/>semantic head T1<br/>semantic head T2]
    end
```

```mermaid
flowchart LR
    X1[Image T1] --> ENC1[ResNet-18<br/>shared weights]
    X2[Image T2] --> ENC2[ResNet-18<br/>shared weights]
    ENC1 -- layer4 --> COT[Cross-temporal attention<br/>T1 ↔ T2, gated<br/>Bi-SRNet-lite only]
    ENC2 -- layer4 --> COT
    COT --> SD1[Semantic decoder<br/>shared] --> S1[Semantic map T1<br/>6 classes]
    COT --> SD2[Semantic decoder<br/>shared] --> S2[Semantic map T2<br/>6 classes]
    ENC1 -- 4 scales --> CAT[concat T1,T2 features]
    ENC2 -- 4 scales --> CAT
    CAT --> CD[Change decoder] --> C[Change map<br/>binary]
    S1 --> OUT[SCD output<br/>class × change mask]
    S2 --> OUT
    C --> OUT
```

| Model | Idea | Based on | Parameters |
|---|---|---|---:|
| Early Fusion | Stack T1 and T2 as 6 channels, one network, three heads | HRSCD str.2 [[2]](https://arxiv.org/abs/1810.08452) | {PARAMS['early_fusion']}M |
| SSCD | Siamese encoder; semantic decoder per date (shared) + change decoder on concatenated features | SSCD-l [[3]](https://arxiv.org/abs/2108.06103) | {PARAMS['sscd']}M |
| Bi-SRNet-lite | SSCD + cross-temporal attention on the deepest features + semantic-consistency loss | Bi-SRNet [[3]](https://arxiv.org/abs/2108.06103) | {PARAMS['bisrnet']}M |

Encoder: ResNet-18 [[7]](https://arxiv.org/abs/1512.03385), decoder: FPN-style [[8]](https://arxiv.org/abs/1612.03144).

### Loss

```mermaid
flowchart LR
    C[change logits] --> LC[change loss<br/>BCE / focal / Dice / OHEM]
    S[semantic logits T1, T2<br/>changed pixels only] --> LS[semantic loss<br/>CE / weighted / focal / Dice / OHEM]
    P[softmax T1 · T2] --> LK[consistency loss<br/>agree if unchanged,<br/>differ if changed]
    LC --> T((total = L_change + 0.5·L_sem + L_cons))
    LS --> T
    LK --> T
```

## Dataset and class imbalance

SECOND has {imb['n_pairs']:,} public 512×512 image pairs with 6 land-cover classes. Labels exist only where
change happened. Only **{imb['change_pixel_ratio']*100:.1f}%** of pixels change (no-change : change =
{imb['change_vs_nochange_ratio']} : 1). Among changed pixels, the largest class is
**{imb['semantic_imbalance_ratio_max_over_min']:.0f}×** bigger than the smallest. *Playground* appears in only
{imb['T1_images_containing_class']['playground']} of {imb['n_pairs']:,} before-images.

{fig_md("class_distribution.png", "Class distribution")}

{fig_md("transition_matrix.png", "Transition matrix")}

## Results

All numbers are on the held-out **test split (595 pairs)**. Checkpoints are selected by validation
SeK. SeK is the primary metric. Full tables: [docs/RESULTS.md](docs/RESULTS.md).

### Phase A: architecture (plain CE)

{tbl(A, "Architecture")}

{fig_md("architecture_comparison.png", "Architecture comparison")}

### Phase B: class-imbalance techniques on {ARCH_SHORT[best_arch]}

{tbl(BB, "Technique")}

{failed_md}Relative change against the CE baseline (Δ% for metrics, percentage points for class IoU):

{delta_tbl()}

{fig_md("relative_gain_vs_ce.png", "Relative gain vs CE")}

### Per-class IoU

{fig_md("per_class_iou.png", "Per-class IoU")}

### Training curves and error analysis

{fig_md("training_curves.png", "Validation SeK curves")}

{fig_md("confusion_best.png", "Confusion matrix of best run")}

### Qualitative results

{qual_md}

### Comparison with published results

{lit_tbl()}

{fig_md("literature_comparison.png", "Comparison with published results")}

> **This comparison is not like-for-like.** Published numbers use 512×512 images, usually larger
> backbones, longer training and their own splits. Ours use 256×256 (¼ of the pixels),
> ResNet-18, 20 epochs and a 70/10/20 split of the 2,968 public pairs. The table shows where the
> results sit, not a claim to beat those methods. The main result here is the controlled
> comparison inside this project.

{diag_md}
## Next steps

The full plan with time estimates is in [docs/ROADMAP.md](docs/ROADMAP.md). In short:

1. **Statistics:** finish the 3-fold cross-validation and add seeds, so every gap has a mean ± std.
2. **Cheap fixes suggested by the error analysis:** tune the change threshold (missed changes are
   the largest error), train the best configuration longer, and fix or drop OHEM.
3. **New method, a transition-aware loss:** weight each pixel by how rare its *from → to* change
   is, and compare it against CE, WCE + Dice and the SeK loss of Mamba-FCS
   [[6]](https://arxiv.org/abs/2508.08232).
4. **Published setting:** 512 px, a larger backbone and 50+ epochs on a GPU.
5. **A second dataset:** Landsat-SCD.
6. **Paper:** write it up and choose a venue.

## Reproduce

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py          # streams SECOND from Google Drive → data/SECOND_256 (needs bsdtar)
python analyze_imbalance.py     # → results/imbalance_stats.json
./run_all.sh                    # phase A + phase B (EPOCHS=20 by default); finished runs are skipped
./run_cv.sh                     # 3-fold cross-validation + train/val curves (~12 h on an M4)
python make_docs.py             # → README.md, docs/*.md, docs/figures/*.png
python make_excel.py            # → results/SECOND_SCD_Results.xlsx (Excel workbook with charts)
```

One run: `python train.py --model bisrnet --loss combo --sampler rare --epochs 20 --tag my_run`.

## Repository layout

| Path | Contents |
|---|---|
| [scd.py](scd.py) | Data loading, the 3 models, all losses, official SECOND metrics |
| [train.py](train.py) | Training and evaluation of a single run → `results/<tag>.json` |
| [run_all.sh](run_all.sh) | Phase A → pick best architecture → phase B |
| [prepare_data.py](prepare_data.py) | Streaming download + 256 px conversion |
| [analyze_imbalance.py](analyze_imbalance.py) | Class and transition statistics |
| [make_docs.py](make_docs.py) | Generates this README, `docs/` and all figures |
| [make_excel.py](make_excel.py) | Builds `results/SECOND_SCD_Results.xlsx`: all results, formulas and charts in one workbook |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Models, losses, metrics and training details |
| [docs/RESULTS.md](docs/RESULTS.md) | Every table, per run |
| [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md) | Error breakdown, cross-validation, fold-by-fold fit diagnosis |
| [run_cv.sh](run_cv.sh) | 3-fold cross-validation of all 12 configurations |
| [docs/ENGINEERING_NOTES.md](docs/ENGINEERING_NOTES.md) | Problems hit while running on a laptop and their fixes |
| [docs/REFERENCES.md](docs/REFERENCES.md) | Paper, link and code location for every model, loss and metric |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Next steps towards a paper |
| [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) | Short summary for supervisors and reviewers |
| [requirements.txt](requirements.txt) | Exact package versions |
| `results/` | Per-run JSON (metrics, confusion matrix, training history), CSV summary |
| `logs/` | Training logs |

## Limitations

- **One seed per configuration.** Several phase-B techniques differ by only a few tenths of SeK.
  Those gaps may change with another seed, so treat close results as ties.
- **Lower resolution and a short schedule** (256 px, 20 epochs). Absolute scores are lower than
  published ones, as the comparison above explains.
- **Simplified versions of the published models.** They follow the core ideas of HRSCD-str.2,
  SSCD-l and Bi-SRNet, but are not exact copies of those papers' code.
- **Test split taken from the public training release,** because SECOND's official test labels
  are not part of the download used here.

## References

Full table (each model, loss, optimiser and metric → paper, link, code location, our changes):
**[docs/REFERENCES.md](docs/REFERENCES.md)**. Main sources:

1. Yang et al., *Asymmetric Siamese Networks for Semantic Change Detection in Aerial Images*, IEEE TGRS 2022 — SECOND dataset and metrics. [arXiv:2010.05687](https://arxiv.org/abs/2010.05687)
2. Daudt et al., *Multitask Learning for Large-scale Semantic Change Detection*, CVIU 2019 — HRSCD strategies (Early Fusion). [arXiv:1810.08452](https://arxiv.org/abs/1810.08452)
3. Ding et al., *Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images*, IEEE TGRS 2022 — SSCD-l, Bi-SRNet, consistency loss. [arXiv:2108.06103](https://arxiv.org/abs/2108.06103)
4. Ding et al., *Joint Spatio-Temporal Modeling for the Semantic Change Detection in Remote Sensing Images*, IEEE TGRS 2024 — SCanNet. [arXiv:2212.05245](https://arxiv.org/abs/2212.05245)
5. Chen et al., *ChangeMamba*, IEEE TGRS 2024. [arXiv:2404.03425](https://arxiv.org/abs/2404.03425)
6. *Mamba-FCS* (SeK loss; published SECOND table), 2025. [arXiv:2508.08232](https://arxiv.org/abs/2508.08232)
7. He et al., *Deep Residual Learning* (ResNet), CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
8. Lin et al., *Feature Pyramid Networks*, CVPR 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
9. Eigen & Fergus, median-frequency balancing, ICCV 2015. [arXiv:1411.4734](https://arxiv.org/abs/1411.4734)
10. Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019. [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)
11. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
12. Milletari et al., *V-Net* (Dice loss), 3DV 2016. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
13. Shrivastava et al., *Online Hard Example Mining*, CVPR 2016. [arXiv:1604.03540](https://arxiv.org/abs/1604.03540)
14. Gupta et al., *LVIS* (repeat-factor sampling), CVPR 2019. [arXiv:1908.03195](https://arxiv.org/abs/1908.03195)
15. Loshchilov & Hutter, *AdamW*, ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) · Smith & Topin, *One-cycle / Super-Convergence*, 2017. [arXiv:1708.07120](https://arxiv.org/abs/1708.07120)
"""
open(os.path.join(scd.ROOT, "README.md"), "w").write(readme)

# ----------------------------------------------------------------------------- docs/RESULTS.md
results_md = f"""# Full results

Generated by `make_docs.py` from `results/*.json`. Test split = 595 pairs. {status}

## Phase A: architectures (plain CE)

{tbl(A, "Architecture")}

## Phase B: imbalance techniques on {ARCH_SHORT[best_arch]}

{tbl(BB, "Technique")}

### Relative to the CE baseline

{delta_tbl()}

## Per-class IoU (%)

{pc_tbl(A + B)}

## Technique definitions

| Technique | Definition | Reference |
|---|---|---|
""" + "\n".join(f"| {v[0]} | {v[1]} | {TECH_REF.get(k, '')} |" for k, v in TECH.items()) + f"""

## Published SECOND results (reference only)

{lit_tbl()}

## Per-run validation history

""" + "\n".join(
    f"<details><summary>{ARCH_SHORT[x['model']]} + {x['label']} (`{x['tag']}`)</summary>\n\n"
    "| epoch | loss | OA | mIoU | SeK | Fscd |\n|---:|---:|---:|---:|---:|---:|\n"
    + "\n".join(f"| {h['epoch']} | {h['loss']:.4f} | {h['OA']:.2f} | {h['mIoU']:.2f} | "
                f"{h['SeK']:.2f} | {h['Fscd']:.2f} |" for h in x["hist"])
    + "\n\n</details>\n" for x in ALL)
open(os.path.join(D, "RESULTS.md"), "w").write(results_md)

# summary CSV (one row per finished run)
with open(os.path.join(R, "summary.csv"), "w") as f:
    f.write("tag,model,technique,OA,mIoU,SeK,Fscd,IoU_change," +
            ",".join(f"IoU_{c}" for c in scd.CLASS_NAMES) + ",val_SeK,best_epoch,train_min\n")
    for x in ALL:
        f.write(f"{x['tag']},{x['model']},{x['label']},{x['OA']:.2f},{x['mIoU']:.2f},{x['SeK']:.2f},"
                f"{x['Fscd']:.2f},{x['IoU_c']:.2f}," + ",".join(f"{v:.2f}" for v in x["pc"])
                + f",{x['val']:.2f},{x['ep']},{x['mins']}\n")
fold_md = ["# Diagnostics: fold-by-fold\n\nGenerated by `make_docs.py` from `results/cv/*.json`.\n",
           diag_md.replace("docs/figures/", "figures/").replace("## Model diagnostics", "## Summary"),
           "\n## Per fold\n"]
for c in cvs:
    fold_md.append(f"\n### {cname(c)}\n\n| fold | best epoch | train SeK | val SeK | test SeK | "
                   "gap at end | val-loss rise | val-SeK slope (last 5 ep) |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for k in range(c["n"]):
        be = int(np.argmax(c["va_s"][k]))
        fold_md.append(f"| {k} | {be + 1} | {c['train'][k]:.2f} | {c['val'][k]:.2f} | {c['test'][k]:.2f} | "
                       f"{c['gap_end'][k]:.2f} | {c['loss_rise'][k]:.1f}% | {c['tail'][k]:+.2f} |\n")
open(os.path.join(D, "DIAGNOSTICS.md"), "w").write("".join(fold_md))
print(f"README.md, docs/RESULTS.md and {len(os.listdir(FIG))} figures written "
      f"({len(ALL)} runs, pending: {pending or 'none'})")
