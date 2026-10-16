"""Build results/SECOND_SCD_Results.xlsx: all results in one workbook written for a first-time reader.

Run make_docs.py first (it draws a few figures reused here), then:
    .venv/bin/python make_excel.py

Rules followed so a new reader can follow it:
- no unexplained abbreviations: models are called Model 1 / 2 / 3 with their full names, and every
  technique and metric is written out in full (the Glossary maps them to the short forms used in papers);
- the first sheet, "Start Here", answers: what was done, which model was used and why, why the
  main score is lower than accuracy, and why the usual metrics are not enough on their own;
- the usual metrics (accuracy, precision, recall, F1-score, confusion matrix) are reported for every run;
- every sheet has the same layout: title, "what this sheet shows", key takeaway, tables, notes, charts;
- charts are PNG images drawn here with full names, so they look the same in every program;
- raw numbers come from results/*.json; derived numbers are Excel formulas whose values are cached.
"""
import glob
import json
import math
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L
from PIL import Image as PILImage

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "docs", "figures")
XFIG = os.path.join(ROOT, "docs", "figures", "excel")
os.makedirs(XFIG, exist_ok=True)
OUT = os.path.join(R, "SECOND_SCD_Results.xlsx")
CLASSES = ["no change", "non-vegetated ground", "tree", "low vegetation", "water", "building", "playground"]
SEM = CLASSES[1:]

# ============================================================================ names (no short forms)
MODEL_NO = {"early_fusion": 1, "sscd": 2, "bisrnet": 3}
MODEL = {  # number -> (short display, full name, plain explanation, based on, why tried)
    1: ("Model 1: Early Fusion network",
        "Early Fusion convolutional neural network",
        "Puts the before and after photos together as one 6-channel image and gives it to a single network.",
        "High Resolution Semantic Change Detection, strategy 2 (Daudt and others, 2019)",
        "The simplest design. It is the starting point that the other models must beat."),
    2: ("Model 2: Siamese network",
        "Separated Semantic Segmentation and Change Detection network (late fusion)",
        "The same encoder reads the before photo and the after photo separately; a separate branch then "
        "compares them to find changes.",
        "Ding and others, 2022",
        "Tests whether reading the two photos separately works better than mixing them at the start."),
    3: ("Model 3: Bi-temporal Semantic Reasoning Network",
        "Bi-temporal Semantic Reasoning Network (lite version)",
        "Model 2 plus an attention step in which each photo 'looks at' the other, and an extra loss that "
        "asks the two predictions to agree where nothing changed.",
        "Ding and others, 2022",
        "Tests whether letting the two dates exchange information helps further."),
}
TECH = {  # (loss, sampler) -> full name, plain explanation, paper, link
    ("ce", "uniform"): ("Standard cross-entropy loss (baseline)", "Every pixel counts the same.", "Standard", ""),
    ("wce", "uniform"): ("Weighted cross-entropy loss", "Rare classes get a bigger weight in the loss.",
                         "Common practice", ""),
    ("median", "uniform"): ("Median frequency balancing", "Class weight = median class frequency ÷ class frequency.",
                            "Eigen and Fergus, 2015", "https://arxiv.org/abs/1411.4734"),
    ("cb", "uniform"): ("Class-balanced loss", "Class weights from the 'effective number' of examples per class.",
                        "Cui and others, 2019", "https://arxiv.org/abs/1901.05555"),
    ("focal", "uniform"): ("Focal loss", "Gives less weight to pixels the model already gets right easily.",
                           "Lin and others, 2017", "https://arxiv.org/abs/1708.02002"),
    ("dice", "uniform"): ("Cross-entropy + Dice loss", "Adds a loss that measures overlap, so small classes count more.",
                          "Milletari and others, 2016", "https://arxiv.org/abs/1606.04797"),
    ("ohem", "uniform"): ("Online hard example mining", "Trains only on the hardest 25% of pixels.",
                          "Shrivastava and others, 2016", "https://arxiv.org/abs/1604.03540"),
    ("ce", "rare"): ("Cross-entropy + rare-class oversampling",
                     "Same loss; training images that contain rare classes are shown more often.",
                     "Gupta and others, 2019", "https://arxiv.org/abs/1908.03195"),
    ("combo", "uniform"): ("Weighted cross-entropy + Dice loss", "Both of the above together.",
                           "Weighted loss + Milletari and others, 2016", "https://arxiv.org/abs/1606.04797"),
    ("combo", "rare"): ("Weighted cross-entropy + Dice loss + rare-class oversampling",
                        "The combination above, plus showing rare-class images more often.",
                        "As above + Gupta and others, 2019", "https://arxiv.org/abs/1908.03195"),
}
M_MAIN = "Separated Kappa (main score)"
M_F = "F-score for semantic change detection"
M_IOU = "Mean intersection over union"
M_OA = "Overall accuracy"

# ============================================================================ data
runs = {os.path.basename(p)[:-5]: json.load(open(p))
        for p in sorted(glob.glob(os.path.join(R, "[AB]_*.json")))}
imb = json.load(open(os.path.join(R, "imbalance_stats.json")))
best_arch = open(os.path.join(R, "best_arch.txt")).read().split()[-1]


def info(tag):
    r = runs[tag]
    a, t = r["args"], r["test"]
    key = (a["loss"], a["sampler"])
    return dict(tag=tag, model=a["model"], no=MODEL_NO[a["model"]], key=key, tech=TECH[key][0], t=t,
                pc=t["per_class_IoU"], val=max(h["SeK"] for h in r["history"]), ep=r["best_epoch"],
                mins=r["train_minutes"], hist=r["history"], conf=np.array(r["confusion"]))


A = sorted([info(t) for t in runs if t.startswith("A_")], key=lambda x: x["no"])
base = next(x for x in A if x["model"] == best_arch)
m1 = A[0]
B = sorted([info(t) for t in runs if t.startswith("B_")], key=lambda x: -x["t"]["SeK"])
BB = [base] + B
ALL = A + B
best = max(ALL, key=lambda x: x["t"]["SeK"])
short = lambda x: f"Model {x['no']}" + ("" if x["key"] == ("ce", "uniform") else f" + {x['tech']}")
long = lambda x: MODEL[x["no"]][0] + ("" if x["key"] == ("ce", "uniform") else f" + {x['tech']}")


def counts(x):
    C = x["conf"].T.astype(np.int64)  # rows = truth, cols = prediction
    ok = int(np.trace(C[1:, 1:]))
    return dict(total=int(C.sum()), unchanged=int(C[0].sum()), changed=int(C[1:].sum()),
                tp=int(C[1:, 1:].sum()), fn=int(C[1:, 0].sum()), fp=int(C[0, 1:].sum()), tn=int(C[0, 0]),
                ok=ok, wrong=int(C[1:, 1:].sum()) - ok)


for x in ALL:
    x["c"] = counts(x)
always_no_change = best["c"]["unchanged"] / best["c"]["total"]

CV = {}
for p in sorted(glob.glob(os.path.join(R, "cv", "*_f[0-9].json"))):
    r = json.load(open(p))
    CV.setdefault(r["tag"].rsplit("_f", 1)[0], {})[r["args"]["fold"]] = r
CV_ORDER = ["A_early_fusion_ce", "A_sscd_ce", "A_bisrnet_ce", "B_bisrnet_combo", "B_bisrnet_dice",
            "B_bisrnet_wce", "B_bisrnet_median", "B_bisrnet_cb", "B_bisrnet_focal",
            "B_bisrnet_ce_rare", "B_bisrnet_combo_rare", "B_bisrnet_ohem"]
K = 3
cv_done = sum(len(v) for v in CV.values())
cv_complete = [t for t in CV_ORDER if len(CV.get(t, {})) == K]


def cv_name(tag):
    no = 1 if "early" in tag else (2 if "sscd" in tag else 3)
    if tag.startswith("A_"):
        return f"Model {no}"
    rest = tag.split("_", 2)[2]
    key = (rest.replace("_rare", ""), "rare" if rest.endswith("_rare") else "uniform")
    return f"Model {no} + {TECH[key][0]}"


def cv_stats(tag):
    rs = [CV[tag][k] for k in range(K)]
    be = [r["best_epoch"] - 1 for r in rs]
    va = np.array([[h["SeK"] for h in r["history"]] for r in rs])
    tr = np.array([[h["train_SeK"] for h in r["history"]] for r in rs])
    vl = np.array([[h["val_loss"] for h in r["history"]] for r in rs])
    tl = np.array([[h["train_eval_loss"] for h in r["history"]] for r in rs])
    rise = float(np.mean((vl[:, -1] - vl.min(1)) / np.abs(vl.min(1))))
    g_end = float((tr[:, -1] - va[:, -1]).mean())
    slope = float(np.mean([np.polyfit(np.arange(5), v[-5:], 1)[0] for v in va]))
    if va.max() < 5:
        diag = "Failed to learn"
    elif rise > 0.10 and g_end > 3:
        diag = "Overfitting"
    elif slope > 0.15 and g_end < 2:
        diag = "Underfitting (still improving)"
    elif g_end > 3 or rise > 0.10:
        diag = "Mild overfitting"
    else:
        diag = "Good fit"
    return dict(test=[r["test"]["SeK"] for r in rs], val=float(np.mean([va[i, b] for i, b in enumerate(be)])),
                train=float(np.mean([tr[i, b] for i, b in enumerate(be)])), rise=rise, diag=diag,
                va=va, tr=tr, vl=vl, tl=tl, rs=rs)


CVS = {t: cv_stats(t) for t in cv_complete}
ref_cv = CVS.get("A_bisrnet_ce")
wins_all = [t for t in cv_complete if t.startswith("B_") and ref_cv
            and sum(a > b for a, b in zip(CVS[t]["test"], ref_cv["test"])) == K]
cv_top = sorted([t for t in cv_complete if t.startswith("B_")], key=lambda t: -np.mean(CVS[t]["test"]))[:3]

LIT = [("High Resolution Semantic Change Detection, strategy 2", 85.49, 64.43, 10.69, 49.22),
       ("High Resolution Semantic Change Detection, strategy 4", 86.62, 71.15, 18.80, 58.21),
       ("Separated Semantic Segmentation and Change Detection (late fusion)", 87.19, 72.60, 21.86, 61.22),
       ("Bi-temporal Semantic Reasoning Network", 87.84, 73.41, 23.22, 62.61),
       ("Semantic Change Network (SCanNet)", 87.86, 73.42, 23.94, 63.66),
       ("ChangeMamba (state-space model)", 88.12, 73.68, 24.11, 64.03)]

# ============================================================================ charts (full names)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
wrap = lambda s, w=34: "\n".join(textwrap.wrap(s, w))


def clean(ax, grid="x"):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y" if grid == "x" else "x", visible=False)
    ax.tick_params(length=0)


def savefig(fig, name):
    fig.savefig(os.path.join(XFIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def hbars(labels, vals, colors, title, xlabel, fname, fmt="{:.2f}", ref=None, width=8.5):
    fig, ax = plt.subplots(figsize=(width, 0.5 * len(labels) + 1.4))
    y = np.arange(len(labels))
    ax.barh(y, vals, 0.6, color=colors, edgecolor=SURF, linewidth=1.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    lo, hi = min(0, min(vals)), max(vals)
    for i, v in enumerate(vals):
        ax.text(v + (hi - lo) * 0.012 if v >= 0 else v - (hi - lo) * 0.012, i, fmt.format(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1)
    ax.set_xlim(lo * 1.25 if lo < 0 else 0, hi * 1.18)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    clean(ax)
    savefig(fig, fname)


# 1. the step-by-step journey
steps = [(f"Model 1\nEarly Fusion\n(starting point)", m1["t"]["SeK"], GRAY),
         ("Model 2\nSiamese network", A[1]["t"]["SeK"], GRAY),
         ("Model 3\nBi-temporal Semantic\nReasoning Network", base["t"]["SeK"], GRAY),
         (f"Model 3 +\n{wrap(best['tech'], 22)}\n(best)", best["t"]["SeK"], BLUE)]
fig, ax = plt.subplots(figsize=(9, 4.3))
xs = np.arange(len(steps))
ax.bar(xs, [v for _, v, _ in steps], 0.55, color=[c for _, _, c in steps], edgecolor=SURF, linewidth=1.5)
for i, (_, v, c) in enumerate(steps):
    ax.text(i, v + 0.35, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    if i:
        ax.text(i, v / 2, f"{(v - steps[0][1]) / steps[0][1] * 100:+.0f}%\nvs Model 1", ha="center", va="center",
                fontsize=9, color="white" if c == BLUE else INK)
ax.set_xticks(xs, [n for n, _, _ in steps], fontsize=9)
ax.set_ylim(0, max(v for _, v, _ in steps) * 1.2)
ax.set_ylabel("Separated Kappa on the test set")
ax.set_title("Main score, step by step (higher is better)")
clean(ax, "y")
savefig(fig, "journey.png")

# 2. three models, four official scores
mets = [("SeK", "Separated Kappa\n(main score)"), ("Fscd", "F-score for semantic\nchange detection"),
        ("mIoU", "Mean intersection\nover union"), ("OA", "Overall accuracy")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharey=True)
for ax, (k, t) in zip(axes, mets):
    v = [x["t"][k] for x in A]
    y = np.arange(3)
    ax.barh(y, v, 0.6, color=[BLUE if x["model"] == best_arch else GRAY for x in A], edgecolor=SURF)
    for i, vv in enumerate(v):
        ax.text(vv + max(v) * 0.02, i, f"{vv:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [f"Model {x['no']}" for x in A])
    ax.set_xlim(0, max(v) * 1.25)
    ax.set_title(t, fontsize=10)
    clean(ax)
axes[0].invert_yaxis()  # axes share y, so invert once: Model 1 on top
fig.suptitle("Step 1: the three models compared (all with standard cross-entropy loss)", x=0.01, ha="left",
             fontweight="bold", y=1.08)
savefig(fig, "models.png")

# 3. techniques, main score; 4. relative change
hbars([wrap(x["tech"], 40) for x in BB], [x["t"]["SeK"] for x in BB],
      [BLUE if x is best else (INK2 if x is base else GRAY) for x in BB],
      "Step 2: rare-class methods on Model 3 (main score)", "Separated Kappa on the test set",
      "techniques.png", ref=base["t"]["SeK"], width=9)
ok = [x for x in B if x["val"] >= 0.5 * base["val"]]
d = [(x["t"]["SeK"] - base["t"]["SeK"]) / base["t"]["SeK"] * 100 for x in ok]
hbars([wrap(x["tech"], 40) for x in ok], d, [BLUE if v >= 0 else RED for v in d],
      "Change in main score compared with the standard loss (blue = better)",
      "Change in Separated Kappa (%)", "techniques_change.png", fmt="{:+.1f}%", ref=0, width=9)

# 5. standard metrics for the three models and the best method
show = A + [best]
names = [f"Model {x['no']}" if x in A else "Model 3 +\n" + wrap(best["tech"], 24) for x in show]
std = {"Accuracy (changed vs\nunchanged)": [(x["c"]["tp"] + x["c"]["tn"]) / x["c"]["total"] * 100 for x in show],
       "Precision": [x["c"]["tp"] / (x["c"]["tp"] + x["c"]["fp"]) * 100 for x in show],
       "Recall": [x["c"]["tp"] / (x["c"]["tp"] + x["c"]["fn"]) * 100 for x in show]}
std["F1-score"] = [2 * p * r / (p + r) for p, r in zip(std["Precision"], std["Recall"])]
fig, ax = plt.subplots(figsize=(10, 4.2))
w = 0.2
xs = np.arange(len(show))
for j, (k, v) in enumerate(std.items()):
    ax.bar(xs + (j - 1.5) * w, v, w * 0.92, label=k.replace("\n", " "), color=[BLUE, ORANGE, AQUA, "#4a3aa7"][j])
    for i, vv in enumerate(v):
        ax.text(xs[i] + (j - 1.5) * w, vv + 1, f"{vv:.0f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(xs, names, fontsize=9)
ax.set_ylim(0, 105)
ax.set_ylabel("%")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
ax.set_title("Standard metrics for finding changes (changed vs unchanged pixels, test set)")
clean(ax, "y")
savefig(fig, "standard_metrics.png")

# 6. error breakdown
fig, ax = plt.subplots(figsize=(10, 0.45 * len(ALL) + 1.8))
y = np.arange(len(ALL))
left = np.zeros(len(ALL))
for k, col, lab in (("ok", BLUE, "Correct class"), ("wrong", AQUA, "Change found, wrong class"),
                    ("fn", ORANGE, "Change missed (said 'no change')")):
    v = np.array([x["c"][k] / x["c"]["changed"] * 100 for x in ALL])
    ax.barh(y, v, 0.6, left=left, color=col, edgecolor=SURF, linewidth=1.5, label=lab)
    for i in range(len(ALL)):
        if v[i] > 7:
            ax.text(left[i] + v[i] / 2, i, f"{v[i]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if col == BLUE else INK)
    left += v
ax.set_yticks(y, [wrap(short(x), 46) for x in ALL], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("% of truly changed pixels")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=3, fontsize=9)
ax.set_title("What happens to the pixels that really changed")
clean(ax)
savefig(fig, "errors.png")

# 7. cross-validation
if cv_complete:
    order = sorted(cv_complete, key=lambda t: -np.mean(CVS[t]["test"]))
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(order) + 1.5))
    for i, t in enumerate(order):
        v = np.array(CVS[t]["test"])
        m, s = v.mean(), v.std(ddof=1)
        ax.plot([m - s, m + s], [i, i], color=BLUE, lw=2)
        ax.scatter(v, [i] * K, s=22, color=GRAY, zorder=3, edgecolor=SURF)
        ax.scatter([m], [i], s=60, color=BLUE, zorder=4, edgecolor=SURF, linewidth=2)
        ax.text(max(v.max(), m + s) + 0.3, i, f"{m:.2f} ± {s:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(order)), [wrap(cv_name(t), 46) for t in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(np.max(CVS[t]["test"]) for t in order) * 1.25)
    ax.set_xlabel("Separated Kappa on the test set (small grey dot = one fold, big dot = average, line = ± 1 standard deviation)")
    ax.set_title("Stability check: each configuration trained 3 times (3-fold cross-validation)")
    clean(ax)
    savefig(fig, "cv.png")

    # 8. learning curves, one panel per configuration
    n = len(cv_complete)
    cols = 4
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, 3.2 * rows_n + 0.6), squeeze=False)
    for ax, t in zip(axes.ravel(), cv_complete):
        s = CVS[t]
        e = np.arange(1, s["tr"].shape[1] + 1)
        for v, col, lab in ((s["tr"], ORANGE, "training data"), (s["va"], BLUE, "validation data")):
            ax.fill_between(e, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=col, alpha=0.15, lw=0)
            ax.plot(e, v.mean(0), color=col, lw=2, label=lab)
        ax.set_title(wrap(cv_name(t), 38) + f"\n→ {s['diag']}", fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Score on training data vs validation data after each epoch (average of 3 folds; "
                 "x-axis = epoch, y-axis = Separated Kappa)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "learning_curves.png")

# 9. published comparison
lab = [wrap(m[0], 44) for m in LIT] + [wrap("Ours: " + long(x), 44) for x in A + [best]]
val = [m[3] for m in LIT] + [x["t"]["SeK"] for x in A + [best]]
hbars(lab, val, [GRAY] * len(LIT) + [BLUE] * (len(A) + 1),
      "Main score: published papers (grey, 512-pixel images) vs ours (blue, 256-pixel images)",
      "Separated Kappa", "published.png", width=10)

# ============================================================================ workbook helpers
NAVY = "0D366B"
F = lambda **k: Font(name="Arial", **{"size": 10, **k})
HEAD = PatternFill("solid", fgColor="184F95")
BAND = PatternFill("solid", fgColor="F3F6FB")
BEST = PatternFill("solid", fgColor="DDF1E6")
TAKE = PatternFill("solid", fgColor="E8F3EC")
NOTE = PatternFill("solid", fgColor="FFF8E6")
QBOX = PatternFill("solid", fgColor="EEF4FC")
NONE = PatternFill()
thin = Side(style="thin", color="D0D0D0")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
LEFT_BAR = Border(left=Side(style="thick", color="1BAF7A"))
PT = 13.5
wb = Workbook()
wb.remove(wb.active)
REFS = {}


def wd(ws, c):
    return ws.column_dimensions[L(c)].width or 8.43


def nlines(text, chars):
    return sum(max(1, math.ceil(len(p) / max(chars, 1))) for p in str(text).split("\n"))


def block(ws, row, text, n, font, fill=NONE, border=None, min_h=15):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n)
    c = ws.cell(row, 1, text)
    c.font, c.fill = font, fill
    c.alignment = Alignment(wrap_text=True, vertical="top", indent=1 if fill is not NONE else 0)
    if border:
        c.border = border
    chars = sum(wd(ws, k) for k in range(1, n + 1)) * 1.05 * 10 / font.size - 4
    ws.row_dimensions[row].height = max(min_h, nlines(text, chars) * PT * font.size / 10 + 6)
    return row + 1


def new_sheet(name, title, what, takeaway, widths):
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False
    for i, w_ in enumerate(widths):
        ws.column_dimensions[L(i + 1)].width = w_
    n = len(widths)
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    r = block(ws, 1, title, n, F(size=16, bold=True, color=NAVY), min_h=26)
    r = block(ws, r, "What this sheet shows: " + what, n, F(color="52514E"))
    r = block(ws, r, "Key takeaway: " + takeaway, n, F(size=11, bold=True, color="0B4F2F"), fill=TAKE,
              border=LEFT_BAR, min_h=20)
    return ws, r + 1, n


def heading(ws, row, text):
    ws.cell(row, 1, text).font = F(size=12, bold=True, color=NAVY)
    ws.row_dimensions[row].height = 20
    return row + 1


def note(ws, row, text, n):
    return block(ws, row, text, n, F(size=9, italic=True, color="52514E"), fill=NOTE) + 1


def qa(ws, row, question, answer, n):
    row = block(ws, row, question, n, F(size=11, bold=True, color=NAVY), fill=QBOX)
    return block(ws, row, answer, n, F(size=10), fill=QBOX) + 1


def table(ws, row, headers, rows, fmts=None, best_row=None, wrap_cols=()):
    hh = 1
    for j, h in enumerate(headers):
        c = ws.cell(row, 1 + j, h)
        c.font, c.fill, c.border = F(bold=True, color="FFFFFF"), HEAD, BOX
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        hh = max(hh, nlines(h, wd(ws, 1 + j) * 0.95 - 1))
    ws.row_dimensions[row].height = hh * PT + 8
    for i, rv in enumerate(rows):
        rr = row + 1 + i
        nl = 1
        for j, v in enumerate(rv):
            c = ws.cell(rr, 1 + j, v)
            is_text = isinstance(v, str) and not v.startswith("=")
            wrp = j == 0 or j in wrap_cols
            c.font = F(bold=(best_row == i))
            c.border = BOX
            c.fill = BEST if best_row == i else (BAND if i % 2 else NONE)
            c.alignment = Alignment(vertical="center", wrap_text=wrp,
                                    horizontal="left" if is_text or j == 0 else "right")
            if fmts and j < len(fmts) and fmts[j]:
                c.number_format = fmts[j]
            if wrp and is_text:
                nl = max(nl, nlines(v, wd(ws, 1 + j) * 1.05 - 2))
        ws.row_dimensions[rr].height = max(16, nl * PT + 5)
    return row + 1, row + len(rows), row + len(rows) + 2


def picture(ws, row, path, n, max_px=1000):
    if not os.path.exists(path):
        return note(ws, row, f"(Figure {os.path.basename(path)} not found.)", n)
    w_, h_ = PILImage.open(path).size
    target = min(max_px, sum(wd(ws, k) for k in range(1, n + 1)) * 7.0, w_)
    img = XLImage(path)
    img.width, img.height = int(target), int(h_ * target / w_)
    ws.add_image(img, f"A{row}")
    px, k = 0.0, row
    while px < img.height:
        ws.row_dimensions[k].height = 15
        px += 20
        k += 1
    return k + 2


def link(cell, url, text=None):
    cell.value = text or url
    cell.hyperlink = url
    cell.font = F(color="1C5CAB", underline="single")


X = lambda f: os.path.join(XFIG, f)
D = lambda f: os.path.join(FIG, f)
S_START, S_MODELS, S_STD = "Start Here", "Models Used", "Standard Metrics"
S_STEP1, S_STEP2 = "Step 1 Compare Models", "Step 2 Rare-Class Methods"
q = lambda s: f"'{s}'"

# ============================================================================ Models Used
ws, r, n = new_sheet(S_MODELS, "Which machine learning models were used, and why",
                     "the three deep-learning models built for this project, what each one does, the paper it is "
                     "based on, why it was tried, and how well it did.",
                     f"All three are convolutional neural networks. Model 3 (Bi-temporal Semantic Reasoning Network) "
                     f"scored highest, so it was used for all later experiments.", [26, 34, 44, 34, 36, 14, 22])
rows = []
for x in A:
    m = MODEL[x["no"]]
    rows.append([m[0], m[1], m[2], m[3], m[4], x["t"]["SeK"],
                 "Best — used for Step 2" if x["model"] == best_arch else "Not chosen"])
_, _, r = table(ws, r, ["Model", "Full name", "How it works (plain words)", "Based on paper", "Why we tried it",
                        "Main score (Separated Kappa)", "Decision"], rows, fmts=[None] * 5 + ["0.00", None],
                best_row=[x["model"] for x in A].index(best_arch), wrap_cols=(1, 2, 3, 4, 6))
r = heading(ws, r, "What all three models have in common")
common = [
    ("Type of model", "Deep learning: convolutional neural network with an encoder (reads the image) and a decoder "
     "(draws the answer map)."),
    ("Type of learning", "Supervised learning: every training pixel comes with the correct answer."),
    ("Encoder", "Residual Network with 18 layers, pre-trained on the ImageNet photo collection."),
    ("Decoder", "Feature Pyramid Network style: combines details from several image scales."),
    ("Output", "For every pixel: (1) changed or not, (2) land-cover class before, (3) land-cover class after."),
    ("Programming", "Written from scratch in Python with the PyTorch library (file scd.py). The models follow the "
     "ideas of the papers but are simplified versions, not copies of the original code."),
    ("Why the same encoder for all", "So that differences between the models come only from their design, "
     "not from a bigger or smaller encoder."),
]
_, _, r = table(ws, r, ["Item", "Description"], [list(c) for c in common], wrap_cols=(1,))
for k in range(r - 2 - len(common), r - 1):
    ws.merge_cells(start_row=k, start_column=2, end_row=k, end_column=n)
for i, (_, txt) in enumerate(common):
    k = r - 1 - len(common) + i
    ws.row_dimensions[k].height = max(18, nlines(txt, sum(wd(ws, c) for c in range(2, n + 1)) - 4) * PT + 6)
r = picture(ws, r, X("models.png"), n, 1000)

# ============================================================================ Step 1
ws, r, n = new_sheet(S_STEP1, "Step 1: comparing the three models",
                     "the three models trained in exactly the same way (same data split, same encoder, 20 "
                     "training rounds), all with the standard cross-entropy loss. All scores are on the 595 test "
                     "image pairs the models never saw during training.",
                     f"Model 3 is best. Changing the model design alone raised the main score by "
                     f"{(base['t']['SeK'] - m1['t']['SeK']) / m1['t']['SeK']:.0%} compared with Model 1 — the "
                     "biggest improvement in the whole project.",
                     [40, 14, 16, 14, 13, 12, 14, 13])
start = r + 1
rows = []
for i, x in enumerate(A):
    rr = start + i
    rows.append([MODEL[x["no"]][0], x["t"]["SeK"], x["t"]["Fscd"], x["t"]["mIoU"], x["t"]["OA"], x["mins"],
                 f"=(B{rr}-$B${start})/$B${start}", f"=RANK(B{rr},$B${start}:$B${start + 2})"])
f1, l1, r = table(ws, r, ["Model", M_MAIN + " (%)", M_F + " (%)", M_IOU + " (%)", M_OA + " (%)",
                          "Training time (minutes)", "Main score gain over Model 1", "Rank"], rows,
                  fmts=[None, "0.00", "0.00", "0.00", "0.00", "0.0", "+0%;-0%;\"—\"", "0"],
                  best_row=[x["model"] for x in A].index(best_arch))
REFS["m1"], REFS["m3"] = f1, f1 + 2
r = note(ws, r, "Green row = model chosen for Step 2 (chosen by its score on the validation data, not the test data). "
         "Gain and rank are Excel formulas. The four scores are explained on the 'Start Here' and 'Glossary' sheets.", n)
r = picture(ws, r, X("models.png"), n, 1000)

# ============================================================================ Step 2
n_better = sum(x["t"]["SeK"] > base["t"]["SeK"] for x in B)
worse = [x["tech"] for x in B if x["t"]["SeK"] <= base["t"]["SeK"]]
ws, r, n = new_sheet(S_STEP2, "Step 2: methods for rare classes, tested on Model 3",
                     "nine ways of helping the model learn rare classes (water, playground), each trained on "
                     "Model 3 and compared with the standard cross-entropy loss (first row).",
                     f"{n_better} of 9 methods beat the standard loss. Best in this run: {B[0]['tech']} "
                     f"(+{(B[0]['t']['SeK'] - base['t']['SeK']) / base['t']['SeK']:.1%}). Worse: "
                     f"{'; '.join(worse)}.", [40, 44, 28, 13, 15, 13, 12, 12, 12, 13, 9, 11])
start = r + 1
rows = []
for i, x in enumerate(BB):
    rr = start + i
    t_ = TECH[x["key"]]
    rows.append([t_[0], t_[1], t_[2], x["t"]["SeK"], x["t"]["Fscd"], x["t"]["mIoU"], x["t"]["OA"], x["pc"][4],
                 x["pc"][6], f"=(D{rr}-$D${start})/$D${start}", f"=RANK(D{rr},$D${start}:$D${start + len(BB) - 1})",
                 "— (reference)" if i == 0 else f"=IF(D{rr}>$D${start},\"Yes\",\"No\")"])
f2, l2, r = table(ws, r, ["Method", "What it does", "Paper", M_MAIN + " (%)", M_F + " (%)", M_IOU + " (%)",
                          M_OA + " (%)", "Water: intersection over union (%)",
                          "Playground: intersection over union (%)", "Main score change vs standard loss",
                          "Rank", "Better than standard loss?"], rows,
                  fmts=[None] * 3 + ["0.00"] * 6 + ["+0.0%;-0.0%;\"—\"", "0", None], best_row=1, wrap_cols=(1, 2))
REFS["tech"] = (f2, l2)
for i, x in enumerate(BB):
    if TECH[x["key"]][3]:
        c = ws.cell(f2 + i, 3)
        link(c, TECH[x["key"]][3], TECH[x["key"]][2])
        c.border, c.alignment = BOX, Alignment(wrap_text=True, vertical="center")
    ws.cell(f2 + i, 12).alignment = Alignment(horizontal="center", vertical="center")
ws.conditional_formatting.add(f"J{f2 + 1}:J{l2}", ColorScaleRule(
    start_type="num", start_value=-0.08, start_color="F4C7C3", mid_type="num", mid_value=0,
    mid_color="FFFFFF", end_type="num", end_value=0.08, end_color="B7E1CD"))
r = note(ws, r, "Green row = best method in this run. Red/green shading = worse/better than the standard loss. "
         "Online hard example mining did not train properly with its default setting (hardest 25% of pixels); "
         "this shows that setting failing, not that the method can never work. Each method was trained once here; "
         "the 'Stability Check' sheet repeats every method 3 times. There, the top methods are within about 0.1 "
         "of each other, so they should be treated as equally good.", n)
r = picture(ws, r, X("techniques.png"), n, 900)
r = picture(ws, r, X("techniques_change.png"), n, 900)

# ============================================================================ Standard metrics
_acc = [(x["c"]["tp"] + x["c"]["tn"]) / x["c"]["total"] for x in ALL if x["val"] >= 0.5 * base["val"]]
acc_lo, acc_hi = min(_acc), max(_acc)
ws, r, n = new_sheet(S_STD, "Standard metrics: accuracy, precision, recall, F1-score and confusion matrix",
                     "the usual classification metrics for every run, computed for the question 'did this pixel "
                     "change or not?'. All numbers come from pixel counts on the test set (both dates together); "
                     "the metric columns are Excel formulas you can check.",
                     f"Accuracy looks high ({acc_lo:.0%}–{acc_hi:.0%}) for every model, even the weak ones, because "
                     f"{always_no_change:.0%} of pixels never change. Recall and F1-score show the real "
                     "differences: the best method finds many more changes than Model 1.",
                     [44, 15, 14, 14, 14, 14, 12, 12, 12, 12, 14])
hdr = ["Model + method", "Truly changed, predicted changed (true positives)",
       "Truly changed, predicted unchanged (false negatives)", "Truly unchanged, predicted changed (false positives)",
       "Truly unchanged, predicted unchanged (true negatives)", "All pixels",
       "Accuracy", "Precision", "Recall", "F1-score", "Accuracy of a model that always says 'no change'"]
start = r + 1
rows = []
for i, x in enumerate(ALL):
    rr = start + i
    c = x["c"]
    rows.append([long(x), c["tp"], c["fn"], c["fp"], c["tn"], f"=SUM(B{rr}:E{rr})",
                 f"=(B{rr}+E{rr})/F{rr}", f"=B{rr}/(B{rr}+D{rr})", f"=B{rr}/(B{rr}+C{rr})",
                 f"=2*H{rr}*I{rr}/(H{rr}+I{rr})", f"=(D{rr}+E{rr})/F{rr}"])
f3, l3, r = table(ws, r, hdr, rows, fmts=[None] + ["#,##0"] * 5 + ["0.0%"] * 5)
ws.conditional_formatting.add(f"J{f3}:J{l3}", ColorScaleRule(
    start_type="min", start_color="FFFFFF", end_type="max", end_color="9EC5F4"))
r = note(ws, r, "Accuracy = (true positives + true negatives) ÷ all pixels. Precision = true positives ÷ all pixels "
         "predicted as changed. Recall = true positives ÷ all truly changed pixels. F1-score = harmonic mean of "
         "precision and recall. Positive = 'changed'. Blue shading = F1-score.", n)
r = picture(ws, r, X("standard_metrics.png"), n, 950)
r = heading(ws, r, "How these relate to the metrics used in change-detection papers")
rel = [
    ("Accuracy", "Overall accuracy (official SECOND metric) is the same idea over 7 labels: 'no change' + 6 "
     "land-cover classes."),
    ("Precision, recall, F1-score", "F-score for semantic change detection (official) is the F1-score computed on "
     "changed pixels, but a change only counts as correct if the land-cover class is also correct."),
    ("Confusion matrix", "Shown in full (7 × 7 classes) on the 'Errors' sheet. The four counts above are taken "
     "from it."),
    ("Cross-entropy (log loss)", "This is the training loss. Its value on training and validation data after "
     "every epoch is on the 'Learning Curves' sheet. Here it is combined with other loss terms, so it is not a pure "
     "log loss."),
    ("ROC curve and area under it", "Not computed. It needs the model's probability for every pixel, and the "
     "experiments stored only the final yes/no decisions (threshold 0.5). It can be added from a saved model."),
    ("Why not use only these?", f"With {always_no_change:.0%} unchanged pixels, accuracy rewards saying 'no change'. "
     "And these metrics only check whether a change was found, not whether its land-cover classes are right. "
     "Separated Kappa checks both and removes the easy 'no change' part, which is why papers use it as the main score."),
]
_, _, r = table(ws, r, ["Metric", "Relation"], [list(x_) for x_ in rel], wrap_cols=(1,))
for k in range(r - 2 - len(rel), r - 1):
    ws.merge_cells(start_row=k, start_column=2, end_row=k, end_column=n)
for i, (_, txt) in enumerate(rel):
    k = r - 1 - len(rel) + i
    ws.row_dimensions[k].height = max(18, nlines(txt, sum(wd(ws, c_) for c_ in range(2, n + 1)) - 4) * PT + 6)

# ============================================================================ Results per class
ws, r, n = new_sheet("Results per Class", "How well is each land-cover class recognised?",
                     "intersection over union (overlap between predicted and true area, 0–100) for every class and "
                     "every run. Darker blue = better.",
                     f"Rare classes improve the most: water from {m1['pc'][4]:.1f} (Model 1) to {best['pc'][4]:.1f}, "
                     f"and playground from {m1['pc'][6]:.1f} to {best['pc'][6]:.1f} (best run).", [50] + [14] * 7)
f4, l4, r = table(ws, r, ["Model + method"] + CLASSES, [[long(x)] + list(x["pc"]) for x in ALL],
                  fmts=[None] + ["0.0"] * 7)
ws.conditional_formatting.add(f"B{f4}:H{l4}", ColorScaleRule(
    start_type="num", start_value=0, start_color="FFFFFF", mid_type="num", mid_value=40, mid_color="9EC5F4",
    end_type="num", end_value=90, end_color="3987E5"))
r = note(ws, r, "'No change' is always easy (about 85) because it covers 80% of all pixels. Water and playground "
         "together are under 3% of changed pixels.", n)

# ============================================================================ Errors
miss_big = sum(1 for x in ALL if x["c"]["fn"] > x["c"]["wrong"])
ws, r, n = new_sheet("Errors", "What kind of mistakes do the models make?",
                     "every truly changed pixel is given the correct class, the wrong class, or is missed (the model "
                     "says 'no change'). Also the confusion matrix of the best model.",
                     f"Missing a change is the biggest mistake in {miss_big} of {len(ALL)} runs. Getting the class "
                     "wrong is less common. So the next improvement should help the model find more changes.",
                     [50, 15, 14, 14, 14, 12, 12, 12])
start = r + 1
rows = []
for i, x in enumerate(ALL):
    rr = start + i
    c = x["c"]
    rows.append([long(x), c["changed"], c["ok"], c["wrong"], c["fn"], f"=C{rr}/B{rr}", f"=D{rr}/B{rr}",
                 f"=E{rr}/B{rr}"])
f5, l5, r = table(ws, r, ["Model + method", "Truly changed pixels", "Correct class", "Wrong class",
                          "Missed (said 'no change')", "Correct %", "Wrong class %", "Missed %"], rows,
                  fmts=[None] + ["#,##0"] * 4 + ["0.0%"] * 3)
ws.conditional_formatting.add(f"H{f5}:H{l5}", ColorScaleRule(
    start_type="min", start_color="FFFFFF", end_type="max", end_color="F4B183"))
r = note(ws, r, "Counts from each run's test confusion matrix. Percentages are formulas. Orange = share of changes missed.", n)
r = picture(ws, r, X("errors.png"), n, 950)
r = picture(ws, r, D("confusion_best.png"), n, 600)

# ============================================================================ Stability check
ws, r, n = new_sheet("Stability Check", "Are the results stable? (3-fold cross-validation)",
                     "each configuration trained 3 times, each time on a different part of the training data "
                     "(1,582 training and 791 validation pairs), and scored on the same 595 test pairs. "
                     f"Finished: {cv_done} of {K * len(CV_ORDER)} trainings.",
                     ("The order of the three models is the same in every fold. "
                      f"{len(wins_all)} methods beat the standard loss in all 3 folds. The top three "
                      f"({'; '.join(cv_name(t).split(' + ', 1)[1] for t in cv_top)}) are within "
                      f"{np.mean(CVS[cv_top[0]]['test']) - np.mean(CVS[cv_top[-1]]['test']):.2f} of each other, "
                      "so they are equally good.") if len(cv_top) == 3 else "Cross-validation still running.",
                     [50, 9, 9, 9, 9, 12, 11, 12, 12, 12, 16, 22])
start = r + 1
rowof = {t: start + i for i, t in enumerate(CV_ORDER)}
rows = []
for t in CV_ORDER:
    rr = rowof[t]
    folds = CV.get(t, {})
    tests = [folds[k]["test"]["SeK"] if k in folds else None for k in range(K)]
    s = CVS.get(t)
    ref = "A_early_fusion_ce" if t.startswith("A_") else "A_bisrnet_ce"
    qq = rowof[ref]
    win = "— (reference)" if ref == t else \
        f"=IF(AND(E{rr}=3,E{qq}=3),((B{rr}>B{qq})+(C{rr}>C{qq})+(D{rr}>D{qq}))&\" of 3\",\"—\")"
    rows.append([cv_name(t)] + tests + [
        f"=COUNT(B{rr}:D{rr})", f"=IF(E{rr}>0,AVERAGE(B{rr}:D{rr}),\"—\")",
        f"=IF(E{rr}>1,STDEV(B{rr}:D{rr}),\"—\")", s["val"] if s else None, s["train"] if s else None,
        f"=IF(AND(ISNUMBER(I{rr}),ISNUMBER(H{rr})),I{rr}-H{rr},\"—\")", win,
        s["diag"] if s else ("running" if folds else "not started")])
f6, l6, r = table(ws, r, ["Configuration", "Fold 1", "Fold 2", "Fold 3", "Folds done", "Average test score",
                          "Standard deviation", "Average validation score", "Average training score",
                          "Gap (training − validation)", "Beats its reference in", "Fit diagnosis"], rows,
                  fmts=[None] + ["0.00"] * 3 + ["0"] + ["0.00"] * 5 + [None, None], wrap_cols=(11,))
for i in range(len(CV_ORDER)):
    ws.cell(f6 + i, 11).alignment = Alignment(horizontal="center", vertical="center")
r = note(ws, r, "All scores are Separated Kappa. Reference: Models 2 and 3 are compared with Model 1; methods with "
         "Model 3 + standard loss. Fit diagnosis (averaged over folds): Overfitting = validation loss ends more "
         "than 10% above its lowest value AND training score is more than 3 points above validation score; Mild "
         "overfitting = only one of the two; Underfitting = validation score still rising by more than 0.15 per "
         "epoch with a gap below 2; Failed = best validation score below 5. These thresholds were chosen for this "
         "project. Cross-validation models train on fewer images (1,582) than the main runs (2,077), so their "
         "scores are slightly lower; this sheet is for stability and ranking.", n)
r = picture(ws, r, X("cv.png"), n, 950)

# ============================================================================ Learning curves
dc = {}
for t in cv_complete:
    dc[CVS[t]["diag"]] = dc.get(CVS[t]["diag"], 0) + 1
ws, r, n = new_sheet("Learning Curves", "Overfitting or underfitting? Training vs validation curves",
                     "the score on the model's own training data (orange) and on unseen validation data (blue) after "
                     "every epoch (one pass over the training data), averaged over the 3 folds.",
                     ("Diagnosis: " + ", ".join(f"{v} × {k.lower()}" for k, v in dc.items()) + ". The training "
                      "score keeps rising while the validation score grows only slowly after about 12–15 epochs. "
                      "More data or regularisation should help more than simply training longer.")
                     if cv_complete else "Cross-validation still running.", [14] * 10)
r = picture(ws, r, X("learning_curves.png"), n, 1000)
if cv_complete:
    r = heading(ws, r, "Loss values (cross-entropy based) on training and validation data — average of 3 folds")
    E = CVS[cv_complete[0]]["tl"].shape[1]
    rows = [[cv_name(t), float(CVS[t]["tl"].mean(0)[0]), float(CVS[t]["tl"].mean(0)[-1]),
             float(CVS[t]["vl"].mean(0)[0]), float(CVS[t]["vl"].mean(0)[-1]), float(CVS[t]["vl"].mean(0).min()),
             CVS[t]["diag"]] for t in cv_complete]
    ws.column_dimensions["A"].width = 50
    _, _, r = table(ws, r, ["Configuration", "Training loss, epoch 1", f"Training loss, epoch {E}",
                            "Validation loss, epoch 1", f"Validation loss, epoch {E}", "Lowest validation loss",
                            "Fit diagnosis"], rows, fmts=[None] + ["0.000"] * 5 + [None], wrap_cols=(6,))

# ============================================================================ Training progress
ws, r, n = new_sheet("Training Progress", "Validation score after every epoch (main runs)",
                     "the validation Separated Kappa after each of the 20 epochs for the 12 main runs. The epoch "
                     "with the best validation score was kept and scored once on the test set.",
                     "Models 2 and 3 learn much faster and reach higher scores than Model 1; all runs level off "
                     "in the last epochs.", [8] + [14] * 12)
E = len(ALL[0]["hist"])
_, _, r = table(ws, r, ["Epoch"] + [short(x) for x in ALL], [[e + 1] + [x["hist"][e]["SeK"] for x in ALL]
                                                            for e in range(E)], fmts=["0"] + ["0.00"] * len(ALL))
ws.row_dimensions[r - 2 - E - 1].height = 80

# ============================================================================ Comparison with papers
reach = {1: 10.69, 2: 21.86, 3: 23.22}
ws, r, n = new_sheet("Comparison with Papers", "How do our scores compare with published papers?",
                     "our results next to results reported in papers on the same dataset. The papers use 512 × 512 "
                     "images, bigger networks and longer training, so this is a reference, not a like-for-like contest.",
                     f"Our models reach {min(x['t']['SeK'] / reach[x['no']] for x in A):.0%}–"
                     f"{max(x['t']['SeK'] / reach[x['no']] for x in A):.0%} of the published score of the same "
                     "design, with a quarter of the pixels, a small encoder and 20 epochs on a laptop. The order of "
                     "the designs matches the papers.", [56, 34, 13, 13, 13, 13, 16])
rows = [[m, "Published (512 × 512 images)", oa, mi, se, fs, "—"] for m, oa, mi, se, fs in LIT]
start = r + 1
for x in A + [best]:
    li = {1: 0, 2: 2, 3: 3}[x["no"]]
    rr = start + len(rows)
    rows.append(["Ours: " + long(x), "Ours (256 × 256 images, 18-layer encoder, 20 epochs)", x["t"]["OA"],
                 x["t"]["mIoU"], x["t"]["SeK"], x["t"]["Fscd"], f"=E{rr}/E{start + li}"])
f7, _, r = table(ws, r, ["Method", "Setting", M_OA + " (%)", M_IOU + " (%)", M_MAIN + " (%)", M_F + " (%)",
                         "Share of the published main score"], rows,
                 fmts=[None, None] + ["0.00"] * 4 + ["0%"], wrap_cols=(1,))
for i in range(len(rows)):
    ws.cell(f7 + i, 7).alignment = Alignment(horizontal="center", vertical="center")
r = note(ws, r, "Published values as listed in Mamba-FCS (arXiv 2508.08232), Table II. 'Share' compares each of our "
         "models with the paper version of the same design (Model 1 ↔ strategy 2, Model 2 ↔ Separated Semantic "
         "Segmentation and Change Detection, Model 3 ↔ Bi-temporal Semantic Reasoning Network).", n)
r = picture(ws, r, X("published.png"), n, 950)

# ============================================================================ Predictions
ws, r, n = new_sheet("Predictions", "What the best model actually predicts",
                     "four test image pairs: the two photos, the true labels, and the best model's prediction. White = "
                     "no change; colours = land-cover class (legend under the picture).",
                     "Large changes such as new buildings are found well; thin or small changed areas are often "
                     "missed, as the 'Errors' sheet shows.", [12] * 12)
r = picture(ws, r, D("qualitative.png"), n, 1000)

# ============================================================================ Glossary
ws, r, n = new_sheet("Glossary", "Glossary: every term and short form",
                     "plain-language meaning of each term, and the short form used for it in research papers.",
                     "Separated Kappa is the main score. For every score in this workbook, higher is better "
                     "(except error rates).", [34, 16, 72])
terms = [
    ("Semantic change detection", "SCD", "Compare two aerial photos of the same place taken at different times; for "
     "every pixel, decide whether it changed and what it was before and after."),
    ("SEmantic Change detectiON Dataset", "SECOND", "Public benchmark: 2,968 aerial image pairs, 6 land-cover classes."),
    ("Time 1 / Time 2", "T1 / T2", "The before photo and the after photo."),
    ("Convolutional neural network", "CNN", "A deep-learning model made for images."),
    ("Residual Network with 18 layers", "ResNet-18", "The image encoder used in all three models."),
    ("Feature Pyramid Network", "FPN", "The decoder style: combines information from several image scales."),
    ("Early Fusion network", "—", "Model 1. Both photos are joined at the input."),
    ("Separated Semantic Segmentation and Change Detection (late fusion)", "SSCD-l", "Model 2. Siamese design."),
    ("Bi-temporal Semantic Reasoning Network", "Bi-SRNet", "Model 3 (we use a lite version)."),
    ("High Resolution Semantic Change Detection", "HRSCD", "Paper/dataset that introduced the early-fusion strategy."),
    ("Separated Kappa", "SeK", "Main score of the dataset: agreement on changed classes with the easy 'no change' "
     "part removed, scaled by how well changes are found. 0 = no better than chance, 100 = perfect."),
    ("F-score for semantic change detection", "Fscd", "F1-score on changed pixels where the class must also be right."),
    ("Intersection over union", "IoU", "Overlap between predicted and true area of a class, 0–100."),
    ("Mean intersection over union", "mIoU", "Average of the 'changed' and 'unchanged' intersection over union."),
    ("Overall accuracy", "OA", "Share of pixels with the correct label."),
    ("Precision / recall / F1-score", "—", "Precision: share of predicted changes that are real. Recall: share of "
     "real changes found. F1-score: their harmonic mean."),
    ("True / false positive / negative", "TP / FP / TN / FN", "The four cells of a yes/no confusion matrix "
     "(positive = changed)."),
    ("Receiver operating characteristic, area under the curve", "ROC-AUC", "Ranking quality over all decision "
     "thresholds. Not computed here (see 'Standard Metrics')."),
    ("Cross-entropy", "CE", "The standard training loss (also called log loss)."),
    ("Binary cross-entropy", "BCE", "Cross-entropy for a yes/no output (changed or not)."),
    ("Weighted cross-entropy", "WCE", "Cross-entropy where rare classes count more."),
    ("Online hard example mining", "OHEM", "Train only on the hardest pixels."),
    ("Adaptive moment estimation with decoupled weight decay", "AdamW", "The optimiser used to train the models."),
    ("Epoch", "—", "One pass over all training images. Every model here trains for 20 epochs."),
    ("Training / validation / test set", "—", "Training: used to learn. Validation: used to choose the best epoch. "
     "Test: used once for the final score."),
    ("Cross-validation (3-fold)", "CV", "Train 3 times, each time holding out a different part of the data."),
    ("Overfitting / underfitting", "—", "Overfitting: much better on training data than on new data. Underfitting: "
     "not good even on training data."),
]
_, _, r = table(ws, r, ["Full term", "Short form in papers", "Meaning"], [list(t) for t in terms], wrap_cols=(2,))

# ============================================================================ References
ws, r, n = new_sheet("References", "Sources of every model and method",
                     "the paper behind each component, with a clickable link. Full details and code locations: "
                     "docs/REFERENCES.md in the repository.",
                     "All models and methods come from published work; this project's contribution is the controlled "
                     "comparison and the error analysis.", [22, 50, 36, 52])
refs = [
    ("Dataset and main score", "SECOND dataset; Separated Kappa", "Yang and others, IEEE TGRS 2022", "https://arxiv.org/abs/2010.05687"),
    ("Model 1", "Early Fusion (High Resolution Semantic Change Detection, strategy 2)", "Daudt and others, 2019", "https://arxiv.org/abs/1810.08452"),
    ("Model 2", "Separated Semantic Segmentation and Change Detection (late fusion)", "Ding and others, 2022", "https://arxiv.org/abs/2108.06103"),
    ("Model 3", "Bi-temporal Semantic Reasoning Network", "Ding and others, 2022", "https://arxiv.org/abs/2108.06103"),
    ("Encoder", "Residual Network", "He and others, 2016", "https://arxiv.org/abs/1512.03385"),
    ("Decoder", "Feature Pyramid Network", "Lin and others, 2017", "https://arxiv.org/abs/1612.03144"),
    ("Attention", "Non-local (attention) blocks", "Wang and others, 2018", "https://arxiv.org/abs/1711.07971"),
    ("Method", "Median frequency balancing", "Eigen and Fergus, 2015", "https://arxiv.org/abs/1411.4734"),
    ("Method", "Class-balanced loss", "Cui and others, 2019", "https://arxiv.org/abs/1901.05555"),
    ("Method", "Focal loss", "Lin and others, 2017", "https://arxiv.org/abs/1708.02002"),
    ("Method", "Dice loss", "Milletari and others, 2016", "https://arxiv.org/abs/1606.04797"),
    ("Method", "Online hard example mining", "Shrivastava and others, 2016", "https://arxiv.org/abs/1604.03540"),
    ("Method", "Rare-class oversampling (repeat-factor sampling)", "Gupta and others, 2019", "https://arxiv.org/abs/1908.03195"),
    ("Training", "Adaptive moment estimation with decoupled weight decay", "Loshchilov and Hutter, 2019", "https://arxiv.org/abs/1711.05101"),
    ("Training", "One-cycle learning-rate schedule", "Smith and Topin, 2017", "https://arxiv.org/abs/1708.07120"),
    ("Validation", "K-fold cross-validation", "Kohavi, 1995", "https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf"),
    ("Comparison", "Semantic Change Network", "Ding and others, 2024", "https://arxiv.org/abs/2212.05245"),
    ("Comparison", "ChangeMamba", "Chen and others, 2024", "https://arxiv.org/abs/2404.03425"),
    ("Comparison", "Mamba-FCS (published results table)", "2025", "https://arxiv.org/abs/2508.08232"),
]
f8, _, r = table(ws, r, ["Used for", "Component", "Paper", "Link"], [list(x_) for x_ in refs], wrap_cols=(1, 2))
for i, x_ in enumerate(refs):
    c = ws.cell(f8 + i, 4)
    link(c, x_[3])
    c.border, c.alignment = BOX, Alignment(vertical="center")

# ============================================================================ Setup
ws, r, n = new_sheet("Setup", "How the experiments were run",
                     "settings shared by every run, and the exact definition of each score.",
                     "Every run used the same data split, random seed, encoder and training schedule, so differences "
                     "between rows come only from the model or method being tested.", [26, 110])
setup = [
    ("Data", "SECOND public training release: 2,968 image pairs."),
    ("Image size", "Reduced from 512 × 512 to 256 × 256 pixels with the Pillow library: photos with Lanczos "
     "resampling (keeps detail), labels with nearest-neighbour (keeps class numbers valid). No model was used for resizing."),
    ("Split", "Random 70 / 10 / 20 percent (seed 0): 2,077 training, 296 validation, 595 test pairs."),
    ("Cross-validation", "Training + validation pairs (2,373) re-split into 3 folds (seed 1); test pairs unchanged."),
    ("Augmentation", "Random horizontal and vertical flips, 90-degree rotations, and swapping the before/after photos."),
    ("Optimiser", "Adaptive moment estimation with decoupled weight decay; learning rate 0.0005 (decoders) and "
     "0.0001 (encoder); weight decay 0.0001."),
    ("Schedule", "One-cycle schedule with 10% warm-up; 20 epochs; batch size 8."),
    ("Standard loss", "Binary cross-entropy for 'changed or not' + 0.5 × cross-entropy for the class (only on changed "
     "pixels); Model 3 also adds a consistency loss."),
    ("Model selection", "Keep the epoch with the best validation score; score it once on the test set."),
    ("Hardware", "Apple M4 laptop with 16 GB memory, using its built-in graphics processor; about 10 minutes "
     "(Model 1) to 22 minutes (Models 2 and 3) per training."),
    ("Software", "Python 3.11, PyTorch 2.14 (full list in requirements.txt)."),
    ("Overall accuracy", "Share of pixels with the correct label (7 labels: 'no change' + 6 classes)."),
    ("Mean intersection over union", "Average of the intersection over union of 'no change' and of 'change'."),
    ("Separated Kappa", "Cohen's kappa over the changed classes (the no-change/no-change cell removed) × "
     "e^(intersection over union of change − 1)."),
    ("F-score for semantic change detection", "Harmonic mean of precision and recall of correct classes on changed pixels."),
    ("Code", "github.com/Chrishabh2002/Second (scd.py, train.py, run_all.sh, run_cv.sh)."),
]
_, _, r = table(ws, r, ["Item", "Value"], [list(s_) for s_ in setup], wrap_cols=(1,))

# ============================================================================ Next steps
ws, r, n = new_sheet("Next Steps", "Next steps towards a publication",
                     "the plan, in order of priority. Full plan: docs/ROADMAP.md in the repository.",
                     "The main missing piece is a new method. The planned one is a transition-aware loss that targets "
                     "the biggest error found here: missed changes on rare types of change.", [6, 64, 56, 22])
steps_ = [
    (1, "Repeat the best methods with 2 more random seeds and a significance test", "The top methods are very close", "About 1 day"),
    (2, "Tune the decision threshold for 'changed' on the validation data", "Missed changes are the largest error", "Hours, no retraining"),
    (3, "Add the receiver operating characteristic curve and its area", "Completes the standard metrics", "Under 1 hour"),
    (4, "Fix or drop online hard example mining (keep 50–70% of pixels)", "Its default setting failed", "2–3 trainings"),
    (5, "Transition-aware loss (new method)", "Weights each pixel by how rare its before → after change is", "1–2 weeks"),
    (6, "Published setting on a graphics server (512 pixels, larger encoder, 50+ epochs)", "Needed to compare with papers fairly", "About 1 week"),
    (7, "Second dataset (Landsat-SCD)", "Shows the result is not specific to one dataset", "About 1 week"),
    (8, "Write the paper", "Venue to be chosen with the supervisor", "2–3 weeks"),
]
_, _, r = table(ws, r, ["#", "Step", "Why", "Time"], [list(s_) for s_ in steps_], wrap_cols=(1, 2))
for i in range(len(steps_)):
    ws.cell(r - 2 - len(steps_) + i, 1).alignment = Alignment(horizontal="center", vertical="center")

# ============================================================================ Start Here (built last, shown first)
t0, t1 = REFS["tech"]
ws, r, n = new_sheet(S_START, "Semantic change detection on the SECOND dataset — start here",
                     "a plain-language summary of the whole project: what was done, which model was used and why, "
                     "how good the results are, and how to read the other sheets.",
                     f"The best model finds and labels changes {(best['t']['SeK'] - m1['t']['SeK']) / m1['t']['SeK']:.0%} "
                     f"better than the starting model (main score {m1['t']['SeK']:.2f} → {best['t']['SeK']:.2f}). "
                     "Most of the gain came from choosing a better model design.", [36, 46, 62])
r = qa(ws, r, "What is this project?",
       "Two aerial photos of the same place, taken at different times, are compared. For every pixel the computer "
       "decides: did it change? If yes, what was it before and what is it now (for example tree → building)? The data "
       "is the public SECOND dataset (2,968 photo pairs, 6 land-cover classes). The hard part: only 20% of pixels change, "
       "and some classes (water, playground) are very rare.", n)
r = heading(ws, r, "The answer in numbers (test set of 595 photo pairs the models never saw)")
k0 = r + 1
kpis = [
    ("Starting model (Model 1) — main score", f"='{S_STEP1}'!B{REFS['m1']}", "0.00", "Separated Kappa, 0–100, higher is better"),
    ("Chosen model (Model 3) — main score", f"='{S_STEP1}'!B{REFS['m3']}", "0.00", "Bi-temporal Semantic Reasoning Network"),
    ("Best result — main score", f"=MAX('{S_STEP2}'!D{t0}:D{t1})", "0.00", "Model 3 + best method for rare classes"),
    ("Best method for rare classes", f"=INDEX('{S_STEP2}'!A{t0}:A{t1},MATCH(B{k0 + 2},'{S_STEP2}'!D{t0}:D{t1},0))",
     None, "In the single main run; in the 3-fold check the top methods are tied"),
    ("Total improvement", f"=(B{k0 + 2}-B{k0})/B{k0}", "0%", "Best result vs starting model"),
    ("Overall accuracy of the best result", best["t"]["OA"] / 100, "0.0%", "Share of pixels labelled correctly"),
    ("Changes found by the best result (recall)", best["c"]["tp"] / (best["c"]["tp"] + best["c"]["fn"]), "0.0%",
     "Share of truly changed pixels that the model marked as changed"),
    ("Methods that beat the standard loss in all 3 folds", f"{len(wins_all)} of 9", None, "Stability check"),
]
_, _, r = table(ws, r, ["Measure", "Value", "Meaning"], [[a, b, d] for a, b, _, d in kpis], wrap_cols=(2,))
for i, (_, _, fm, _) in enumerate(kpis):
    c = ws.cell(k0 + i, 2)
    c.font, c.alignment = F(size=12, bold=True, color=NAVY), Alignment(horizontal="right", vertical="center")
    if fm:
        c.number_format = fm
r = picture(ws, r, X("journey.png"), n, 820)
r = heading(ws, r, "Which machine learning model did we use, and why?")
why = [
    ("1. Built three models", "All are convolutional neural networks (deep learning) with the same 18-layer Residual "
     "Network encoder. Model 1 joins both photos at the start; Model 2 reads them separately (Siamese); Model 3 adds "
     "attention between the two dates.", "To find out which design handles before/after photos best, under exactly "
     "the same conditions."),
    ("2. Picked the best", f"Model 3 had the highest validation score (test main score {base['t']['SeK']:.2f} vs "
     f"{A[1]['t']['SeK']:.2f} and {m1['t']['SeK']:.2f}).", "Chosen on validation data, so the test score stays a fair, "
     "unseen check."),
    ("3. Helped it with rare classes", "Tried 9 methods (weighted losses, Dice loss, focal loss, oversampling, …) on "
     "Model 3.", "Rare classes like water were almost never found by Model 1."),
    ("4. Checked stability", "Repeated every configuration 3 times (3-fold cross-validation).", "A single run can be "
     "lucky; repeating shows which results are real."),
]
_, _, r = table(ws, r, ["Step", "What was done", "Why"], [list(w_) for w_ in why], wrap_cols=(1, 2))
r = qa(ws, r, "Why is the main score only about 15 when accuracy is 84%?",
       f"They measure different things. The main score (Separated Kappa) is NOT a percentage of correct pixels. It is "
       f"a strict score that removes the easy 'no change' pixels and checks whether changes are found AND labelled "
       f"with the right classes; 0 means no better than chance. On this dataset even the best published models "
       f"score only about 24–25. Accuracy is high for every model because {always_no_change:.0%} of pixels never "
       f"change: a model that always answers 'no change' would already get about {always_no_change:.0%} accuracy "
       f"while finding zero changes.", n)
r = qa(ws, r, "Why not just use the usual metrics (accuracy, precision, recall, F1-score)?",
       "They are reported too — see the 'Standard Metrics' sheet. But on their own they are misleading here: accuracy "
       "rewards saying 'no change', and precision/recall/F1 only check whether a change was found, not whether its "
       "land-cover classes are right. Research papers on this dataset therefore use Separated Kappa as the main score, "
       "plus the F-score for semantic change detection, mean intersection over union and overall accuracy. Using the "
       "same scores makes our results comparable with published work.", n)
r = qa(ws, r, "Why are our scores lower than in research papers?",
       "Papers use full-size 512 × 512 images, larger encoders and 50–200 epochs on large graphics servers. This project "
       "used 256 × 256 images (a quarter of the pixels), an 18-layer encoder and 20 epochs on a laptop. The ranking of "
       "the three designs still matches the papers (see 'Comparison with Papers').", n)
r = heading(ws, r, "How to read this workbook (click a sheet name)")
contents = [
    (S_MODELS, "The three models: what they are, which paper, why tried, result"),
    (S_STD, "Accuracy, precision, recall, F1-score and confusion counts for every run"),
    (S_STEP1, "Step 1: the three models compared"),
    (S_STEP2, "Step 2: nine methods for rare classes"),
    ("Results per Class", "Score of each land-cover class"),
    ("Errors", "What kind of mistakes the models make"),
    ("Stability Check", "Every configuration repeated 3 times"),
    ("Learning Curves", "Overfitting or underfitting?"),
    ("Training Progress", "Validation score after every epoch"),
    ("Comparison with Papers", "Our scores next to published papers"),
    ("Predictions", "Example outputs of the best model"),
    ("Glossary", "Every term and short form explained"),
    ("References", "Paper and link for every component"),
    ("Setup", "Settings and exact score definitions"),
    ("Next Steps", "Plan towards a publication"),
]
c0 = r + 1
_, _, r = table(ws, r, ["Sheet", "What you will find"], [[a, b] for a, b in contents])
ws.merge_cells(start_row=c0 - 1, start_column=2, end_row=c0 - 1, end_column=3)
for i, (name, _) in enumerate(contents):
    c = ws.cell(c0 + i, 1)
    link(c, f"#'{name}'!A1", name)
    c.border = BOX
    ws.merge_cells(start_row=c0 + i, start_column=2, end_row=c0 + i, end_column=3)
    ws.row_dimensions[c0 + i].height = 18

# ----------------------------------------------------------------------------- order, save, cache values
order = [S_START, S_MODELS, S_STD, S_STEP1, S_STEP2, "Results per Class", "Errors", "Stability Check",
         "Learning Curves", "Training Progress", "Comparison with Papers", "Predictions", "Glossary",
         "References", "Setup", "Next Steps"]
wb._sheets = [wb[s] for s in order]
for w_ in wb.worksheets:
    w_.sheet_properties.tabColor = {S_START: "0D366B", "Glossary": "1BAF7A", "Next Steps": "EB6834"}.get(w_.title, "6DA7EC")
wb.active = 0
wb.calculation.fullCalcOnLoad = True
if os.environ.get("PREVIEW_SHEET"):  # used only to render a preview of one sheet
    wb.move_sheet(os.environ["PREVIEW_SHEET"], offset=-wb.sheetnames.index(os.environ["PREVIEW_SHEET"]))
    wb.active = 0
    OUT = os.environ["PREVIEW_OUT"]
wb.save(OUT)


def cache_values(path):
    """Store each formula's computed value in the file, so previews that do not recalculate show numbers."""
    import re
    import shutil
    import tempfile
    import zipfile
    from xml.sax.saxutils import escape

    import formulas
    sol = formulas.ExcelModel().loads(path).finish().calculate()
    fname = os.path.basename(path)
    zin = zipfile.ZipFile(path)
    wbx = zin.read("xl/workbook.xml").decode()
    rels = zin.read("xl/_rels/workbook.xml.rels").decode()
    target = {m.group(1): m.group(2) for m in re.finditer(r'Id="(rId\d+)"[^>]*Target="/?(?:xl/)?([^"]+)"', rels)}
    target.update({m.group(2): m.group(1) for m in re.finditer(r'Target="/?(?:xl/)?([^"]+)"[^>]*Id="(rId\d+)"', rels)})
    sheets = {"xl/" + target[m.group(2)]: m.group(1)
              for m in re.finditer(r'<sheet name="([^"]+)"[^>]*r:id="(rId\d+)"', wbx)}
    n_ok, bad = 0, []

    def fill(xml, title):
        def rep(m):
            nonlocal n_ok
            attrs, ref, f = m.group(1), m.group(2), m.group(3)
            v = sol.get(f"'[{fname}]{title.upper()}'!{ref}")
            v = v.value[0, 0] if v is not None and hasattr(v, "value") else v
            if v is None or isinstance(v, formulas.tokens.operand.XlError):
                bad.append(f"{title}!{ref}")
                return m.group(0)
            n_ok += 1
            if isinstance(v, str):
                return f'<c{attrs} t="str"><f>{f}</f><v>{escape(v)}</v></c>'
            if isinstance(v, (bool, np.bool_)):
                return f'<c{attrs} t="b"><f>{f}</f><v>{int(v)}</v></c>'
            return f"<c{attrs}><f>{f}</f><v>{float(v)!r}</v></c>"
        return re.sub(r'<c( r="([A-Z]+\d+)"[^>]*?)><f>(.*?)</f><v\s*/></c>', rep, xml)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in sheets:
                data = fill(data.decode(), sheets[item.filename]).encode()
            zout.writestr(item, data)
    zin.close()
    shutil.move(tmp, path)
    if bad:
        raise SystemExit(f"formula errors in: {bad[:20]}")
    return n_ok


print(f"cached {cache_values(OUT)} formula values")
print(f"wrote {OUT}  ({len(wb.worksheets)} sheets, cross-validation runs {cv_done}/{K * len(CV_ORDER)})")
