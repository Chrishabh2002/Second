"""Collect results/*.json into results/RESULTS.md, results/summary.csv and plots."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scd

R = os.path.join(scd.ROOT, "results")
runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, "[AB]_*.json")))]
imb = json.load(open(os.path.join(R, "imbalance_stats.json")))

LOSS_DESC = {
    "ce": "Plain cross-entropy (no balancing)",
    "wce": "Weighted CE (inverse-sqrt freq) + BCE pos_weight",
    "median": "Median-frequency balancing + BCE pos_weight",
    "cb": "Class-balanced loss (effective number, Cui 2019)",
    "focal": "Focal loss (gamma=2, alpha=0.75 for change)",
    "dice": "CE + Dice loss",
    "ohem": "OHEM (top-25% hardest pixels)",
    "combo": "Weighted CE + Dice",
}


def row(r):
    t, a = r["test"], r["args"]
    val_sek = max(h["SeK"] for h in r["history"])
    tech = LOSS_DESC[a["loss"]] + (" + rare-class oversampling" if a["sampler"] == "rare" else "")
    return dict(tag=r["tag"], model=a["model"], tech=tech, val_sek=val_sek, ep=r["best_epoch"],
                mins=r["train_minutes"], **{k: t[k] for k in ("OA", "mIoU", "SeK", "Fscd", "IoU_c")},
                pc=t["per_class_IoU"])


rows = [row(r) for r in runs]
A = [x for x in rows if x["tag"].startswith("A_")]
B = [x for x in rows if x["tag"].startswith("B_")]
best_arch = open(os.path.join(R, "best_arch.txt")).read().strip().split()[-1] \
    if os.path.exists(os.path.join(R, "best_arch.txt")) else None
base = next((x for x in A if x["model"] == best_arch), None)
B_all = ([base] if base else []) + sorted(B, key=lambda x: -x["SeK"])

hdr = "| Run | Model | Technique | OA | mIoU | **SeK** | Fscd | IoU(change) | val SeK | best ep | min |\n" \
      "|---|---|---|---|---|---|---|---|---|---|---|\n"
fmt = lambda x: (f"| {x['tag']} | {x['model']} | {x['tech']} | {x['OA']:.2f} | {x['mIoU']:.2f} | "
                 f"**{x['SeK']:.2f}** | {x['Fscd']:.2f} | {x['IoU_c']:.2f} | {x['val_sek']:.2f} | "
                 f"{x['ep']} | {x['mins']} |\n")
pc_hdr = "| Run | " + " | ".join(scd.CLASS_NAMES) + " |\n|---|" + "---|" * 7 + "\n"
pc_fmt = lambda x: f"| {x['tag']} | " + " | ".join(f"{v:.1f}" for v in x["pc"]) + " |\n"

md = ["# SECOND semantic change detection — experiment log\n",
      "All numbers are on the held-out **test split (595 pairs)**; model checkpoint chosen by "
      "best validation SeK. Metrics follow the official SECOND definitions (SeK is the primary "
      "metric).\n",
      "\n## 1. Class imbalance in SECOND\n",
      f"- Pairs: {imb['n_pairs']}; changed pixels: {imb['change_pixel_ratio']*100:.1f}% "
      f"(no-change : change = {imb['change_vs_nochange_ratio']} : 1)\n",
      f"- Largest / smallest semantic class (changed pixels, both dates): "
      f"{imb['semantic_imbalance_ratio_max_over_min']}x\n",
      "- Share of changed pixels per class (T1 / T2): " + ", ".join(
          f"{k} {imb['T1_class_pct_of_changed'][k]}% / {imb['T2_class_pct_of_changed'][k]}%"
          for k in imb["T1_class_pct_of_changed"]) + "\n",
      "- Top transitions: " + ", ".join(f"{k} {v}%" for k, v in
                                        list(imb["transitions_pct"].items())[:5]) + "\n",
      "- Rarest transitions: " + ", ".join(f"{k} {v}%" for k, v in
                                           list(imb["transitions_pct"].items())[-5:]) + "\n",
      "\n## 2. Architecture comparison (plain CE)\n", hdr] + [fmt(x) for x in sorted(A, key=lambda x: -x["SeK"])]
if best_arch:
    md += [f"\nSelected for phase 2 (best validation SeK): **{best_arch}**\n"]
md += ["\n## 3. Imbalance techniques (on the selected architecture)\n", hdr] + [fmt(x) for x in B_all]
md += ["\n## 4. Per-class IoU on test (%)\n", pc_hdr] + [pc_fmt(x) for x in A + B_all[1:]]
md = "".join(md)
open(os.path.join(R, "RESULTS.md"), "w").write(md)

with open(os.path.join(R, "summary.csv"), "w") as f:
    f.write("tag,model,technique,OA,mIoU,SeK,Fscd,IoU_change,val_SeK,best_epoch,train_min\n")
    for x in A + B:
        f.write(f"{x['tag']},{x['model']},\"{x['tech']}\",{x['OA']:.2f},{x['mIoU']:.2f},"
                f"{x['SeK']:.2f},{x['Fscd']:.2f},{x['IoU_c']:.2f},{x['val_sek']:.2f},{x['ep']},{x['mins']}\n")

# plot: SeK / Fscd per run
if rows:
    order = sorted(A, key=lambda x: -x["SeK"]) + sorted(B, key=lambda x: -x["SeK"])
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(order) + 1.5))
    y = range(len(order))
    ax.barh([i + 0.2 for i in y], [x["SeK"] for x in order], 0.4, label="SeK", color="#2a6fdb")
    ax.barh([i - 0.2 for i in y], [x["Fscd"] / 3 for x in order], 0.4, label="Fscd / 3",
            color="#9bbcf0")
    ax.set_yticks(list(y), [x["tag"] for x in order])
    ax.invert_yaxis()
    ax.set_xlabel("test score (%)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "sek_comparison.png"), dpi=130)
print(md)
