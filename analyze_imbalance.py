"""Quantify class imbalance in SECOND (labels only). Writes results/imbalance_stats.json."""
import glob, json, os
import numpy as np
from PIL import Image
import scd

d = os.path.join(scd.ROOT, "data", "SECOND_256")
names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "label1", "*.png")))
l1 = np.stack([np.asarray(Image.open(os.path.join(d, "label1", n))) for n in names])
l2 = np.stack([np.asarray(Image.open(os.path.join(d, "label2", n))) for n in names])
tot = l1.size
change = (l1 > 0)
out = {"n_pairs": len(names), "change_pixel_ratio": float(change.mean())}
for t, lab in (("T1", l1), ("T2", l2)):
    c = np.bincount(lab.ravel(), minlength=7)
    out[f"{t}_class_pixel_pct_of_all"] = {scd.CLASS_NAMES[i]: round(100 * c[i] / tot, 3) for i in range(7)}
    out[f"{t}_class_pct_of_changed"] = {scd.CLASS_NAMES[i]: round(100 * c[i] / c[1:].sum(), 2) for i in range(1, 7)}
    out[f"{t}_images_containing_class"] = {scd.CLASS_NAMES[i]: int(((lab == i).reshape(len(names), -1).any(1)).sum()) for i in range(1, 7)}
# from->to transitions (30 change types)
tr = np.bincount((l1[change].astype(int) * 7 + l2[change]), minlength=49).reshape(7, 7)[1:, 1:]
pairs = {f"{scd.CLASS_NAMES[i+1]} -> {scd.CLASS_NAMES[j+1]}": int(tr[i, j]) for i in range(6) for j in range(6) if tr[i, j]}
s = sum(pairs.values())
out["transitions_pct"] = {k: round(100 * v / s, 3) for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])}
c_all = np.bincount(np.concatenate([l1.ravel(), l2.ravel()]), minlength=7)[1:]
out["semantic_imbalance_ratio_max_over_min"] = round(float(c_all.max() / c_all.min()), 1)
out["change_vs_nochange_ratio"] = round(float((~change).sum() / change.sum()), 2)
per_img = change.reshape(len(names), -1).mean(1)
out["per_image_change_ratio"] = {"min": float(per_img.min()), "median": float(np.median(per_img)), "max": float(per_img.max())}
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/imbalance_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))
