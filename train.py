"""Train + evaluate one SCD configuration on SECOND and record results.

Example:  python train.py --model sscd --loss ce --sampler uniform --tag sscd_ce
Every run writes results/<tag>.json and appends one row to results/all_results.csv.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import scd

p = argparse.ArgumentParser()
p.add_argument("--model", default="sscd", choices=["early_fusion", "sscd", "bisrnet"])
p.add_argument("--loss", default="ce",
               choices=["ce", "wce", "median", "cb", "focal", "dice", "ohem", "combo"])
p.add_argument("--sampler", default="uniform", choices=["uniform", "rare"])
p.add_argument("--consistency", type=int, default=-1, help="-1: on only for bisrnet")
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", required=True)
args = p.parse_args()
consistency = (args.model == "bisrnet") if args.consistency < 0 else bool(args.consistency)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs(os.path.join(scd.ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(scd.ROOT, "checkpoints"), exist_ok=True)

names, im1, im2, l1, l2 = scd.load_all()
tr, va, te = scd.split_indices(len(names), seed=0)  # split fixed across all runs
counts, change_ratio = scd.class_stats(l1, l2, tr)
print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}  change ratio = {change_ratio:.3f}")

# Rare-class oversampling: image weight = max over present classes of (1/sqrt(class freq))
if args.sampler == "rare":
    freq = counts[1:] / counts[1:].sum()
    cls_w = 1 / np.sqrt(freq)
    img_w = np.ones(len(tr))
    for j, i in enumerate(tr):
        present = np.unique(np.concatenate([np.unique(l1[i]), np.unique(l2[i])]))
        present = present[present > 0]
        if len(present):
            img_w[j] = cls_w[present - 1].max()
    img_w = img_w / img_w.sum()

model = scd.build_model(args.model).to(device)
crit = scd.SCDLoss(args.loss, counts, change_ratio, consistency).to(device)
enc_params = list(model.enc.parameters())
enc_ids = {id(q) for q in enc_params}
opt = torch.optim.AdamW([
    {"params": enc_params, "lr": args.lr * 0.2},
    {"params": [q for q in model.parameters() if id(q) not in enc_ids], "lr": args.lr}],
    weight_decay=1e-4)
iters_per_epoch = len(tr) // args.bs
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[args.lr * 0.2, args.lr], total_steps=args.epochs * iters_per_epoch,
    pct_start=0.1)

data = (im1, im2, l1, l2)
best, best_ep, history = -1, -1, []
ckpt = os.path.join(scd.ROOT, "checkpoints", f"{args.tag}.pt")
t0 = time.time()
for ep in range(args.epochs):
    model.train()
    order = (np.random.choice(tr, len(tr), p=img_w) if args.sampler == "rare"
             else np.random.permutation(tr))
    tot = 0.0
    for it in range(iters_per_epoch):
        b = np.sort(order[it * args.bs:(it + 1) * args.bs])
        x1, x2, y1, y2 = scd.to_batch(im1, im2, l1, l2, b, device, augment=True)
        loss = crit(model(x1, x2), y1, y2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
    m, _ = scd.evaluate(model, data, va, device)
    history.append({"epoch": ep + 1, "loss": tot / iters_per_epoch,
                    **{k: v for k, v in m.items() if k != "per_class_IoU"}})
    if m["SeK"] > best:
        best, best_ep = m["SeK"], ep + 1
        torch.save(model.state_dict(), ckpt)
    print(f"[{args.tag}] ep {ep + 1:02d} loss {tot / iters_per_epoch:.4f} | val "
          f"OA {m['OA']:.2f} mIoU {m['mIoU']:.2f} SeK {m['SeK']:.2f} Fscd {m['Fscd']:.2f} "
          f"| {(time.time() - t0) / 60:.1f} min", flush=True)

model.load_state_dict(torch.load(ckpt, map_location=device))
test, hist = scd.evaluate(model, data, te, device)
train_min = (time.time() - t0) / 60
print(f"[{args.tag}] TEST (best val epoch {best_ep}): OA {test['OA']:.2f} mIoU {test['mIoU']:.2f} "
      f"SeK {test['SeK']:.2f} Fscd {test['Fscd']:.2f}")
print("per-class IoU:", dict(zip(scd.CLASS_NAMES, test["per_class_IoU"])))

rec = {"tag": args.tag, "args": vars(args), "consistency": consistency, "best_epoch": best_ep,
       "train_minutes": round(train_min, 1), "test": test, "confusion": hist.tolist(),
       "history": history}
with open(os.path.join(scd.ROOT, "results", f"{args.tag}.json"), "w") as f:
    json.dump(rec, f, indent=1)
csv_path = os.path.join(scd.ROOT, "results", "all_results.csv")
new = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["tag", "model", "loss", "sampler", "consistency", "seed", "epochs",
                    "best_epoch", "OA", "mIoU", "SeK", "Fscd", "IoU_change"]
                   + [f"IoU_{c}" for c in scd.CLASS_NAMES] + ["train_min"])
    w.writerow([args.tag, args.model, args.loss, args.sampler, consistency, args.seed,
                args.epochs, best_ep] + [round(test[k], 2) for k in
                                         ("OA", "mIoU", "SeK", "Fscd", "IoU_c")]
               + test["per_class_IoU"] + [round(train_min, 1)])
