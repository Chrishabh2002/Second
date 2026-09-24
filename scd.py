"""Semantic change detection on SECOND: data, models, losses, metrics."""
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
NUM_SEM = 6  # land-cover classes 1..6 (0 = no change)
CLASS_NAMES = ["no-change", "non-veg ground", "tree", "low vegetation", "water",
               "building", "playground"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- data
class LazyPNG:
    """Array-like view of one sub-folder; PNGs are decoded on indexing to keep RAM low
    (holding the whole set in memory pushed this 16 GB machine into swap)."""

    def __init__(self, d, names):
        self.d, self.names = d, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if np.isscalar(idx):
            return np.asarray(Image.open(os.path.join(self.d, self.names[idx])))
        return np.stack([self[int(i)] for i in idx])


def load_all(size=256):
    d = os.path.join(ROOT, "data", f"SECOND_{size}")
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "label1", "*.png")))
    return (names,) + tuple(LazyPNG(os.path.join(d, s), names)
                            for s in ("im1", "im2", "label1", "label2"))


def split_indices(n, seed=0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.1 * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def to_batch(im1, im2, l1, l2, idx, device, augment=False):
    x1 = torch.from_numpy(im1[idx]).permute(0, 3, 1, 2).float() / 255.
    x2 = torch.from_numpy(im2[idx]).permute(0, 3, 1, 2).float() / 255.
    y1 = torch.from_numpy(l1[idx]).long()
    y2 = torch.from_numpy(l2[idx]).long()
    if augment:  # same random flip / rot90 for the whole batch (cheap, applied to all tensors)
        if np.random.rand() < 0.5:
            x1, x2, y1, y2 = [t.flip(-1) for t in (x1, x2, y1, y2)]
        if np.random.rand() < 0.5:
            x1, x2, y1, y2 = [t.flip(-2) for t in (x1, x2, y1, y2)]
        k = np.random.randint(4)
        if k:
            x1, x2, y1, y2 = [t.rot90(k, (-2, -1)) for t in (x1, x2, y1, y2)]
        if np.random.rand() < 0.5:  # swap temporal order
            x1, x2, y1, y2 = x2, x1, y2, y1
    x1 = (x1 - MEAN) / STD
    x2 = (x2 - MEAN) / STD
    return [t.contiguous().to(device) for t in (x1, x2, y1, y2)]


# --------------------------------------------------------------------------- models
class Encoder(nn.Module):
    """ImageNet-pretrained ResNet-18; returns features at strides 4, 8, 16, 32."""

    def __init__(self, in_ch=3):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        if in_ch != 3:
            w = r.conv1.weight.data
            r.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            r.conv1.weight.data = torch.cat([w] * (in_ch // 3), 1) / (in_ch // 3)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layers = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


def cbr(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o),
                         nn.ReLU(inplace=True))


class Decoder(nn.Module):
    """FPN-style decoder: fuse 4 scales -> 64 ch at stride 4."""

    def __init__(self, chans=(64, 128, 256, 512), mid=64):
        super().__init__()
        self.lat = nn.ModuleList([cbr(c, mid, 1) for c in chans])
        self.smooth = nn.ModuleList([cbr(mid, mid) for _ in chans[:-1]])

    def forward(self, feats):
        x = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.smooth[i](x + self.lat[i](feats[i]))
        return x


class EarlyFusion(nn.Module):
    """HRSCD str.2-like: stack both images as 6 channels, one network, three heads."""

    def __init__(self):
        super().__init__()
        self.enc = Encoder(6)
        self.dec = Decoder()
        self.head_c = nn.Conv2d(64, 1, 1)
        self.head_s1 = nn.Conv2d(64, NUM_SEM, 1)
        self.head_s2 = nn.Conv2d(64, NUM_SEM, 1)

    def forward(self, x1, x2):
        f = self.dec(self.enc(torch.cat([x1, x2], 1)))
        up = lambda t: F.interpolate(t, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        return up(self.head_c(f)), up(self.head_s1(f)), up(self.head_s2(f)), None


class CrossTemporalAttn(nn.Module):
    """Bi-SRNet Cot-SR style: each date's deepest feature attends to the other date."""

    def __init__(self, c=512, r=8):
        super().__init__()
        self.q = nn.Conv2d(c, c // r, 1)
        self.k = nn.Conv2d(c, c // r, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.g = nn.Parameter(torch.zeros(1))

    def attend(self, a, b):
        B, C, H, W = a.shape
        q = self.q(a).flatten(2).transpose(1, 2)
        k = self.k(b).flatten(2)
        v = self.v(b).flatten(2)
        att = torch.softmax(q @ k / math.sqrt(k.shape[1]), -1)
        return a + self.g * (v @ att.transpose(1, 2)).view(B, C, H, W)

    def forward(self, a, b):
        return self.attend(a, b), self.attend(b, a)


class SSCD(nn.Module):
    """Siamese SCD (Ding et al., SSCD-l). bisrnet=True adds cross-temporal attention
    (Bi-SRNet reasoning block); semantic consistency loss is applied in the loss."""

    def __init__(self, bisrnet=False):
        super().__init__()
        self.enc = Encoder(3)
        self.sem_dec = Decoder()
        self.cd_dec = Decoder(chans=(128, 256, 512, 1024))
        self.head_s = nn.Conv2d(64, NUM_SEM, 1)
        self.head_c = nn.Conv2d(64, 1, 1)
        self.cot = CrossTemporalAttn() if bisrnet else None

    def forward(self, x1, x2):
        f1, f2 = self.enc(x1), self.enc(x2)
        if self.cot is not None:
            f1[-1], f2[-1] = self.cot(f1[-1], f2[-1])
        s1, s2 = self.sem_dec(f1), self.sem_dec(f2)
        c = self.cd_dec([torch.cat([a, b], 1) for a, b in zip(f1, f2)])
        up = lambda t: F.interpolate(t, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        return up(self.head_c(c)), up(self.head_s(s1)), up(self.head_s(s2)), (s1, s2)


def build_model(name):
    return {"early_fusion": lambda: EarlyFusion(),
            "sscd": lambda: SSCD(False),
            "bisrnet": lambda: SSCD(True)}[name]()


# --------------------------------------------------------------------------- losses
def class_stats(l1, l2, idx):
    """Pixel counts of each semantic class (1..6) over changed pixels, plus change ratio."""
    counts = np.zeros(NUM_SEM + 1, np.int64)
    for lab in (l1[idx], l2[idx]):
        counts += np.bincount(lab.ravel(), minlength=NUM_SEM + 1)
    change_ratio = (l1[idx] > 0).mean()
    return counts, change_ratio


def sem_weights(counts, kind):
    c = counts[1:].astype(np.float64)
    freq = c / c.sum()
    if kind == "inv_sqrt":      # inverse square-root frequency
        w = 1 / np.sqrt(freq)
    elif kind == "median":      # median-frequency balancing (Eigen & Fergus)
        w = np.median(freq) / freq
    elif kind == "effective":   # class-balanced loss, effective number (Cui et al. 2019)
        beta = 1 - 1 / c.mean()
        w = (1 - beta) / (1 - np.power(beta, c))
    else:
        w = np.ones_like(c)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def focal_ce(logits, target, gamma=2.0, weight=None, ignore_index=-1):
    logp = F.log_softmax(logits, 1)
    ce = F.nll_loss(logp, target, weight=weight, ignore_index=ignore_index, reduction="none")
    valid = target != ignore_index
    pt = torch.exp(-F.nll_loss(logp, target.clamp(min=0), reduction="none"))
    loss = ((1 - pt) ** gamma * ce)[valid]
    return loss.mean() if loss.numel() else logits.sum() * 0


def focal_bce(logit, target, gamma=2.0, alpha=None):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    loss = (1 - pt) ** gamma * ce
    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))
    return loss.mean()


def dice_multiclass(logits, target, ignore_index=-1, eps=1.0):
    valid = (target != ignore_index).unsqueeze(1).float()
    prob = torch.softmax(logits, 1) * valid
    oh = F.one_hot(target.clamp(min=0), logits.shape[1]).permute(0, 3, 1, 2).float() * valid
    inter = (prob * oh).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def dice_binary(logit, target, eps=1.0):
    p = torch.sigmoid(logit)
    return 1 - (2 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)


def ohem_ce(logits, target, ignore_index=-1, keep=0.25, weight=None):
    ce = F.cross_entropy(logits, target, weight=weight, ignore_index=ignore_index,
                         reduction="none")[target != ignore_index]
    if ce.numel() == 0:
        return logits.sum() * 0
    k = max(1, int(keep * ce.numel()))
    return ce.topk(k).values.mean()


def ohem_bce(logit, target, keep=0.25):
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none").flatten()
    return ce.topk(max(1, int(keep * ce.numel()))).values.mean()


class SCDLoss(nn.Module):
    """Change loss (binary) + semantic loss on changed pixels (+ optional consistency)."""

    def __init__(self, kind, counts, change_ratio, consistency=False):
        super().__init__()
        self.kind = kind
        self.consistency = consistency
        wkind = {"wce": "inv_sqrt", "median": "median", "cb": "effective",
                 "combo": "inv_sqrt"}.get(kind)
        self.register_buffer("w", sem_weights(counts, wkind) if wkind else torch.ones(NUM_SEM))
        self.pos_weight = torch.tensor([float((1 - change_ratio) / change_ratio)]).sqrt() \
            if wkind else None

    def forward(self, out, y1, y2):
        c_logit, s1, s2, feats = out
        c_logit = c_logit.squeeze(1)
        change = (y1 > 0).float()
        t1, t2 = y1 - 1, y2 - 1  # -1 -> ignored (no-change pixels)
        k, w = self.kind, self.w.to(s1.device)
        if k in ("ce",):
            lc = F.binary_cross_entropy_with_logits(c_logit, change)
            ls = sum(F.cross_entropy(s, t, ignore_index=-1) for s, t in ((s1, t1), (s2, t2)))
        elif k in ("wce", "median", "cb"):
            lc = F.binary_cross_entropy_with_logits(
                c_logit, change, pos_weight=self.pos_weight.to(c_logit.device))
            ls = sum(F.cross_entropy(s, t, weight=w, ignore_index=-1) for s, t in ((s1, t1), (s2, t2)))
        elif k == "focal":
            lc = focal_bce(c_logit, change, 2.0, alpha=0.75)
            ls = sum(focal_ce(s, t, 2.0) for s, t in ((s1, t1), (s2, t2)))
        elif k == "dice":
            lc = F.binary_cross_entropy_with_logits(c_logit, change) + dice_binary(c_logit, change)
            ls = sum(F.cross_entropy(s, t, ignore_index=-1) + dice_multiclass(s, t)
                     for s, t in ((s1, t1), (s2, t2)))
        elif k == "ohem":
            lc = ohem_bce(c_logit, change)
            ls = sum(ohem_ce(s, t) for s, t in ((s1, t1), (s2, t2)))
        elif k == "combo":  # weighted CE + Dice
            lc = F.binary_cross_entropy_with_logits(
                c_logit, change, pos_weight=self.pos_weight.to(c_logit.device)) \
                + dice_binary(c_logit, change)
            ls = sum(F.cross_entropy(s, t, weight=w, ignore_index=-1) + dice_multiclass(s, t)
                     for s, t in ((s1, t1), (s2, t2)))
        else:
            raise ValueError(k)
        loss = lc + 0.5 * ls
        if self.consistency and feats is not None:
            loss = loss + semantic_consistency_loss(s1, s2, change)
        return loss


def semantic_consistency_loss(s1, s2, change):
    """Bi-SRNet: semantic predictions should agree on unchanged pixels and differ on changed."""
    p1, p2 = torch.softmax(s1, 1), torch.softmax(s2, 1)
    cos = F.cosine_similarity(p1, p2, dim=1)
    return ((1 - cos) * (1 - change) + cos * change).mean()


# --------------------------------------------------------------------------- metrics
def fast_hist(pred, label, n=NUM_SEM + 1):
    k = (label >= 0) & (label < n)
    return np.bincount(n * pred[k].astype(int) + label[k], minlength=n * n).reshape(n, n)


def cal_kappa(hist):
    if hist.sum() == 0:
        return 0.0
    po = np.diag(hist).sum() / hist.sum()
    pe = (hist.sum(0) * hist.sum(1)).sum() / hist.sum() ** 2
    return 0.0 if pe == 1 else (po - pe) / (1 - pe)


def scd_metrics(hist):
    """Official SECOND metrics (Yang et al.; same code as Bi-SRNet / SCanNet).
    hist rows = prediction, cols = ground truth, class 0 = no-change."""
    hist = hist.astype(np.float64)
    hist_fg = hist[1:, 1:]
    c2hist = np.array([[hist[0, 0], hist[0, 1:].sum()], [hist[1:, 0].sum(), hist_fg.sum()]])
    hist_n0 = hist.copy()
    hist_n0[0, 0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    sek = kappa_n0 * math.exp(iu[1]) / math.e
    pixel_sum = hist.sum()
    change_pred = pixel_sum - hist.sum(1)[0]
    change_label = pixel_sum - hist.sum(0)[0]
    tp = np.diag(hist_fg).sum()
    prec, rec = tp / max(change_pred, 1), tp / max(change_label, 1)
    fscd = stats.hmean([prec, rec]) if prec > 0 and rec > 0 else 0.0
    oa = np.diag(hist).sum() / pixel_sum
    per_class_iou = np.diag(hist) / (hist.sum(0) + hist.sum(1) - np.diag(hist) + 1e-12)
    return {"OA": oa * 100, "mIoU": (iu[0] + iu[1]) / 2 * 100, "SeK": sek * 100,
            "Fscd": fscd * 100, "IoU_nc": iu[0] * 100, "IoU_c": iu[1] * 100,
            "per_class_IoU": (per_class_iou * 100).round(2).tolist()}


@torch.no_grad()
def evaluate(model, data, idx, device, bs=16):
    im1, im2, l1, l2 = data
    model.eval()
    hist = np.zeros((NUM_SEM + 1, NUM_SEM + 1), np.int64)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x1, x2, y1, y2 = to_batch(im1, im2, l1, l2, b, device)
        c, s1, s2, _ = model(x1, x2)
        change = (torch.sigmoid(c.squeeze(1)) > 0.5).long()
        p1 = (s1.argmax(1) + 1) * change
        p2 = (s2.argmax(1) + 1) * change
        for p, y in ((p1, y1), (p2, y2)):
            hist += fast_hist(p.cpu().numpy().ravel(), y.cpu().numpy().ravel())
    return scd_metrics(hist), hist
