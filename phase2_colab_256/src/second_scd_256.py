# Exported from the matching notebook in notebooks/. Run the notebook on Colab/Kaggle (GPU); this file is for reading and diffing.
import os, re, sys, json, math, subprocess
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from PIL import Image

EPOCHS, BATCH, LR, SIZE = 20, 8, 3e-4, 256
torch.manual_seed(0); np.random.seed(0)
device = torch.device('cuda')

from google.colab import drive
drive.mount('/content/drive')
WORK = Path('/content/drive/MyDrive/SECOND_SCD')
(WORK / 'results').mkdir(parents=True, exist_ok=True)

FULL_FORM = {
    'SCD': 'Semantic Change Detection',
    'SECOND': 'SEmantic Change detectiON Dataset',
    'SSCD': 'Siamese Semantic Change Detection',
    'Bi-SRNet': 'Bi-temporal Semantic Reasoning Network',
    'ResNet-18': 'Residual Network with 18 layers',
    'FPN': 'Feature Pyramid Network',
    'CE': 'Cross-Entropy loss',
    'WCE': 'Weighted Cross-Entropy loss',
    'SeK': 'Separated Kappa',
    'Fscd': 'F1-score for Semantic Change Detection',
    'mIoU': 'mean Intersection over Union',
    'IoU': 'Intersection over Union',
    'OA': 'Overall Accuracy',
    'F1': 'F1-score (harmonic mean of Precision and Recall)',
    'T1 / T2': 'Image at Time 1 / Time 2',
}
display(pd.DataFrame(FULL_FORM.items(), columns=['Short form', 'Full form']))

CLASSES = ['no change', 'non-vegetated ground', 'tree', 'low vegetation', 'water', 'building', 'playground']
COLOURS = np.array([[255, 255, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [0, 0, 255], [128, 0, 0], [255, 0, 0]], np.uint8)
DATA_FILE = WORK / 'SECOND_256.npz'


def to_class(rgb):
    out = np.zeros(rgb.shape[:2], np.uint8)
    for k, c in enumerate(COLOURS):
        out[(rgb == c).all(-1)] = k
    return out


if not DATA_FILE.exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
    import gdown
    gdown.download(id='1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc', output='/content/second.rar', quiet=False)
    subprocess.run('apt-get -qq install -y libarchive-tools > /dev/null && mkdir -p /content/raw && bsdtar -xf /content/second.rar -C /content/raw', shell=True, check=True)
    root = next(Path('/content/raw').rglob('im1')).parent
    names = sorted(p.name for p in (root / 'label1').glob('*.png'))
    load_img = lambda f, n: np.asarray(Image.open(root / f / n).convert('RGB').resize((SIZE, SIZE), Image.LANCZOS))
    load_lab = lambda f, n: np.asarray(Image.fromarray(to_class(np.asarray(Image.open(root / f / n).convert('RGB')))).resize((SIZE, SIZE), Image.NEAREST))
    np.savez(DATA_FILE,
             img1=np.stack([load_img('im1', n) for n in names]), img2=np.stack([load_img('im2', n) for n in names]),
             lab1=np.stack([load_lab('label1', n) for n in names]), lab2=np.stack([load_lab('label2', n) for n in names]))

d = np.load(DATA_FILE)
IMG1, IMG2, LAB1, LAB2 = d['img1'], d['img2'], d['lab1'], d['lab2']
N = len(IMG1)
perm = np.random.RandomState(0).permutation(N)
TRAIN, VAL, TEST = perm[:int(0.7 * N)], perm[int(0.7 * N):int(0.8 * N)], perm[int(0.8 * N):]

pixels = np.bincount(np.concatenate([LAB1.ravel(), LAB2.ravel()]), minlength=7)
print(pd.Series({'Image pairs': N, 'Image size': f'{SIZE} x {SIZE}', 'Train / Validation / Test pairs': f'{len(TRAIN)} / {len(VAL)} / {len(TEST)}',
                 'Changed pixels (%)': round(100 * (LAB1 > 0).mean(), 2)}).to_string())

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(CLASSES[1:], 100 * pixels[1:] / pixels[1:].sum(), color='#0f766e')
ax[0].set(title='Share of each land-cover class in changed pixels', ylabel='%')
ax[0].tick_params(axis='x', rotation=25)
ax[1].pie([1 - (LAB1 > 0).mean(), (LAB1 > 0).mean()], labels=['no change', 'change'], autopct='%1.1f%%', colors=['#cbd5e1', '#ef4444'])
ax[1].set_title('Changed vs unchanged pixels')
plt.tight_layout(); plt.show()

G = [torch.from_numpy(a).to(device) for a in (IMG1, IMG2, LAB1, LAB2)]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def batch(ids, augment=False):
    i = torch.as_tensor(np.sort(ids), device=device)
    x1, x2 = [(G[k][i].permute(0, 3, 1, 2).float() / 255 - MEAN) / STD for k in (0, 1)]
    y1, y2 = G[2][i].long(), G[3][i].long()
    if augment:
        if np.random.rand() < 0.5:
            x1, x2, y1, y2 = [t.flip(-1) for t in (x1, x2, y1, y2)]
        k = np.random.randint(4)
        x1, x2, y1, y2 = [t.rot90(k, (-2, -1)).contiguous() for t in (x1, x2, y1, y2)]
    return x1, x2, y1, y2

def resnet18(in_ch=3):
    r = torchvision.models.resnet18(weights='DEFAULT')
    if in_ch == 6:
        w = r.conv1.weight.data
        r.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        r.conv1.weight.data = torch.cat([w, w], 1) / 2
    return nn.ModuleList([nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1), r.layer2, r.layer3, r.layer4])


def encode(layers, x):
    feats = []
    for layer in layers:
        x = layer(x)
        feats.append(x)
    return feats


class FPN(nn.Module):
    def __init__(self, chans=(64, 128, 256, 512)):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in chans])
        self.out = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())

    def forward(self, feats):
        x = self.lat[-1](feats[-1])
        for i in (2, 1, 0):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode='bilinear') + self.lat[i](feats[i])
        return self.out(x)


def up(x):
    return F.interpolate(x, size=(SIZE, SIZE), mode='bilinear')


class EarlyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = resnet18(6), FPN()
        self.change, self.sem1, self.sem2 = nn.Conv2d(64, 1, 1), nn.Conv2d(64, 6, 1), nn.Conv2d(64, 6, 1)

    def forward(self, x1, x2):
        f = self.dec(encode(self.enc, torch.cat([x1, x2], 1)))
        return up(self.change(f)), up(self.sem1(f)), up(self.sem2(f))


class Siamese(nn.Module):
    def __init__(self, attention=False):
        super().__init__()
        self.enc, self.sem_dec, self.change_dec = resnet18(3), FPN(), FPN((128, 256, 512, 1024))
        self.sem, self.change = nn.Conv2d(64, 6, 1), nn.Conv2d(64, 1, 1)
        self.attention = nn.MultiheadAttention(512, 4, batch_first=True) if attention else None

    def cross(self, a, b):
        B, C, H, W = a.shape
        q, kv = a.flatten(2).transpose(1, 2), b.flatten(2).transpose(1, 2)
        return a + self.attention(q, kv, kv)[0].transpose(1, 2).view(B, C, H, W)

    def forward(self, x1, x2):
        f1, f2 = encode(self.enc, x1), encode(self.enc, x2)
        if self.attention is not None:
            f1[-1], f2[-1] = self.cross(f1[-1], f2[-1]), self.cross(f2[-1], f1[-1])
        c = self.change_dec([torch.cat([a, b], 1) for a, b in zip(f1, f2)])
        return up(self.change(c)), up(self.sem(self.sem_dec(f1))), up(self.sem(self.sem_dec(f2)))


MODELS = {
    'Early Fusion': EarlyFusion,
    'SSCD (Siamese Semantic Change Detection)': lambda: Siamese(False),
    'Bi-SRNet (Bi-temporal Semantic Reasoning Network)': lambda: Siamese(True),
}

counts = pixels[1:].astype(float)
CLASS_WEIGHT = torch.tensor(1 / np.sqrt(counts / counts.sum()), dtype=torch.float32, device=device)
CLASS_WEIGHT /= CLASS_WEIGHT.mean()


def dice(prob, onehot):
    return 1 - (2 * (prob * onehot).sum() + 1) / (prob.sum() + onehot.sum() + 1)


def compute_loss(out, y1, y2, loss_name, consistency):
    c, s1, s2 = [o.float() for o in out]
    c = c.squeeze(1)
    changed = (y1 > 0).float()
    weight = CLASS_WEIGHT if 'WCE' in loss_name else None
    loss = F.binary_cross_entropy_with_logits(c, changed)
    for s, y in ((s1, y1 - 1), (s2, y2 - 1)):
        ce = F.cross_entropy(s, y, weight=weight, ignore_index=-1, reduction='none')
        if loss_name.startswith('Focal'):
            ce = (1 - torch.exp(-ce)) ** 2 * ce
        loss = loss + 0.5 * ce.sum() / (y >= 0).sum().clamp(min=1)
        if 'Dice' in loss_name:
            mask = (y >= 0).unsqueeze(1)
            loss = loss + 0.5 * dice(torch.softmax(s, 1) * mask, F.one_hot(y.clamp(min=0), 6).permute(0, 3, 1, 2) * mask)
    if 'Dice' in loss_name:
        loss = loss + dice(torch.sigmoid(c), changed)
    if consistency:
        sim = F.cosine_similarity(torch.softmax(s1, 1), torch.softmax(s2, 1), dim=1)
        loss = loss + ((1 - sim) * (1 - changed) + sim * changed).mean()
    return loss

def metrics(cm):
    cm = cm.astype(float)
    tn, fn, fp, tp = cm[0, 0], cm[0, 1:].sum(), cm[1:, 0].sum(), cm[1:, 1:].sum()
    right = np.trace(cm[1:, 1:])
    iou_change, iou_nochange = tp / (tp + fp + fn), tn / (tn + fp + fn)
    c0 = cm.copy(); c0[0, 0] = 0
    po, pe = np.trace(c0) / c0.sum(), (c0.sum(0) * c0.sum(1)).sum() / c0.sum() ** 2
    kappa = (po - pe) / (1 - pe)
    p, r = right / (tp + fp), right / (tp + fn)
    return {
        'SeK (Separated Kappa) %': 100 * kappa * math.exp(iou_change - 1),
        'Fscd (F1-score for Semantic Change Detection) %': 100 * 2 * p * r / (p + r),
        'mIoU (mean Intersection over Union) %': 50 * (iou_change + iou_nochange),
        'OA (Overall Accuracy) %': 100 * np.trace(cm) / cm.sum(),
        'Precision %': 100 * tp / (tp + fp),
        'Recall %': 100 * tp / (tp + fn),
        'F1 (F1-score) %': 100 * 2 * tp / (2 * tp + fp + fn),
        **{f'IoU {CLASSES[k]} %': 100 * cm[k, k] / (cm[k].sum() + cm[:, k].sum() - cm[k, k]) for k in range(7)},
    }


SEK = 'SeK (Separated Kappa) %'


@torch.no_grad()
def evaluate(model, ids):
    model.eval()
    cm = torch.zeros(49, dtype=torch.long, device=device)
    for s in range(0, len(ids), 16):
        x1, x2, y1, y2 = batch(ids[s:s + 16])
        with torch.autocast('cuda', dtype=torch.float16):
            c, s1, s2 = model(x1, x2)
        changed = (c.squeeze(1) > 0).long()
        for sem, y in ((s1, y1), (s2, y2)):
            cm += torch.bincount((7 * (sem.argmax(1) + 1) * changed + y).flatten(), minlength=49)
    return cm.view(7, 7).cpu().numpy()

def train(model_name, loss_name):
    file = WORK / 'results' / (re.sub('[^A-Za-z0-9]+', '_', f'{model_name}__{loss_name}') + '.json')
    if file.exists():
        return json.loads(file.read_text())
    torch.manual_seed(0); np.random.seed(0)
    model = MODELS[model_name]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler()
    consistency = model_name.startswith('Bi-SRNet')
    best, history = -1e9, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order, total = np.random.permutation(TRAIN), 0.0
        for s in range(0, len(order) - BATCH + 1, BATCH):
            x1, x2, y1, y2 = batch(order[s:s + BATCH], augment=True)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x1, x2)
            loss = compute_loss(out, y1, y2, loss_name, consistency)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()
        val_sek = metrics(evaluate(model, VAL))[SEK]
        history.append({'epoch': epoch, 'train_loss': total / (len(order) // BATCH), 'val_SeK': val_sek})
        print(f'{model_name[:30]:30s} | {loss_name:28s} | epoch {epoch:2d} | train loss {history[-1]["train_loss"]:.3f} | validation SeK {val_sek:.2f}')
        if val_sek > best:
            best, best_state = val_sek, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, WORK / 'results' / (file.stem + '.pt'))
    result = {'Model': model_name, 'Loss': loss_name, 'Best validation SeK': best,
              'confusion': evaluate(model, TEST).tolist(), 'history': history, 'weights': str(WORK / 'results' / (file.stem + '.pt'))}
    file.write_text(json.dumps(result))
    return result

RUNS = []
for name in MODELS:
    RUNS.append(train(name, 'CE (Cross-Entropy)'))
BEST_MODEL = max(RUNS, key=lambda r: r['Best validation SeK'])['Model']

for loss_name in ['WCE (Weighted Cross-Entropy)', 'Focal loss', 'CE (Cross-Entropy) + Dice', 'WCE (Weighted Cross-Entropy) + Dice']:
    RUNS.append(train(BEST_MODEL, loss_name))

RESULTS = pd.DataFrame([{'Model': r['Model'], 'Loss': r['Loss'], **metrics(np.array(r['confusion']))} for r in RUNS])
RESULTS.index = range(1, len(RESULTS) + 1)
pd.set_option('display.max_columns', None, 'display.width', 250)
display(RESULTS.round(2))
RESULTS.round(2).to_excel(WORK / 'SECOND_SCD_Results.xlsx')
FINAL = max(RUNS, key=lambda r: r['Best validation SeK'])
print('Best model:', FINAL['Model'], '|', FINAL['Loss'])
print('Saved:', WORK / 'SECOND_SCD_Results.xlsx')

main = ['SeK (Separated Kappa) %', 'Fscd (F1-score for Semantic Change Detection) %', 'mIoU (mean Intersection over Union) %', 'OA (Overall Accuracy) %']
exp1 = RESULTS.iloc[:3].set_index('Model')[main]
exp1.index = ['Early Fusion', 'SSCD', 'Bi-SRNet']
exp1.plot.bar(figsize=(12, 4.5), rot=0, title='Model comparison on test set (loss: CE = Cross-Entropy)', ylabel='%')
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

exp2 = RESULTS[RESULTS.Model == BEST_MODEL].set_index('Loss')[main]
exp2.plot.barh(figsize=(12, 5), title=f'Loss comparison on {BEST_MODEL}', xlabel='%')
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

iou = RESULTS.set_index(RESULTS.Model.str.split(' ').str[0] + ' | ' + RESULTS.Loss)[[f'IoU {c} %' for c in CLASSES]]
iou.columns = CLASSES
iou.T.plot.bar(figsize=(14, 4.5), rot=20, title='IoU (Intersection over Union) of each class on test set', ylabel='%')
plt.legend(fontsize=7); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for r in RUNS:
    label = r['Model'].split(' ')[0] + ' | ' + r['Loss']
    ax[0].plot([h['val_SeK'] for h in r['history']], label=label)
    ax[1].plot([h['train_loss'] for h in r['history']], label=label)
ax[0].set(title='Validation SeK (Separated Kappa) per epoch', xlabel='epoch', ylabel='%')
ax[1].set(title='Training loss per epoch', xlabel='epoch', ylabel='loss')
ax[0].legend(fontsize=7); plt.tight_layout(); plt.show()

cm = np.array(FINAL['confusion'], float)
cm_pct = 100 * cm / cm.sum(0, keepdims=True)
plt.figure(figsize=(7, 6))
plt.imshow(cm_pct, cmap='Blues')
for i in range(7):
    for j in range(7):
        plt.text(j, i, f'{cm_pct[i, j]:.0f}', ha='center', va='center', color='white' if cm_pct[i, j] > 60 else 'black')
plt.xticks(range(7), CLASSES, rotation=45, ha='right'); plt.yticks(range(7), CLASSES)
plt.xlabel('True class'); plt.ylabel('Predicted class'); plt.title('Confusion matrix of best model (% of each true class)')
plt.colorbar(); plt.tight_layout(); plt.show()

model = MODELS[FINAL['Model']]().to(device)
model.load_state_dict(torch.load(FINAL['weights']))
model.eval()
ids = np.sort(TEST[np.argsort(-(LAB1[TEST] > 0).mean((1, 2)))[[5, 50, 100]]])
with torch.no_grad():
    x1, x2, _, _ = batch(ids)
    c, s1, s2 = [o.float() for o in model(x1, x2)]
    changed = (c.squeeze(1) > 0).long()
    p1, p2 = ((s1.argmax(1) + 1) * changed).cpu().numpy(), ((s2.argmax(1) + 1) * changed).cpu().numpy()
fig, ax = plt.subplots(3, 6, figsize=(16, 8))
for r, i in enumerate(ids):
    for c_, (img, title) in enumerate([(IMG1[i], 'T1 image'), (IMG2[i], 'T2 image'), (COLOURS[LAB1[i]], 'T1 true'),
                                        (COLOURS[p1[r]], 'T1 predicted'), (COLOURS[LAB2[i]], 'T2 true'), (COLOURS[p2[r]], 'T2 predicted')]):
        ax[r, c_].imshow(img); ax[r, c_].set_title(title); ax[r, c_].axis('off')
plt.suptitle('Test predictions (white = no change, grey = ground, dark green = tree, green = low vegetation, blue = water, maroon = building, red = playground)')
plt.tight_layout(); plt.show()
