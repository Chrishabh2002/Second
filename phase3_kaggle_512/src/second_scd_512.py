# Exported from the matching notebook in notebooks/. Run the notebook on Colab/Kaggle (GPU); this file is for reading and diffing.
# ===================== 1. Setup and configuration =====================
import os, re, sys, json, math, time, shutil, subprocess, zipfile, faulthandler, gc, ctypes
try:   # limit glibc malloc arenas: stops RAM from slowly growing when many threads decode images
    LIBC = ctypes.CDLL('libc.so.6'); LIBC.mallopt(-8, 2)   # M_ARENA_MAX = 2
except Exception:
    LIBC = None
faulthandler.enable(file=sys.__stderr__, all_threads=True)   # if the kernel crashes, the low-level reason is printed in Logs
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from PIL import Image
from IPython.display import display

# ---- main settings ----
SIZE         = 512          # native SECOND resolution
BACKBONE     = 'resnet34'   # 'resnet18' | 'resnet34' | 'resnet50'
PRETRAINED   = True         # ImageNet weights (needs Internet)
EPOCHS       = 30           # epochs for each comparison run (8 runs)
FINAL_EPOCHS = 50           # epochs for the proposed model
BATCH        = 8            # 8 fits on T4 (16 GB) at 512 with mixed precision
LR           = 3e-4
WEIGHT_DECAY = 1e-4
TIME_LIMIT_H = 11.3         # stop safely before Kaggle's 12-hour limit (training resumes on re-run)
RAW_DIR      = None         # optional: path to a folder that contains im1/ im2/ label1/ label2/
QUICK_TEST   = False        # True = run a 5-minute check of the whole pipeline on a few images
USE_BOTH_GPUS = False       # keep False: 2-GPU DataParallel crashed the Kaggle kernel; one T4 is enough

torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPU  = torch.cuda.device_count()
AMP    = device.type == 'cuda'
START  = time.time()

ON_KAGGLE = Path('/kaggle/working').exists()
ON_COLAB  = 'google.colab' in sys.modules
if ON_KAGGLE:
    WORK, DATA = Path('/kaggle/working/SECOND_SCD_512'), Path('/tmp/second_data')   # /tmp is not saved to output (keeps the version small)
elif ON_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK, DATA = Path('/content/drive/MyDrive/SECOND_SCD_512'), Path('/content/second_data')
else:
    WORK, DATA = Path('SECOND_SCD_512').resolve(), Path('second_data').resolve()
RESULTS = WORK / ('results_quicktest' if QUICK_TEST else 'results')
for p in (WORK, DATA, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

# ---- resume: copy finished / partial runs from a previous Kaggle output added as Input ----
if ON_KAGGLE and Path('/kaggle/input').exists():
    copied = 0
    for f in Path('/kaggle/input').rglob(f'{RESULTS.name}/*'):
        if f.suffix in ('.json', '.pt') and not (RESULTS / f.name).exists():
            shutil.copy(f, RESULTS / f.name); copied += 1
    if copied:
        print(f'Resumed {copied} result / checkpoint files from a previous run.')

gpu = ', '.join(torch.cuda.get_device_name(i) for i in range(N_GPU)) or 'none (CPU only - training will be very slow)'
print(f'PyTorch {torch.__version__} | GPU: {gpu} | results -> {RESULTS}')

FULL_FORM = {
    'SCD': 'Semantic Change Detection',
    'SECOND': 'SEmantic Change detectiON Dataset',
    'SSCD': 'Siamese Semantic Change Detection',
    'Bi-SRNet': 'Bi-temporal Semantic Reasoning Network',
    'SR block': 'Semantic Reasoning block (self-attention / cross-attention)',
    'Bi-SRNet-DF': 'Bi-SRNet with Differential Features (proposed)',
    'SE': 'Squeeze-and-Excitation channel attention',
    'TTA': 'Test-Time Augmentation (flip averaging)',
    'ResNet-34': 'Residual Network with 34 layers',
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

# ===================== 2. Data: find or download SECOND, read 512 x 512 images from disk =====================
CLASSES = ['no change', 'non-vegetated ground', 'tree', 'low vegetation', 'water', 'building', 'playground']
COLOURS = np.array([[255, 255, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [0, 0, 255], [128, 0, 0], [255, 0, 0]], np.uint8)
CODES = [(int(r) << 16) | (int(g) << 8) | int(b) for r, g, b in COLOURS]
NAMES = ('img1', 'img2', 'lab1', 'lab2')


def to_class(rgb):
    code = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)
    out = np.zeros(code.shape, np.uint8)
    for k, c in enumerate(CODES):
        out[code == c] = k
    return out


def find_raw():
    roots = ([Path(RAW_DIR)] if RAW_DIR else []) + [DATA] + ([Path('/kaggle/input')] if ON_KAGGLE else [])
    found = []
    for root in roots:
        if root.exists():
            for p in root.rglob('label1'):
                d = p.parent
                if all((d / s).is_dir() for s in ('im1', 'im2', 'label2')):
                    found.append((len(list(p.glob('*.png'))), d))
    return max(found)[1] if found else None


def download_second():
    print('SECOND not found locally - downloading from Google Drive ...', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
    import gdown
    rar = DATA / 'second.rar'
    try:
        gdown.download(id='1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc', output=str(rar), quiet=False)
    except Exception as e:
        raise RuntimeError('Google Drive download failed (often a daily quota limit). On Kaggle use "Add Input" and add a SECOND dataset '
                           '(folders im1, im2, label1, label2), or set RAW_DIR to its path, then run this cell again.') from e
    if not shutil.which('bsdtar'):
        subprocess.run('apt-get -qq update > /dev/null; apt-get -qq install -y libarchive-tools > /dev/null', shell=True)
    (DATA / 'raw').mkdir(exist_ok=True)
    print('Extracting ...', flush=True)
    subprocess.run(['bsdtar', '-xf', str(rar), '-C', str(DATA / 'raw')], check=True)
    rar.unlink()


class LazyStack:
    # reads 512 x 512 PNGs from disk only when needed (no multi-GB cache, so RAM and disk stay low)
    def __init__(self, raw, folder, names, label):
        self.raw, self.folder, self.names, self.label = raw, folder, names, label

    def __len__(self):
        return len(self.names)

    def read(self, i):
        im = Image.open(self.raw / self.folder / self.names[i]).convert('RGB')
        if im.size != (SIZE, SIZE):   # SECOND is already 512 x 512, so this normally never runs
            im = im.resize((SIZE, SIZE), Image.NEAREST if self.label else Image.LANCZOS)
        a = np.asarray(im)
        return to_class(a) if self.label else a

    def __getitem__(self, idx):
        if np.ndim(idx) == 0:
            return self.read(int(idx))
        return np.stack([self.read(int(i)) for i in idx])


def disk_ram():
    du = shutil.disk_usage(DATA)
    try:
        import psutil; ram = f'{psutil.virtual_memory().available / 1e9:.1f} GB RAM free'
    except Exception:
        ram = ''
    return f'{du.free / 1e9:.1f} GB disk free, {ram}'


print('Looking for SECOND ...', disk_ram(), flush=True)
raw = find_raw()
if raw is None:
    download_second()
    raw = find_raw()
if raw is None:
    raise RuntimeError('Could not find folders im1, im2, label1, label2 after download. Add a SECOND dataset via "Add Input" or set RAW_DIR.')
names = sorted(p.name for p in (raw / 'label1').glob('*.png'))
names = [n for n in names if all((raw / f / n).exists() for f in ('im1', 'im2', 'label2'))]
print(f'Using SECOND from {raw} ({len(names)} image pairs) |', disk_ram(), flush=True)
IMG1, IMG2, LAB1, LAB2 = [LazyStack(raw, f, names, f.startswith('label')) for f in ('im1', 'im2', 'label1', 'label2')]
N = len(IMG1)

perm = np.random.RandomState(0).permutation(N)          # same split as the 256 x 256 experiment
TRAIN, VAL, TEST = perm[:int(0.7 * N)], perm[int(0.7 * N):int(0.8 * N)], perm[int(0.8 * N):]
if QUICK_TEST:
    TRAIN, VAL, TEST = TRAIN[:16], VAL[:8], TEST[:8]
    EPOCHS = FINAL_EPOCHS = 1


def count_pixels(ids):
    one = lambda i: np.bincount(LAB1[i].ravel(), minlength=7) + np.bincount(LAB2[i].ravel(), minlength=7)
    with ThreadPoolExecutor(8) as ex:
        return np.sum(list(ex.map(one, ids)), axis=0)


stats_file = RESULTS / 'train_pixel_counts.json'
if stats_file.exists() and not QUICK_TEST:
    pixels = np.array(json.loads(stats_file.read_text()))
else:
    print('Counting class pixels in the training set ...', flush=True)
    pixels = count_pixels(TRAIN)
    stats_file.write_text(json.dumps(pixels.tolist()))       # class statistics from the training set only (no test leakage)
change_share = 1 - pixels[0] / pixels.sum()
print(pd.Series({'Image pairs': N, 'Image size': f'{SIZE} x {SIZE}',
                 'Train / Validation / Test pairs': f'{len(TRAIN)} / {len(VAL)} / {len(TEST)}',
                 'Changed pixels in train (%)': round(100 * change_share, 2)}).to_string())

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(CLASSES[1:], 100 * pixels[1:] / pixels[1:].sum(), color='#0f766e')
ax[0].set(title='Share of each land-cover class in changed pixels (train)', ylabel='%')
ax[0].tick_params(axis='x', rotation=25)
ax[1].pie([1 - change_share, change_share], labels=['no change', 'change'], autopct='%1.1f%%', colors=['#cbd5e1', '#ef4444'])
ax[1].set_title('Changed vs unchanged pixels')
plt.tight_layout(); plt.show()

# ===================== 3. Batches with background prefetching and GPU augmentation =====================
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
POOL = ThreadPoolExecutor(4)


def load_np(ids):
    i = np.sort(ids)
    return [torch.from_numpy(A[i]) for A in (IMG1, IMG2, LAB1, LAB2)]


def batches(ids, bs, shuffle=False, drop_last=False):
    order = np.random.permutation(ids) if shuffle else np.asarray(ids)
    stop = len(order) - bs + 1 if drop_last else len(order)
    chunks = [order[s:s + bs] for s in range(0, max(stop, 0), bs)]
    futs = [POOL.submit(load_np, c) for c in chunks[:6]]
    for k in range(len(chunks)):
        if k + 6 < len(chunks):
            futs.append(POOL.submit(load_np, chunks[k + 6]))
        b = futs[k].result(); futs[k] = None   # drop the reference so finished batches are freed
        yield b


def to_device(b, augment=False):
    x1, x2 = [t.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255 for t in b[:2]]
    y1, y2 = [t.to(device, non_blocking=True).long() for t in b[2:]]
    if augment:
        B = x1.shape[0]
        # colour jitter (each date separately: contrast +-10 %, brightness +-5 %)
        for x in (x1, x2):
            m = x.mean((1, 2, 3), keepdim=True)
            x.sub_(m).mul_(1 + 0.2 * (torch.rand(B, 1, 1, 1, device=device) - 0.5)).add_(m + 0.2 * (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.5).clamp_(0, 1)
        # temporal swap: T1 <-> T2 (the change mask is symmetric)
        if np.random.rand() < 0.5:
            x1, x2, y1, y2 = x2, x1, y2, y1
        # random scale-up and crop back to 512
        if np.random.rand() < 0.5:
            s = np.random.uniform(1.0, 1.5); S = int(SIZE * s)
            top, left = np.random.randint(0, S - SIZE + 1, 2)
            x1, x2 = [F.interpolate(x, size=(S, S), mode='bilinear', align_corners=False)[..., top:top + SIZE, left:left + SIZE] for x in (x1, x2)]
            y1, y2 = [F.interpolate(y[:, None].float(), size=(S, S), mode='nearest')[:, 0, top:top + SIZE, left:left + SIZE].long() for y in (y1, y2)]
        # flips and 90-degree rotations
        if np.random.rand() < 0.5:
            x1, x2, y1, y2 = [t.flip(-1) for t in (x1, x2, y1, y2)]
        k = np.random.randint(4)
        x1, x2, y1, y2 = [t.rot90(k, (-2, -1)) for t in (x1, x2, y1, y2)]
    x1, x2 = [((x - MEAN) / STD).contiguous() for x in (x1, x2)]
    return x1, x2, y1.contiguous(), y2.contiguous()

# ===================== 4. Models =====================
CH = {'resnet18': (64, 128, 256, 512), 'resnet34': (64, 128, 256, 512), 'resnet50': (256, 512, 1024, 2048)}[BACKBONE]
D = 128   # decoder width


def backbone(in_ch=3):
    r = getattr(torchvision.models, BACKBONE)(weights='DEFAULT' if PRETRAINED else None)
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


def cbr(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def head(i, o):
    return nn.Sequential(cbr(i, D), nn.Conv2d(D, o, 1))


class FPN(nn.Module):
    # Panoptic-FPN style: top-down pathway, then every level is upsampled to 1/4 and fused
    def __init__(self, chans):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, D, 1) for c in chans])
        self.smooth = nn.ModuleList([cbr(D, D) for _ in chans])
        self.out = cbr(4 * D, D)

    def forward(self, feats):
        p = self.lat[3](feats[3]); levels = [p]
        for i in (2, 1, 0):
            p = F.interpolate(p, size=feats[i].shape[-2:], mode='bilinear', align_corners=False) + self.lat[i](feats[i])
            levels.insert(0, p)
        size = levels[0].shape[-2:]
        return self.out(torch.cat([F.interpolate(s(l), size=size, mode='bilinear', align_corners=False)
                                   for s, l in zip(self.smooth, levels)], 1))


class SR(nn.Module):
    # Semantic Reasoning block: attention of a over b (self-attention when b is a), residual with learnable gate
    def __init__(self, c, d=256, heads=4):
        super().__init__()
        d = min(c, d); self.h = heads
        self.q, self.k, self.v, self.o = nn.Conv2d(c, d, 1), nn.Conv2d(c, d, 1), nn.Conv2d(c, d, 1), nn.Conv2d(d, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, a, b=None):
        b = a if b is None else b
        B, _, H, W = a.shape
        split = lambda t: t.flatten(2).view(B, self.h, -1, H * W).transpose(2, 3)
        out = F.scaled_dot_product_attention(split(self.q(a)), split(self.k(b)), split(self.v(b)))
        return a + self.gamma * self.o(out.transpose(2, 3).reshape(B, -1, H, W))


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1), nn.ReLU(inplace=True), nn.Conv2d(c // r, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


def up(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class EarlyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = backbone(6), FPN(CH)
        self.change, self.sem1, self.sem2 = head(D, 1), head(D, 6), head(D, 6)

    def forward(self, x1, x2):
        f = self.dec(encode(self.enc, torch.cat([x1, x2], 1)))
        size = x1.shape[-2:]
        return up(self.change(f), size), up(self.sem1(f), size), up(self.sem2(f), size)


class SSCD(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.sem_dec, self.change_dec = backbone(3), FPN(CH), FPN([2 * c for c in CH])
        self.sem, self.change = head(D, 6), head(D, 1)

    def features(self, x1, x2):
        return encode(self.enc, x1), encode(self.enc, x2)

    def forward(self, x1, x2):
        f1, f2 = self.features(x1, x2)
        c = self.change_dec([torch.cat([a, b], 1) for a, b in zip(f1, f2)])
        size = x1.shape[-2:]
        return up(self.change(c), size), up(self.sem(self.sem_dec(f1)), size), up(self.sem(self.sem_dec(f2)), size)


class BiSRNet(SSCD):
    # intra-temporal SR (self-attention in each date) + cross-temporal SR (each date attends to the other), at 1/16 and 1/32
    def __init__(self):
        super().__init__()
        self.sr = nn.ModuleDict({str(i): SR(CH[i]) for i in (2, 3)})
        self.cross = nn.ModuleDict({str(i): SR(CH[i]) for i in (2, 3)})

    def features(self, x1, x2):
        f1, f2 = encode(self.enc, x1), encode(self.enc, x2)
        for i in (2, 3):
            a, b = self.sr[str(i)](f1[i]), self.sr[str(i)](f2[i])
            f1[i], f2[i] = self.cross[str(i)](a, b), self.cross[str(i)](b, a)
        return f1, f2


class BiSRNetDF(BiSRNet):
    # Proposed: differential features [T1, T2, |T1 - T2|] with SE attention at every scale,
    # and the change head also sees the semantic-difference prior |sem(T1) - sem(T2)|
    def __init__(self):
        super().__init__()
        self.fuse = nn.ModuleList([nn.Sequential(cbr(3 * c, c, 1), SE(c)) for c in CH])
        self.change_dec = FPN(CH)
        self.change = head(2 * D, 1)

    def forward(self, x1, x2):
        f1, f2 = self.features(x1, x2)
        c = self.change_dec([fu(torch.cat([a, b, (a - b).abs()], 1)) for fu, a, b in zip(self.fuse, f1, f2)])
        s1, s2 = self.sem_dec(f1), self.sem_dec(f2)
        size = x1.shape[-2:]
        return up(self.change(torch.cat([c, (s1 - s2).abs()], 1)), size), up(self.sem(s1), size), up(self.sem(s2), size)


EF, SS, BI = 'Early Fusion', 'SSCD (Siamese Semantic Change Detection)', 'Bi-SRNet (Bi-temporal Semantic Reasoning Network)'
PROPOSED = 'Proposed: Bi-SRNet-DF (Differential Features + SE fusion)'
MODELS = {EF: EarlyFusion, SS: SSCD, BI: BiSRNet, PROPOSED: BiSRNetDF}
SHORT = {EF: 'Early Fusion', SS: 'SSCD', BI: 'Bi-SRNet', PROPOSED: 'Bi-SRNet-DF (proposed)'}
USES_CONSISTENCY = {BI, PROPOSED}


def build(name):
    m = MODELS[name]().to(device)
    return nn.DataParallel(m) if (USE_BOTH_GPUS and N_GPU > 1) else m   # single GPU by default (DataParallel crashed the kernel on Kaggle)


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


for name in MODELS:
    print(f'{SHORT[name]:24s} {sum(p.numel() for p in MODELS[name]().parameters()) / 1e6:6.1f} M parameters')

# ===================== 5. Losses =====================
counts = pixels[1:].astype(float) + 1.0   # +1 avoids divide-by-zero when a class is absent
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
    if consistency:   # Bi-SRNet semantic consistency: same class where unchanged, different class where changed
        sim = F.cosine_similarity(torch.softmax(s1, 1), torch.softmax(s2, 1), dim=1)
        loss = loss + ((1 - sim) * (1 - changed) + sim * changed).mean()
    return loss

# ===================== 6. Metrics and evaluation (with optional flip TTA and several thresholds) =====================
def metrics(cm):
    cm = cm.astype(float)
    tn, fn, fp, tp = cm[0, 0], cm[0, 1:].sum(), cm[1:, 0].sum(), cm[1:, 1:].sum()
    right = np.trace(cm[1:, 1:])
    iou_change, iou_nochange = tp / (tp + fp + fn), tn / (tn + fp + fn)
    c0 = cm.copy(); c0[0, 0] = 0
    po, pe = np.trace(c0) / c0.sum(), (c0.sum(0) * c0.sum(1)).sum() / c0.sum() ** 2
    kappa = (po - pe) / (1 - pe)
    p, r = right / max(tp + fp, 1), right / max(tp + fn, 1)
    return {
        'SeK (Separated Kappa) %': 100 * kappa * math.exp(iou_change - 1),
        'Fscd (F1-score for Semantic Change Detection) %': 100 * 2 * p * r / max(p + r, 1e-12),
        'mIoU (mean Intersection over Union) %': 50 * (iou_change + iou_nochange),
        'OA (Overall Accuracy) %': 100 * np.trace(cm) / cm.sum(),
        'Precision %': 100 * tp / max(tp + fp, 1),
        'Recall %': 100 * tp / max(tp + fn, 1),
        'F1 (F1-score) %': 100 * 2 * tp / max(2 * tp + fp + fn, 1),
        **{f'IoU {CLASSES[k]} %': 100 * cm[k, k] / max(cm[k].sum() + cm[:, k].sum() - cm[k, k], 1) for k in range(7)},
    }


SEK, MIOU = 'SeK (Separated Kappa) %', 'mIoU (mean Intersection over Union) %'
FLIPS = [(), (-1,), (-2,), (-2, -1)]


@torch.no_grad()
def predict(model, x1, x2, tta=False):
    pc = ps1 = ps2 = 0
    flips = FLIPS if tta else [()]
    for f in flips:
        a, b = (x1.flip(f), x2.flip(f)) if f else (x1, x2)
        with torch.autocast(device.type, dtype=torch.float16, enabled=AMP):
            c, s1, s2 = model(a, b)
        c, s1, s2 = [(t.float().flip(f) if f else t.float()) for t in (c, s1, s2)]
        pc, ps1, ps2 = pc + torch.sigmoid(c.squeeze(1)), ps1 + torch.softmax(s1, 1), ps2 + torch.softmax(s2, 1)
    n = len(flips)
    return pc / n, ps1 / n, ps2 / n


@torch.no_grad()
def evaluate(model, ids, thrs=(0.5,), tta=False):
    # returns {threshold: 7 x 7 confusion matrix}, rows = predicted, columns = true (T1 and T2 pooled)
    model.eval()
    cms = {t: torch.zeros(49, dtype=torch.long, device=device) for t in thrs}
    for b in batches(ids, BATCH):
        x1, x2, y1, y2 = to_device(b)
        pc, ps1, ps2 = predict(model, x1, x2, tta)
        a1, a2 = ps1.argmax(1) + 1, ps2.argmax(1) + 1
        for t in thrs:
            changed = (pc > t).long()
            cms[t] += torch.bincount((7 * a1 * changed + y1).flatten(), minlength=49)
            cms[t] += torch.bincount((7 * a2 * changed + y2).flatten(), minlength=49)
    return {t: cm.view(7, 7).cpu().numpy() for t, cm in cms.items()}

# ===================== 7. Training (resumable, with time guard) =====================
class TimeUp(Exception):
    pass


def run_file(model_name, loss_name):
    return RESULTS / (re.sub('[^A-Za-z0-9]+', '_', f'{model_name}__{loss_name}').strip('_') + '.json')


def fit_label(gap, drop, best_epoch, epochs):
    if gap > 12: return 'Overfit (train SeK much higher than validation)'
    if best_epoch >= epochs - 1 and drop < 0.1: return 'Still improving (more epochs may help)'
    return 'Good fit'


def ram_used():
    try:
        import psutil; return f'RAM used {psutil.Process().memory_info().rss / 1e9:.1f} GB'
    except Exception:
        return ''


def train(model_name, loss_name, epochs):
    file = run_file(model_name, loss_name)
    if file.exists():
        r = json.loads(file.read_text())
        print(f'Loaded finished run: {SHORT[model_name]} | {loss_name} | best val SeK {r["Best validation SeK"]:.2f}')
        return r
    torch.manual_seed(0); np.random.seed(0)
    model = build(model_name)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = max(len(TRAIN) // BATCH, 1)
    warm = steps   # 1 epoch linear warm-up, then cosine decay per iteration
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i / (epochs * steps), 1))))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    consistency = model_name in USES_CONSISTENCY
    best, best_epoch, history, start_epoch = -1e9, 0, [], 1

    last = file.with_name(file.stem + '_last.pt')
    best_file = file.with_suffix('.pt')
    if last.exists():   # resume a run that was interrupted
        ck = torch.load(last, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
        best, best_epoch, history, start_epoch = ck['best'], ck['best_epoch'], ck['history'], ck['epoch'] + 1
        print(f'Resuming {SHORT[model_name]} | {loss_name} from epoch {start_epoch}')

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time(); model.train(); total = 0.0
        for b in batches(TRAIN, BATCH, shuffle=True, drop_last=True):
            x1, x2, y1, y2 = to_device(b, augment=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=AMP):
                out = model(x1, x2)
            loss = compute_loss(out, y1, y2, loss_name, consistency)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
        m = metrics(evaluate(model, VAL)[0.5])
        gc.collect()
        if LIBC is not None: LIBC.malloc_trim(0)   # give freed RAM back to the system every epoch
        history.append({'epoch': epoch, 'train_loss': total / steps, 'val_SeK': m[SEK], 'val_mIoU': m[MIOU], 'seconds': time.time() - t0})
        print(f'{SHORT[model_name]:22s} | {loss_name:28s} | epoch {epoch:2d}/{epochs} | loss {history[-1]["train_loss"]:.3f} '
              f'| val SeK {m[SEK]:5.2f} | val mIoU {m[MIOU]:5.2f} | {history[-1]["seconds"]:.0f}s | {ram_used()}')
        if m[SEK] > best:
            best, best_epoch = m[SEK], epoch
            torch.save(unwrap(model).state_dict(), best_file)
        torch.save({'model': unwrap(model).state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                    'best': best, 'best_epoch': best_epoch, 'history': history, 'epoch': epoch}, last)
        if (time.time() - START) / 3600 > TIME_LIMIT_H and epoch < epochs:
            raise TimeUp

    unwrap(model).load_state_dict(torch.load(best_file, map_location=device))
    train_sek = metrics(evaluate(model, np.sort(TRAIN)[:300])[0.5])[SEK]
    drop = best - history[-1]['val_SeK']
    result = {'Model': model_name, 'Loss': loss_name, 'Best validation SeK': best, 'Best epoch': best_epoch, 'Epochs': epochs,
              'Train SeK': train_sek, 'Validation drop after best epoch': drop,
              'Fit': fit_label(train_sek - best, drop, best_epoch, epochs),
              'confusion': evaluate(model, TEST)[0.5].tolist(), 'history': history, 'weights': str(best_file)}
    file.write_text(json.dumps(result))
    last.unlink(missing_ok=True)
    return result


def load_model(r):
    model = build(r['Model'])
    unwrap(model).load_state_dict(torch.load(r['weights'], map_location=device))
    return model.eval()

# ===================== 8. Run all experiments =====================
# Exp 1: architectures with CE  ->  Exp 2: losses on the best architecture  ->  Exp 3: proposed model with the best loss
LOSSES = ['WCE (Weighted Cross-Entropy)', 'Focal loss', 'CE (Cross-Entropy) + Dice', 'WCE (Weighted Cross-Entropy) + Dice']
RUNS, FINISHED = [], False
try:
    for name in (EF, SS, BI):
        RUNS.append(train(name, 'CE (Cross-Entropy)', EPOCHS))
    BEST_MODEL = max(RUNS, key=lambda r: r['Best validation SeK'])['Model']
    for loss_name in LOSSES:
        RUNS.append(train(BEST_MODEL, loss_name, EPOCHS))
    BEST_LOSS = max([r for r in RUNS if r['Model'] == BEST_MODEL], key=lambda r: r['Best validation SeK'])['Loss']
    RUNS.append(train(PROPOSED, BEST_LOSS, FINAL_EPOCHS))
    FINISHED = True
except TimeUp:
    print(f'\n*** Time limit of {TIME_LIMIT_H} h reached. Progress is saved in {RESULTS}. '
          'Start a new session, add this notebook\'s output as Input, and run all cells again to continue. ***')
print(f'Finished runs: {len(RUNS)} | elapsed {(time.time() - START) / 3600:.2f} h')

# ===================== 9. Final model: flip TTA + change threshold tuned on validation =====================
FINAL = max(RUNS, key=lambda r: r['Best validation SeK'])
tta_file = RESULTS / 'final_tta.json'
if tta_file.exists() and json.loads(tta_file.read_text())['weights'] == FINAL['weights']:
    TTA = json.loads(tta_file.read_text())
else:
    model = load_model(FINAL)
    thrs = tuple(np.round(np.arange(0.25, 0.751, 0.05), 2).tolist())
    val = evaluate(model, VAL, thrs, tta=True)
    val_sek = {t: metrics(val[t])[SEK] for t in thrs}
    best_t = max(val_sek, key=val_sek.get)
    TTA = {'weights': FINAL['weights'], 'Model': FINAL['Model'], 'Loss': f"{FINAL['Loss']} + TTA + threshold {best_t}",
           'threshold': best_t, 'val_SeK_by_threshold': val_sek, 'confusion': evaluate(model, TEST, (best_t,), tta=True)[best_t].tolist()}
    tta_file.write_text(json.dumps(TTA))
    del model
print(f"Final model: {SHORT[FINAL['Model']]} | {FINAL['Loss']} | tuned change threshold = {TTA['threshold']}")
plt.figure(figsize=(7, 3))
plt.plot([float(t) for t in TTA['val_SeK_by_threshold']], list(TTA['val_SeK_by_threshold'].values()), 'o-', color='#0f766e')
plt.axvline(TTA['threshold'], color='#ef4444', ls='--')
plt.title('Validation SeK vs change threshold (with flip TTA)'); plt.xlabel('threshold'); plt.ylabel('SeK %')
plt.tight_layout(); plt.show()

# ===================== 10. Results tables and Excel file =====================
ALL = RUNS + [TTA]
RESULTS_DF = pd.DataFrame([{'Model': r['Model'], 'Loss': r['Loss'], **metrics(np.array(r['confusion']))} for r in ALL])
RESULTS_DF.index = range(1, len(RESULTS_DF) + 1)
pd.set_option('display.max_columns', None, 'display.width', 250)
print('Test-set results at 512 x 512'); display(RESULTS_DF.round(2))


def tp_rows(r):
    cm = np.array(r['confusion'], float)
    tn, fn, fp, tp = cm[0, 0], cm[0, 1:].sum(), cm[1:, 0].sum(), cm[1:, 1:].sum()
    return {'Model': SHORT[r['Model']], 'Loss': r['Loss'], 'TP (True Positive)': int(tp), 'FP (False Positive)': int(fp),
            'FN (False Negative)': int(fn), 'TN (True Negative)': int(tn),
            'TPR (True Positive Rate) = Recall %': 100 * tp / (tp + fn), 'FPR (False Positive Rate) %': 100 * fp / (fp + tn),
            'FNR (False Negative Rate) %': 100 * fn / (tp + fn), 'TNR (True Negative Rate) = Specificity %': 100 * tn / (tn + fp),
            'Precision %': 100 * tp / (tp + fp)}


TPFN = pd.DataFrame([tp_rows(r) for r in ALL], index=range(1, len(ALL) + 1))
cm = np.array(TTA['confusion'], float)
PER_CLASS = pd.DataFrame({'TP (True Positive)': np.diag(cm).astype(int), 'FP (False Positive)': (cm.sum(1) - np.diag(cm)).astype(int),
                          'FN (False Negative)': (cm.sum(0) - np.diag(cm)).astype(int),
                          'Precision %': 100 * np.diag(cm) / np.maximum(cm.sum(1), 1), 'Recall %': 100 * np.diag(cm) / np.maximum(cm.sum(0), 1)}, index=CLASSES)
FIT = pd.DataFrame([{'Model': SHORT[r['Model']], 'Loss': r['Loss'], 'Best epoch': r['Best epoch'], 'Epochs': r['Epochs'],
                     'Train SeK': r['Train SeK'], 'Validation SeK': r['Best validation SeK'],
                     'Gap = Train SeK - Validation SeK': r['Train SeK'] - r['Best validation SeK'],
                     'Validation drop after best epoch': r['Validation drop after best epoch'], 'Fit': r['Fit']} for r in RUNS],
                   index=range(1, len(RUNS) + 1))

# previous 256 x 256 results (from SECOND_SCD_Results.xlsx) for direct comparison - same split
PREV_256 = {('Early Fusion', 'CE (Cross-Entropy)'): (9.41, 47.90, 64.79, 84.57),
            ('SSCD', 'CE (Cross-Entropy)'): (12.41, 51.94, 66.47, 85.68),
            ('Bi-SRNet', 'CE (Cross-Entropy)'): (12.87, 52.63, 67.07, 85.44),
            ('Bi-SRNet', 'WCE (Weighted Cross-Entropy)'): (13.25, 52.19, 67.47, 85.57),
            ('Bi-SRNet', 'Focal loss'): (12.78, 51.96, 66.92, 85.62),
            ('Bi-SRNet', 'CE (Cross-Entropy) + Dice'): (13.43, 52.99, 67.72, 85.28),
            ('Bi-SRNet', 'WCE (Weighted Cross-Entropy) + Dice'): (13.92, 53.38, 67.60, 84.68)}
MAIN = ['SeK (Separated Kappa) %', 'Fscd (F1-score for Semantic Change Detection) %', MIOU, 'OA (Overall Accuracy) %']
rows = []
for r in RUNS:
    prev = PREV_256.get((SHORT[r['Model']], r['Loss']))
    if prev:
        now = metrics(np.array(r['confusion']))
        rows.append({'Model': SHORT[r['Model']], 'Loss': r['Loss'],
                     **{f'{k.split(" ")[0]} 256': p for k, p in zip(MAIN, prev)},
                     **{f'{k.split(" ")[0]} 512': now[k] for k in MAIN},
                     'SeK gain (points)': now[SEK] - prev[0]})
COMPARE_256 = pd.DataFrame(rows)
print('\n256 x 256 vs 512 x 512 (same split)'); display(COMPARE_256.round(2))

# published results on SECOND (ResNet-34-class backbones, 512 x 512). Splits can differ slightly between papers.
PAPERS = pd.DataFrame([
    ('ASN-ATL (SECOND dataset paper)', None, 70.0, 16.8, None),
    ('HRSCD.str4', 86.62, 71.15, 18.80, 58.21),
    ('SSCD-l', 87.19, 72.60, 21.86, 61.22),
    ('Bi-SRNet', 87.48, 73.38, 22.43, 61.62),
    ('SCanNet', 87.46, 73.20, 23.34, 62.59),
    ('DEFO-MLTSCD', 87.80, 73.76, 23.73, 62.73),
    ('LSAFNet (2024)', 87.66, 74.01, 24.32, 63.20),
], columns=['Method', 'OA %', 'mIoU %', 'SeK %', 'Fscd %'])
best_row = RESULTS_DF.iloc[-1]
OURS = pd.DataFrame([{'Method': f'OURS: {SHORT[TTA["Model"]]} + TTA (512)', 'OA %': best_row['OA (Overall Accuracy) %'],
                      'mIoU %': best_row[MIOU], 'SeK %': best_row[SEK], 'Fscd %': best_row['Fscd (F1-score for Semantic Change Detection) %']}])
VS_PAPERS = pd.concat([PAPERS, OURS], ignore_index=True).sort_values('SeK %', ignore_index=True)
VS_PAPERS.index += 1
print('\nComparison with published results on SECOND'); display(VS_PAPERS.round(2))

xlsx = WORK / 'SECOND_SCD_512_Results.xlsx'
with pd.ExcelWriter(xlsx) as w:
    RESULTS_DF.round(2).to_excel(w, sheet_name='Results')
    TPFN.round(2).to_excel(w, sheet_name='TP FP FN TN')
    PER_CLASS.round(2).to_excel(w, sheet_name='Per-class TP FP FN')
    FIT.round(2).to_excel(w, sheet_name='Fit check')
    COMPARE_256.round(2).to_excel(w, sheet_name='256 vs 512')
    VS_PAPERS.round(2).to_excel(w, sheet_name='vs Papers')
print('Saved:', xlsx)

# ===================== 11. Charts =====================
short_label = lambda r: SHORT[r['Model']].split(' ')[0] + ' | ' + r['Loss']
exp1 = RESULTS_DF[RESULTS_DF.Loss == 'CE (Cross-Entropy)'].set_index('Model')[MAIN]
exp1.index = [SHORT[m] for m in exp1.index]
exp1.plot.bar(figsize=(12, 4.5), rot=0, title='Exp 1 - model comparison on test set (CE loss, 512 x 512)', ylabel='%')
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

if 'BEST_MODEL' in globals():
    exp2 = RESULTS_DF[(RESULTS_DF.Model == BEST_MODEL) & ~RESULTS_DF.Loss.str.contains('TTA')].set_index('Loss')[MAIN]
    exp2.plot.barh(figsize=(12, 5), title=f'Exp 2 - loss comparison on {SHORT[BEST_MODEL]}', xlabel='%')
    plt.legend(fontsize=8); plt.tight_layout(); plt.show()

if len(COMPARE_256):
    c = COMPARE_256.set_index(COMPARE_256.Model + ' | ' + COMPARE_256.Loss)[['SeK 256', 'SeK 512']]
    c.plot.barh(figsize=(12, 5), color=['#94a3b8', '#0f766e'], title='SeK: 256 x 256 vs 512 x 512 (same test split)', xlabel='SeK %')
    plt.tight_layout(); plt.show()

v = VS_PAPERS.set_index('Method')
colors = ['#ef4444' if m.startswith('OURS') else '#94a3b8' for m in v.index]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
v['SeK %'].plot.barh(ax=ax[0], color=colors, title='SeK vs published methods on SECOND')
v['mIoU %'].plot.barh(ax=ax[1], color=colors, title='mIoU vs published methods on SECOND')
ax[1].set_xlim(60, 80); plt.tight_layout(); plt.show()

iou = RESULTS_DF[[f'IoU {c} %' for c in CLASSES]].copy()
iou.index = [short_label(r) for r in ALL]
iou.columns = CLASSES
iou.T.plot.bar(figsize=(14, 4.5), rot=20, title='IoU of each class on test set', ylabel='%')
plt.legend(fontsize=7); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for r in RUNS:
    ax[0].plot([h['val_SeK'] for h in r['history']], label=short_label(r))
    ax[1].plot([h['train_loss'] for h in r['history']], label=short_label(r))
ax[0].set(title='Validation SeK per epoch', xlabel='epoch', ylabel='%')
ax[1].set(title='Training loss per epoch', xlabel='epoch', ylabel='loss')
ax[0].legend(fontsize=7); plt.tight_layout(); plt.show()

cm = np.array(TTA['confusion'], float)
cm_pct = 100 * cm / np.maximum(cm.sum(0, keepdims=True), 1)
plt.figure(figsize=(7, 6))
plt.imshow(cm_pct, cmap='Blues')
for i in range(7):
    for j in range(7):
        plt.text(j, i, f'{cm_pct[i, j]:.0f}', ha='center', va='center', color='white' if cm_pct[i, j] > 60 else 'black')
plt.xticks(range(7), CLASSES, rotation=45, ha='right'); plt.yticks(range(7), CLASSES)
plt.xlabel('True class'); plt.ylabel('Predicted class'); plt.title('Confusion matrix of final model (% of each true class)')
plt.colorbar(); plt.tight_layout(); plt.show()

# ===================== 12. Test predictions of the final model =====================
model = load_model(FINAL)
order = TEST[np.argsort([-(LAB1[i] > 0).mean() for i in TEST])]   # reads test labels once (~10 s)
ids = np.sort(order[[min(k, len(order) - 1) for k in (5, 50, 100)]])
x1, x2, _, _ = to_device(load_np(ids))
pc, ps1, ps2 = predict(model, x1, x2, tta=True)
changed = (pc > TTA['threshold']).long()
p1, p2 = ((ps1.argmax(1) + 1) * changed).cpu().numpy(), ((ps2.argmax(1) + 1) * changed).cpu().numpy()
fig, ax = plt.subplots(len(ids), 6, figsize=(18, 3.2 * len(ids)), squeeze=False)
for r, i in enumerate(ids):
    for c_, (img, title) in enumerate([(IMG1[i], 'T1 image'), (IMG2[i], 'T2 image'), (COLOURS[LAB1[i]], 'T1 true'),
                                        (COLOURS[p1[r]], 'T1 predicted'), (COLOURS[LAB2[i]], 'T2 true'), (COLOURS[p2[r]], 'T2 predicted')]):
        ax[r, c_].imshow(img); ax[r, c_].set_title(title); ax[r, c_].axis('off')
plt.suptitle('Test predictions at 512 x 512 (white = no change, grey = ground, dark green = tree, green = low vegetation, '
             'blue = water, maroon = building, red = playground)')
plt.tight_layout(); plt.show()

# ===================== 13. Package outputs for download =====================
zip_path = WORK.parent / 'SECOND_SCD_512_outputs.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(xlsx, xlsx.name)
    for f in RESULTS.glob('*'):
        if not f.name.endswith('_last.pt'):
            z.write(f, f'{RESULTS.name}/{f.name}')
print(f'All runs finished: {FINISHED} | total time {(time.time() - START) / 3600:.2f} h')
print('Download:', zip_path, f'({zip_path.stat().st_size / 1e6:.0f} MB) - Excel results + best weights of every run')
