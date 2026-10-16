"""Builds every figure in figures/ from the CSV results and the Kaggle log.
Run from the repository root:  python tools/make_figures.py"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P2, P3 = Path('phase2_colab_256/results'), Path('phase3_kaggle_512/results'); OUT = Path('figures'); OUT.mkdir(exist_ok=True)
# reference categorical palette (fixed order), neutral for "before"
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
BEFORE, INK, MUTED, GRID = '#a8a6a0', '#0b0b0b', '#52514e', '#e7e6e2'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'legend.frameon': False})

SEK, FSCD, MIOU, OA = ('SeK (Separated Kappa) %', 'Fscd (F1-score for Semantic Change Detection) %',
                       'mIoU (mean Intersection over Union) %', 'OA (Overall Accuracy) %')
SHORT = {'Early Fusion': 'Early Fusion', 'SSCD (Siamese Semantic Change Detection)': 'SSCD',
         'Bi-SRNet (Bi-temporal Semantic Reasoning Network)': 'Bi-SRNet',
         'Proposed: Bi-SRNet-DF (Differential Features + SE fusion)': 'Bi-SRNet-DF'}
LOSS = {'CE (Cross-Entropy)': 'CE', 'WCE (Weighted Cross-Entropy)': 'WCE', 'Focal loss': 'Focal',
        'CE (Cross-Entropy) + Dice': 'CE + Dice', 'WCE (Weighted Cross-Entropy) + Dice': 'WCE + Dice',
        'CE (Cross-Entropy) + Dice + TTA + threshold 0.5': 'CE + Dice + TTA'}
old = pd.read_csv(P2 / 'Results.csv', index_col=0)
new = pd.read_csv(P3 / 'Results.csv', index_col=0)
for d in (old, new):
    d['m'] = d.Model.map(SHORT); d['l'] = d.Loss.map(LOSS); d['run'] = d.m + '\n' + d.l


def label_bars(ax, bars, fmt='{:.2f}'):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, fmt.format(b.get_height()), ha='center', va='bottom', fontsize=8, color=INK)


# 0. Journey across the three phases (best test SeK of each phase)
p1 = pd.read_csv('phase1_laptop_256/results/summary.csv')
best1 = p1.loc[p1.SeK.idxmax()]
steps = [('Phase 1 - laptop 256\nEarly Fusion + CE', float(p1.loc[p1.tag == 'A_early_fusion_ce', 'SeK'].iloc[0]), BEFORE),
         ('Phase 1 - laptop 256\nBi-SRNet-lite + WCE+Dice', float(best1.SeK), BEFORE),
         ('Phase 2 - Colab 256\nBi-SRNet + WCE+Dice', float(old[SEK].max()), BEFORE),
         ('Phase 3 - Kaggle 512\nBi-SRNet + CE+Dice', float(new.iloc[5][SEK]), S1),
         ('Phase 3 - Kaggle 512\nBi-SRNet-DF + TTA (proposed)', float(new.iloc[8][SEK]), S1)]
fig, ax = plt.subplots(figsize=(11, 4.2))
bars = ax.bar([s[0] for s in steps], [s[1] for s in steps], color=[s[2] for s in steps], width=0.6)
label_bars(ax, bars)
ax.set_ylabel('Test SeK (%)'); ax.tick_params(axis='x', labelsize=8)
ax.set_title('Project journey: best test SeK in each phase (grey = 256 x 256, blue = 512 x 512)', loc='left', color=INK)
fig.savefig(OUT / '00_journey_all_phases.png'); plt.close(fig)

# 1. Headline: same model + loss, 256 vs 512
fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(7); w = 0.38
b1 = ax.bar(x - w / 2, old[SEK].values, w, color=BEFORE, label='256 x 256 (before)')
b2 = ax.bar(x + w / 2, new[SEK].values[:7], w, color=S1, label='512 x 512 (after)')
label_bars(ax, b1); label_bars(ax, b2)
ax.set_xticks(x, old.run.values, fontsize=8); ax.set_ylabel('Test SeK (%)')
ax.set_title('SeK for the same model and loss: 256 x 256 vs 512 x 512 (average gain +5.33)', loc='left', color=INK)
ax.legend(loc='upper left'); fig.savefig(OUT / '01_sek_256_vs_512.png'); plt.close(fig)

# 2. Best-before vs best-after on all four main metrics
fig, ax = plt.subplots(figsize=(8, 4))
names = ['SeK', 'Fscd', 'mIoU', 'OA']; ob = old.iloc[6]; nb = new.iloc[8]
x = np.arange(4)
b1 = ax.bar(x - w / 2, [ob[k] for k in (SEK, FSCD, MIOU, OA)], w, color=BEFORE, label='Before: Bi-SRNet + WCE+Dice (256)')
b2 = ax.bar(x + w / 2, [nb[k] for k in (SEK, FSCD, MIOU, OA)], w, color=S1, label='After: Bi-SRNet-DF + TTA (512)')
label_bars(ax, b1); label_bars(ax, b2)
ax.set_xticks(x, names); ax.set_ylabel('Test score (%)'); ax.set_ylim(0, 100)
ax.set_title('Best result before vs after', loc='left', color=INK); ax.legend(loc='upper left')
fig.savefig(OUT / '02_best_before_vs_after.png'); plt.close(fig)

# 3 & 4. Experiment 1 (models) and Experiment 2 (losses) at both resolutions
for tag, d in (('256', old), ('512', new)):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    e1 = d[d.l == 'CE'].drop_duplicates('m')
    for i, (k, c) in enumerate(zip((SEK, FSCD, MIOU, OA), (S1, S2, S3, S4))):
        bars = axes[0].bar(np.arange(len(e1)) + (i - 1.5) * 0.2, e1[k].values, 0.18, color=c, label=k.split(' ')[0])
    axes[0].set_xticks(np.arange(len(e1)), e1.m.values); axes[0].set_ylim(0, 100); axes[0].set_ylabel('Test score (%)')
    axes[0].set_title(f'Experiment 1 - models with CE loss ({tag} x {tag})', loc='left', color=INK); axes[0].legend(ncol=4, loc='upper left')
    e2 = d[(d.m == 'Bi-SRNet')]
    bars = axes[1].barh(e2.l.values, e2[SEK].values, color=S1, height=0.55)
    for b in bars:
        axes[1].text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2, f'{b.get_width():.2f}', va='center', fontsize=9, color=INK)
    axes[1].set_xlabel('Test SeK (%)'); axes[1].invert_yaxis(); axes[1].grid(axis='y', visible=False)
    axes[1].set_title(f'Experiment 2 - losses on Bi-SRNet ({tag} x {tag})', loc='left', color=INK)
    fig.savefig(OUT / f'0{3 if tag == "256" else 4}_experiments_{tag}.png'); plt.close(fig)

# 5. Per-class IoU, best before vs best after
cls = ['no change', 'non-vegetated ground', 'tree', 'low vegetation', 'water', 'building', 'playground']
fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(7)
b1 = ax.bar(x - w / 2, [ob[f'IoU {c} %'] for c in cls], w, color=BEFORE, label='Before (256)')
b2 = ax.bar(x + w / 2, [nb[f'IoU {c} %'] for c in cls], w, color=S1, label='After (512)')
label_bars(ax, b1, '{:.1f}'); label_bars(ax, b2, '{:.1f}')
ax.set_xticks(x, cls); ax.set_ylabel('Test IoU (%)'); ax.set_ylim(0, 100)
ax.set_title('IoU per class - best model before vs after (low vegetation and water remain hardest)', loc='left', color=INK)
ax.legend(loc='upper right'); fig.savefig(OUT / '05_per_class_iou.png'); plt.close(fig)

# 6. Fit check: train vs validation SeK
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, tag, base in zip(axes, ('256x256', '512x512'), (P2, P3)):
    f = pd.read_csv(base / 'Fit_check.csv', index_col=0)
    lab = (f.Model.str.split(' ').str[0] + '\n' + f.Loss.map(LOSS).fillna(f.Loss)).values
    x = np.arange(len(f))
    ax.bar(x - w / 2, f['Train SeK'], w, color=S2, label='Train SeK')
    ax.bar(x + w / 2, f['Validation SeK'], w, color=S1, label='Validation SeK')
    gap = (f['Train SeK'] - f['Validation SeK'])
    for i, g in enumerate(gap):
        ax.text(i, max(f['Train SeK'].iloc[i], f['Validation SeK'].iloc[i]) + 0.5, f'gap {g:+.1f}', ha='center', fontsize=8, color=INK)
    ax.set_xticks(x, lab, fontsize=7, rotation=35, ha='right')
    ax.set_title(f'{tag}: train vs validation SeK (small gap = balanced)', loc='left', color=INK)
axes[0].set_ylabel('SeK (%)'); axes[0].legend(loc='upper left')
fig.savefig(OUT / '06_fit_check.png'); plt.close(fig)

# 7. Comparison with published methods
p = pd.read_csv(P3 / 'vs_Papers.csv', index_col=0)
extra = pd.DataFrame([('SRDFNet (2026)', 87.64, 70.31, 20.36, 60.25), ('BT-SCD (2025)', 88.10, 73.45, 23.61, 63.80),
                      ('ChangeMamba (2024)', 88.12, 73.68, 24.11, 64.03), ('DBTANet (2026)', 88.23, 73.74, 24.12, 64.08),
                      ('Mamba-FCS (2025)', 88.62, 74.07, 25.50, 65.78)], columns=p.columns)
p = pd.concat([p, extra]).sort_values('SeK %').reset_index(drop=True)
YEARS = {'ASN-ATL (SECOND dataset paper)': 'ASN-ATL (2021, SECOND paper)', 'HRSCD.str4': 'HRSCD.str4 (2019)', 'SSCD-l': 'SSCD-l (2022)',
         'Bi-SRNet': 'Bi-SRNet (2022)', 'SCanNet': 'SCanNet (2024)', 'DEFO-MLTSCD': 'DEFO-MLTSCD (2024)', 'LSAFNet (2024)': 'LSAFNet (2024)'}
p['Method'] = p.Method.map(lambda m: YEARS.get(m, m))
p['Method'] = p.Method.str.replace('OURS: Bi-SRNet-DF (proposed) + TTA (512)', 'This work: Bi-SRNet-DF + TTA (512)', regex=False)
fig, ax = plt.subplots(figsize=(9, 5.2))
colors = [S1 if m.startswith('This work') else BEFORE for m in p.Method]
bars = ax.barh(p.Method, p['SeK %'], color=colors, height=0.6)
for b in bars:
    ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2, f'{b.get_width():.2f}', va='center', fontsize=8, color=INK)
ax.set_xlim(14, 27); ax.set_xlabel('SeK (%) on SECOND - published numbers; protocols differ, see README')
ax.grid(axis='y', visible=False); ax.set_title('SeK vs published semantic change detection methods', loc='left', color=INK)
fig.savefig(OUT / '07_vs_papers.png'); plt.close(fig)

# 8. Training curves (first Kaggle session log, 512 x 512)
log = Path('phase3_kaggle_512/logs/kaggle_v3_training_session_log.txt').read_text(errors='ignore')
pat = re.compile(r'(Early Fusion|SSCD|Bi-SRNet)\s+\| CE \(Cross-Entropy\)\s+\| epoch\s+(\d+)/\d+ \| loss ([\d.]+) \| val SeK\s+([-\d.]+)')
rows = pd.DataFrame(pat.findall(log), columns=['m', 'ep', 'loss', 'sek']).astype({'ep': int, 'loss': float, 'sek': float}).drop_duplicates(['m', 'ep'])
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for m, c in zip(('Early Fusion', 'SSCD', 'Bi-SRNet'), (S1, S2, S3)):
    d = rows[rows.m == m]
    axes[0].plot(d.ep, d.loss, color=c, lw=2, label=m); axes[1].plot(d.ep, d.sek, color=c, lw=2, label=m)
axes[0].set(xlabel='epoch', ylabel='training loss'); axes[1].set(xlabel='epoch', ylabel='validation SeK (%)')
axes[0].set_title('Training loss per epoch (512 x 512, CE)', loc='left', color=INK)
axes[1].set_title('Validation SeK per epoch (512 x 512, CE)', loc='left', color=INK)
axes[1].legend(loc='lower right'); fig.text(0, -0.03, 'Bi-SRNet shows epochs 1-5 only: the first session was stopped by a RAM limit and resumed in a second session.', fontsize=8, color=MUTED)
fig.savefig(OUT / '08_training_curves_512.png'); plt.close(fig)
print('figures written:', sorted(f.name for f in OUT.glob('*.png')))
