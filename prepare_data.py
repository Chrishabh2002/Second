"""Stream the SECOND train set (.rar from Google Drive) and store a downsampled copy.

Disk is too small to hold the archive plus its extracted contents, so the archive is
never written to disk:  gdown -> bsdtar (rar -> tar stream) -> this script.

Output (data/SECOND_256/):
  im1/XXXXX.png, im2/XXXXX.png   RGB, resized 512 -> SIZE (Lanczos)
  label1/XXXXX.png, label2/XXXXX.png   uint8 class index, resized with NEAREST
Class index: 0 no-change, 1 non-veg ground, 2 tree, 3 low vegetation, 4 water,
             5 building, 6 playground
"""
import io
import os
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image

SIZE = int(os.environ.get("SIZE", 256))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"SECOND_{SIZE}")
GDRIVE_ID = "1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc"  # SECOND_train_set.rar (2.2 GB)

# Official SECOND colour map
COLORS = {
    (255, 255, 255): 0,  # no change
    (128, 128, 128): 1,  # non-vegetated ground surface
    (0, 128, 0): 2,      # tree
    (0, 255, 0): 3,      # low vegetation
    (0, 0, 255): 4,      # water
    (128, 0, 0): 5,      # building
    (255, 0, 0): 6,      # playground
}
LUT = np.full(256 ** 3, 255, dtype=np.uint8)
for (r, g, b), idx in COLORS.items():
    LUT[(r << 16) | (g << 8) | b] = idx


def rgb_to_index(arr):
    arr = arr.astype(np.int64)
    idx = LUT[(arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]]
    unknown = (idx == 255).mean()
    if unknown > 0:
        idx[idx == 255] = 0
    return idx, unknown


def main():
    for d in ("im1", "im2", "label1", "label2"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    gd = subprocess.Popen(
        [os.path.join(os.path.dirname(sys.executable), "gdown"), GDRIVE_ID, "-O", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    bt = subprocess.Popen(["bsdtar", "-cf", "-", "--format=ustar", "@-"],
                          stdin=gd.stdout, stdout=subprocess.PIPE)
    gd.stdout.close()
    n, bad = 0, 0.0
    with tarfile.open(fileobj=bt.stdout, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".png"):
                continue
            parts = m.name.split("/")
            sub, fname = parts[-2], parts[-1]
            if sub not in ("im1", "im2", "label1", "label2"):
                continue
            if os.path.exists(os.path.join(OUT, sub, fname)):  # resume after a broken stream
                n += 1
                continue
            img = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
            if sub.startswith("im"):
                img.resize((SIZE, SIZE), Image.LANCZOS).save(os.path.join(OUT, sub, fname))
            else:
                idx, unknown = rgb_to_index(np.asarray(img))
                bad = max(bad, unknown)
                Image.fromarray(idx).resize((SIZE, SIZE), Image.NEAREST).save(
                    os.path.join(OUT, sub, fname))
            n += 1
            if n % 500 == 0:
                print(f"{n} files ({m.name})", flush=True)
    bt.wait(); gd.wait()
    print(f"done: {n} files, max unknown-colour fraction in a label = {bad:.4f}")


if __name__ == "__main__":
    main()
