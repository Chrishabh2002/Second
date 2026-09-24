# Engineering notes: running full SCD experiments on a 16 GB laptop

These are the practical problems hit while running this project on one Apple M4 laptop (16 GB RAM,
nearly full SSD) and how each one was solved. They are useful for anyone reproducing the work on
similar hardware.

## 1. The dataset archive did not fit on disk

**Problem.** SECOND comes as a single 2.2 GB `.rar` on Google Drive. Downloading it, extracting it
(512 px PNGs) and keeping a resized copy needs several GB of free space. The disk did not have it.

**Fix.** [prepare_data.py](../prepare_data.py) never writes the archive to disk:

```
gdown <id> -O -  |  bsdtar -cf - --format=ustar @-  |  Python tarfile (stream mode)
```

`bsdtar` converts the RAR stream into a tar stream on the fly. Python reads it one member at a
time, resizes the image to 256 px and saves only the small copy. If the stream breaks, files that
already exist are skipped, so a re-run resumes where it stopped. The final dataset takes about
756 MB.

## 2. Loading all images into RAM pushed the machine into swap

**Problem.** The first version of `load_all()` decoded all 2,968 × 4 PNGs into NumPy arrays up
front. Together with the model, the IDE and other apps, this filled memory. macOS grew the swap
file to 3.8 GB, and that swap file used up the last free disk space.

**Fix.** `LazyPNG` in [scd.py](../scd.py) is an array-like view that decodes a PNG only when it
is indexed (`im1[idx]`). Training code did not have to change. A training process now uses about
450 MB of RAM, and swap went back to 0.

## 3. Runs failed with "No space left on device"

**Problem.** With the disk full, `torch.save` failed during checkpointing, and the log files of
later runs could not even be created. The whole first pipeline attempt failed. Only the Early
Fusion run finished.

**Fix.**
- Freed ~33 GB by deleting re-downloadable package caches (`~/.cache/uv`, pip cache), with the
  owner's approval.
- For a while, each run deleted its checkpoint after test evaluation; the metrics JSON is what is
  kept. Once there was enough disk space, checkpoints were kept again for the qualitative figure.
- Failed attempts are kept as `logs/run_all_attempt*.log` for the record.

## 4. The pipeline stopped when the machine restarted

**Problem.** The machine restarted twice while the pipeline was running. The first restart
stopped the *median-frequency* run and the second stopped the *focal* run.

**Fix.** `run_all.sh` skips any run whose `results/<tag>.json` already exists. Re-running the same
command resumes from the first unfinished run. Only the interrupted run is repeated. Partial logs
are kept as `logs/*_interrupted.log`. The pipeline now runs under `caffeinate -i` so the Mac does
not go to idle sleep during training.

## 5. A 368 MB file blocked `git push`

**Problem.** The first commit included `.venv/`, the dataset and checkpoints. GitHub rejects files
over 100 MB (`libtorch_cpu.dylib` is 368 MB). VS Code only showed a misleading "pull first" message.

**Fix.** Add a [.gitignore](../.gitignore) for `.venv/`, `data/`, `checkpoints/`, `*.pt` and
caches, and untrack those paths. Because the large file is still in the first commit's history,
that history has to be rewritten once (for example with `git filter-branch --index-filter 'git rm
-r --cached --ignore-unmatch .venv data checkpoints'`) before the first push.

## 6. Timing

| Model | Minutes per 20-epoch run (M4, MPS) |
|---|---:|
| Early Fusion | ~9.5 |
| SSCD / Bi-SRNet-lite | ~21–23 |

Some runs took longer (the class-balanced run took ~35 minutes). The cause was not investigated;
the most likely reason is other work on the machine at the same time. The full study (3 + 9 runs) takes about 5 hours of machine time.
