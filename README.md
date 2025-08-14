# 🎯  (PatchBandit): Query-Efficient Adaptive Adversarial Patches for Object Detectors

(a.k.a. *PatchBandit*) is a **score-only black-box adversarial patch attack** for object detectors (YOLO family in our experiments). It jointly optimizes **where** to place a patch, **what** texture to paint, and **how large** the patch must grow—under a **hard detector query budget**. PatchBandit interleaves:

- a **contextual bandit** (Thompson Sampling) over candidate paste locations,
- a **NES (Natural Evolution Strategies)** zeroth-order patch optimizer,
- a **progress-triggered size ladder** that expands only when needed,
- an optional **plain+EOT dual success criterion** to avoid over-optimistic “EOT-only” wins,
- optional **appearance / printability penalties** for controlled visual deviation.

We provide a **fully instrumented evaluation harness** that logs per-image metrics (queries, size, scores, area %, appearance diff) to CSV so you can reproduce paper figures, run ablations, and compute downstream detection mAP drops on clean vs patched imagery.

---

## 🔥 TL;DR

- **Black-box**: No gradients, just detector scores.
- **Query-budget aware**: Tune population, transforms, and max iterations.
- **Adaptive**: Grows patch only when progress stalls.
- **Instrumented**: Every run logs a CSV; summarizers & plotting tools included.
- **Physical transfer**: A printed patch trained to suppress *bottle* and "bicycle" succeeds across distances, angles, and unseen bottles.

---

## 📦 Installation

> Tested on **Ubuntu 22.04**, **Python 3.10**, **CUDA-enabled PyTorch**. A GPU is strongly recommended for running the detector attack loops; CPU is fine for summarization.

### 1. Create a Conda env
```bash
conda create -y -n patchbandit python=3.10
conda activate patchbandit
```

### 2. Install Python deps

Please install below packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA
pip install ultralytics==8.* pandas numpy tqdm pillow matplotlib opencv-python
```

### 3. Clone / layout
```
patch-bandit/
├── main.py
├── patch_bandit.py
├── bandit_cts.py
├── patch.py
├── dataloader.py
├── optimizer.py
├── placement_utils.py
├── util.py
├── summarize_runs.py
├── ...
└── VOCdevkit/
    └── VOC2007/   # PASCAL VOC images + annotations
```

### 4. Dataset: PASCAL VOC 2007 (eval split)

Place (or symlink) the standard VOC tree here:
```
VOCdevkit/VOC2007/JPEGImages/*.jpg
VOCdevkit/VOC2007/Annotations/*.xml
VOCdevkit/VOC2007/ImageSets/Main/test.txt   # or your eval list
```

### 5. Detector weights

PatchBandit expects a YOLO and Faster R-CNN detector. We used a YOLO and Faster R-CNN model fine-tuned on VOC; put your checkpoint here (example):
```
runs/detect/train5/weights/best.pt
runs/frcnn_voc/best.pth
```
You can also point to any Ultralytics-compatible `.pt` model (e.g., a YOLOv5/YOLOv8 variant) at runtime.

---

## 🧠 Core Idea

PatchBandit runs an iterative loop:

1. **Place:** Sample a candidate grid cell via contextual bandit (features include normalized coords + simple appearance cues).  
2. **Probe:** Paste a patch and evaluate *M* noisy NES perturbations (optionally each under *K* EOT transforms).  
3. **Reward:** Combine detector score drop (plain + optional EOT) and appearance penalties.  
4. **Update:** NES gradient estimate → texture update.  
5. **Grow if stalled:** If reward doesn’t improve for τ steps, upsample to the next size in a ladder (e.g., 40→60→80→…).  
6. **Stop** when strict suppression thresholds are met or the query budget is exhausted.  

All detector calls are counted.

---

## ⚙️ Quick Start: Single Debug Run (Digital)

Attack 50 images just to see the pipeline work:
```bash
python main.py \
  --num 50 \
  --seed 0 \
  --save-dir runs_debug50 \
  --mode digital \
  --contextual \
  --patch-sizes 40,60,80,120,160 \
  --grow-patience 15 \
  --max-queries 150 \
  --w-stealth 0 \
  --w-print 0 \
  --eot-samples 0
```
Outputs go to: `runs_debug50/single/attack_metrics.csv` + patched images.

---

## 📊 Reproducing Paper Main Runs (2200-Image Eval)

Below are the exact commands used to generate the runs reported in the paper draft. Each attacks **2200** PASCAL VOC images (digital mode). Adjust `--num` for full sweeps.

> **IMPORTANT:** For the *PowerFast* run, set `STOP_ON_TOP_SWAP = True` in `patch_bandit.py` before launching (restore afterward). Other runs expect strict mode (`STOP_ON_TOP_SWAP = False`).  
> All commands assume you are in the project root, with your Conda env active, and VOC data + YOLO weights available.

### 1. Full Contextual (strict baseline for paper)

Contextual UCB placement, adaptive ladder, strict termination.
```bash
python main.py --num 2200 --seed 0 --save-dir runs10_full_ctx_strict \
  --mode digital --contextual \
  --patch-sizes 40,60,80,120,160 \
  --grow-patience 15 --max-queries 400 \
  --w-stealth 0 --w-print 0 --eot-samples 0
```

### 2. Fixed Large (DPatch-style “just go big”)

Single 160-px patch; contextual placement allowed.
```bash
python main.py --num 2200 --seed 0 --save-dir runs10_fixed160 \
  --mode digital --contextual \
  --patch-sizes 160 \
  --max-queries 400 \
  --w-stealth 0 --w-print 0 --eot-samples 0
```

### 3. Stealth-Lite

Modest appearance + printability weighting.
```bash
python main.py --num 2200 --seed 0 --save-dir runs10_stealth_lite \
  --mode digital --contextual \
  --patch-sizes 40,60,80,120,160 \
  --grow-patience 15 --max-queries 400 \
  --w-stealth 0.5 --w-print 0.25 --eot-samples 0
```

### 4. PowerFast (top-swap early exit, shorter budget)

Toggle `STOP_ON_TOP_SWAP=True` in `patch_bandit.py` before running.
```bash
python main.py --num 2200 --seed 0 --save-dir runs10_powerfast \
  --mode digital --contextual \
  --patch-sizes 40,60,80,120,160 \
  --grow-patience 10 --max-queries 150 \
  --w-stealth 0 --w-print 0 --eot-samples 0
```

### 5. Aggressive Growth (τ<sub>grow</sub>=5)

Escalate patch size quickly.
```bash
python main.py --num 2200 --seed 0 --save-dir runs10_gp5 \
  --mode digital --contextual \
  --patch-sizes 40,60,80,120,160 \
  --grow-patience 5 --max-queries 400 \
  --w-stealth 0 --w-print 0 --eot-samples 0
```

### 6. EOT=5 Reward Averaging

Sample 5 random transforms per probe; heavier query cost.
```bash
python main.py --num 2200 --seed 0 --save-dir runs10_eot5 \
  --mode digital --contextual \
  --patch-sizes 40,60,80,120,160 \
  --grow-patience 15 --max-queries 400 \
  --w-stealth 0 --w-print 0 --eot-samples 5
```

### 7. Stealth-Strong

Heavy appearance + printability weighting (stress test).
```bash
python main.py --num 2200 --seed 0 --save-dir runs10_stealth_strong \
  --mode digital --contextual \
  --patch-sizes 40,60,80,120,160 \
  --grow-patience 15 --max-queries 400 \
  --w-stealth 2.0 --w-print 1.0 --eot-samples 0
```

#### (Optional) Non-Contextual & Random Baselines

If you later generate these, they’ll plug directly into the summarizers.

**NoCtx** (ε-greedy / no features):
```bash
python main.py --num 2200 --seed 0 --save-dir runs10_nocx_strict \
  --mode digital \
  --patch-sizes 40,60,80,120,160 \
  --grow-patience 15 --max-queries 400 \
  --w-stealth 0 --w-print 0 --eot-samples 0
```

## 📈 Summarize Results Across Runs

After runs finish, produce per-run and cross-run stats (success %, query medians, size histograms, area buckets, etc.):
```bash
python summarize_runs.py \
  --roots runs10_full_ctx_strict \
          runs10_fixed160 \
          runs10_stealth_lite \
          runs10_powerfast \
          runs10_gp5 \
          runs10_eot5 \
          runs10_stealth_strong
```
Outputs:

* `aggregate_summary.csv` (concatenated per-image rows)  
* Pretty-printed run summaries in the terminal  
* Area bucket breakdowns  

If you later add NoCtx / Random runs, include them in `--roots`.

---

## 🧼 Repair Missing Paths (optional)

If you move run folders, the CSV `patched_path` column may break. Use:
```bash
python repair_run_csv.py --run-dir runs10_full_ctx_strict --backup
```
Repeat for each run as needed. A `.bak` copy is kept.

---

## 📉 Compute mAP Drop (Clean vs Patched)

Use the **v2** script (resolves patched scenes to their correct VOC image IDs; fixes NaN mAP errors you may see with older scripts).

### Strict successes only
```bash
python compute_map_drop_v2.py \
  --voc-root VOCdevkit/VOC2007 \
  --weights runs/detect/train5/weights/best.pt \
  --run-dir runs10_full_ctx_strict \
  --success-only \
  --out-prefix full_ctx_success_v2
```

### Intention-to-treat (all 2200 imgs; impute clean where patch missing)
```bash
python compute_map_drop_v2.py \
  --voc-root VOCdevkit/VOC2007 \
  --weights runs/detect/train5/weights/best.pt \
  --run-dir runs10_full_ctx_strict \
  --impute-clean \
  --out-prefix full_ctx_all_v2
```
Outputs:

* `<prefix>.map.json` (per-class AP + mAP clean / patched / delta)  
* `<prefix>.map.csv` (CSV form)  

Repeat for `runs10_powerfast`, `runs10_fixed160`, etc.

---

## 🧪 Physical Pilot Script

A minimal print-and-capture test (single image → optimize patch → apply to real object):
```bash
python physical.py \
  --image ./physical/x.jpg \
  --mode physical \
  --weights runs/detect/train5/weights/best.pt \
  --out-dir physical_output
```
Then photograph the printed patch on multiple objects and run the detector to log confidence shifts.

---

## 📂 Output Structure (per run)
```
runs10_full_ctx_strict/
└── single/
    ├── attack_metrics.csv      # one row per eval image
    ├── attack_summary.json     # (optional) aggregate info
    ├── img000_succeeded_patch_40_step_6_0.png
    ├── img000_succeeded_patch_40_step_6_0_patch.png   # texture only
    ├── img001_failed_patch_160_step_84_0.png
    └── ...
```

**CSV Columns** (`attack_metrics.csv`)  
`condition,img_id,success,clean_top_class,clean_top_conf,patched_top_class,patched_top_conf,patch_size,queries,patch_area_pct,visual_diff_patch,patched_path`

---
## 🧮 Re-Running Summaries After You Fill In Missing Runs
```bash
python summarize_attack_metrics.py \
  --roots runs10_full_ctx_strict \
          runs10_fixed160 \
          runs10_stealth_lite \
          runs10_powerfast \
          runs10_gp5 \
          runs10_eot5 \
          runs10_stealth_strong
```

## 🤝 Acknowledgments
We build on open-source work from the adversarial ML, object detection, and YOLO communities. In particular:
BibTeX stub (update before release):
```bibtex
@inproceedings{brown2017adversarialpatch,
  title     = {Adversarial Patch},
  author    = {Brown, Tom B. and Man{\'e}, Dandelion and Roy, Aurko and Abadi, Mart{\'\i}n and Gilmer, Justin},
  booktitle = {NIPS Workshop},
  year      = {2017},
  eprint    = {1712.09665},
  archivePrefix = {arXiv}
}

@article{liu2018dpatch,
  title   = {DPatch: An Adversarial Patch Attack on Object Detectors},
  author  = {Liu, Xin and Yang, Huanrui and Liu, Ziwei and Song, Linghao and Li, Hai and Chen, Yiran},
  journal = {arXiv preprint arXiv:1806.02299},
  year    = {2018}
}

@inproceedings{ilyas2018blackbox,
  title={Black-box Adversarial Attacks with Limited Queries and Information},
  author={Ilyas, Andrew and Engstrom, Logan and Athalye, Anish and Lin, Jessy},
  booktitle={ICML},
  year={2018},
  eprint={1804.08598},
  archivePrefix={arXiv}
}
```
