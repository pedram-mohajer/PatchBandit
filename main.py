# main.py
"""
Ablation driver for YOLO patch attacks.

Usage (single run, same as before):
    python main.py --num 10 --contextual --save-dir runs_single

Usage (full ablation battery on 500 images):
    python main.py --ablation --num 500 --mode digital --patch-sizes 40,60,80,120,160 --save-dir runs_ablate

Each ablation condition gets its own subfolder under --save-dir and
produces:
    attack_summary.json
    attack_metrics.csv
A master CSV aggregating all conditions is written to:
    <save-dir>/ablation_master.csv
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import random
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF

from util import (
    train_yolo,
    generate_patch_for_image,
    save_attack_summary,
    MODEL_PATH,
)
from dataloader import PascalVOCLoader, VOC_CLASSES
from ultralytics import YOLO
from patch_bandit import AttackConfig
from tqdm.auto import tqdm

from util import train_faster_r_cnn, FRCNN_MODEL_PATH, FasterRCNNWrapper


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true",
                    help="Run predefined ablation study (multiple conditions).")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--mode", choices=["digital", "physical"], default="digital")
    ap.add_argument("--image-set", default="val")
    ap.add_argument("--num", type=int, default=500,
                    help="#images to attack (use <= dataset size).")
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--contextual", action="store_true", help="use contextual bandit (single-run mode).")
    ap.add_argument("--patch-sizes", type=str, default="40,60,80,120,160")
    ap.add_argument("--grow-patience", type=int, default=50)
    ap.add_argument("--min-improv", type=float, default=0.0)
    ap.add_argument("--max-queries", type=int, default=250)
    ap.add_argument("--w-det", type=float, default=1.0)
    ap.add_argument("--w-stealth", type=float, default=0.5)
    ap.add_argument("--w-print", type=float, default=0.25)
    ap.add_argument("--save-dir", type=str, default="results")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # (optional) override EOT from CLI; if omitted, AttackConfig defaults apply
    ap.add_argument("--eot-samples", type=int, default=None)
    ap.add_argument("--eot-scale", type=float, default=None)
    ap.add_argument("--eot-rot", type=float, default=None)
    ap.add_argument("--eot-color", type=float, default=None)
    ap.add_argument("--bandit-eps", type=float, default=0.2, help="ε-greedy exploration rate for NON-contextual placement (1.0 = fully random).")
    ap.add_argument("--detector", choices=["yolo", "frcnn"], default="frcnn")

    return ap.parse_args()


# ------------------------------------------------------------------
def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Normalisation helpers
# ------------------------------------------------------------------
def denorm_to_uint8(img_tensor):
    """ImageNet unnorm CHW float->uint8 HWC."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img_unnorm = (img_tensor * std + mean).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (img_unnorm * 255).astype(np.uint8)


# ------------------------------------------------------------------
def clean_top_class(model, img_tensor):
    """Return (cls_id, conf, per_cls_dict) for clean image."""
    img_np = denorm_to_uint8(img_tensor)
    pred = model.predict(source=img_np, conf=0.001, verbose=False)[0].boxes
    if pred is None or len(pred) == 0:
        return -1, 0.0, {}
    cls_ids = pred.cls.cpu().numpy().astype(int)
    confs = pred.conf.cpu().numpy()
    per_cls = {}
    for c, s in zip(cls_ids, confs):
        per_cls[c] = max(per_cls.get(c, 0.0), float(s))
    top_cls = max(per_cls.items(), key=lambda x: x[1])[0]
    return top_cls, per_cls[top_cls], per_cls


# ------------------------------------------------------------------
def patched_top_class(model, patched_path):
    """Run detector on saved patched PNG; return (cls_id, conf)."""
    img = Image.open(patched_path).convert("RGB")
    img_np = np.array(img)  # HWC uint8
    pred = model.predict(source=img_np, conf=0.001, verbose=False)[0].boxes
    if pred is None or len(pred) == 0:
        return -1, 0.0
    cls_ids = pred.cls.cpu().numpy().astype(int)
    confs = pred.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    return int(cls_ids[idx]), float(confs[idx])


# ------------------------------------------------------------------
def compute_visual_diff_patch(clean_img_tensor, patched_path, cell, patch_size):
    """
    Mean absolute diff in [0,1] pixel space between clean crop and saved patched crop.
    """
    patched_px = TF.to_tensor(Image.open(patched_path).convert("RGB"))  # [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    clean_px = (clean_img_tensor.cpu() * std + mean).clamp(0, 1)
    x, y = cell
    ph = pw = patch_size
    clean_crop = clean_px[:, y:y + ph, x:x + pw]
    patched_crop = patched_px[:, y:y + ph, x:x + pw]
    if clean_crop.numel() == 0 or patched_crop.numel() == 0:
        return float("nan")
    return torch.mean(torch.abs(clean_crop - patched_crop)).item()


# ------------------------------------------------------------------
def build_attack_cfg_from_args(args: argparse.Namespace) -> AttackConfig:
    sizes = tuple(int(s) for s in args.patch_sizes.split(",") if s.strip())
    cfg = AttackConfig(
        patch_sizes=sizes,
        grow_patience=args.grow_patience,
        min_improv=args.min_improv,
        max_queries=args.max_queries,
        w_det=args.w_det,
        w_stealth=args.w_stealth,
        w_print=args.w_print,
        grid_size=args.grid_size,
        use_contextual=args.contextual,
        save_dir=args.save_dir,
        save_every=args.save_every,
        #bandit_eps=args.bandit_eps,
    )
    # optional CLI overrides for EOT
    if args.eot_samples is not None:
        cfg.eot_samples = args.eot_samples
    if args.eot_scale is not None:
        cfg.eot_scale_jitter = args.eot_scale
    if args.eot_rot is not None:
        cfg.eot_rotate_deg = args.eot_rot
    if args.eot_color is not None:
        cfg.eot_color_jitter = args.eot_color
    return cfg


# ------------------------------------------------------------------
def make_condition_cfgs(base_cfg: AttackConfig) -> dict:
    """
    Build the ablation condition configs from a base FULL config.
    """
    small = base_cfg.patch_sizes[0]
    large = base_cfg.patch_sizes[-1]

    def clone(**over):
        d = base_cfg.__dict__.copy()
        d.update(over)
        return AttackConfig(**d)

    conds = {}

    # FULL (unchanged)
    conds["full"] = base_cfg

    # no_bandit -> random/eps grid
    conds["no_bandit"] = clone(use_contextual=False)

    # no_growth -> fixed smallest size
    conds["no_growth"] = clone(patch_sizes=(small,), grow_patience=10**9)  # effectively never grows

    # no_stealth -> disable stealth & print penalties
    conds["no_stealth"] = clone(w_stealth=0.0, w_print=0.0)

    # no_eot -> no EOT simulation
    conds["no_eot"] = clone(eot_samples=0, eot_scale_jitter=0.0, eot_rotate_deg=0.0, eot_color_jitter=0.0)

    # all_off -> no bandit, no growth, no stealth/print, no eot
    conds["all_off"] = clone(
        patch_sizes=(small,),
        use_contextual=False,
        w_stealth=0.0,
        w_print=0.0,
        eot_samples=0,
        eot_scale_jitter=0.0,
        eot_rotate_deg=0.0,
        eot_color_jitter=0.0,
        grow_patience=10**9,
    )

    return conds


# ------------------------------------------------------------------
def run_condition(
    cond_name: str,
    attack_cfg: AttackConfig,
    *,
    args,
    model,
    image_records,
    out_root: Path,
):
    """
    Run one ablation condition across all image_records.
    image_records: list of (img_tensor, boxes, labels, meta)
    """
    cond_dir = out_root / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)

    results = []
    csv_rows = []

    for i, (img_tensor, boxes, labels, img_meta) in enumerate(
        tqdm(image_records, desc=f"[{cond_name}] images", unit="img")
    ):
        print(f"\n[{cond_name.upper()}] IMAGE {i+1}/{len(image_records)} :: {img_meta}")
        print("[INFO] Ground truth classes in the image:")
        for lbl in labels.tolist():
            name = VOC_CLASSES[lbl] if 0 <= lbl < len(VOC_CLASSES) else "[UNKNOWN]"
            print(f" - {name}")

        # clean detector
        clean_cls_id, clean_conf, _ = clean_top_class(model, img_tensor)
        clean_name = VOC_CLASSES[clean_cls_id] if clean_cls_id >= 0 else "None"
        print(f"[CLEAN TOP] {clean_name} ({clean_conf:.3f})")

        # attack
        save_prefix = f"{cond_dir}/img{i:03d}"
        summary = generate_patch_for_image(
            img_tensor=img_tensor,
            boxes=boxes,
            labels=labels,
            model=model,
            mode=args.mode,
            attack_cfg=attack_cfg,
            save_prefix=save_prefix,
        )
        results.append({"img_index": i, "img_id": str(img_meta), **summary})

        # locate patched PNG
        # NB: summary["final_size"] may be None on total failure; fallback to last hist size
        final_size = summary.get("final_size")
        steps = summary.get("steps")
        patch_size = final_size if final_size is not None else 0

        # success-specific vs failure-specific filename
        if summary["success"] and final_size is not None and steps is not None:
            patched_path = f"{save_prefix}_succeeded_patch_{final_size}_step_{steps}_0.png"
        else:
            # fallback pattern; we don't know size/step reliably; glob last .png (not _patch.png)
            pngs = sorted([p for p in cond_dir.glob(f"img{i:03d}_*.png") if "_patch" not in p.name])
            patched_path = str(pngs[-1]) if pngs else ""

        # patched detector top class
        if patched_path and os.path.exists(patched_path):
            patched_cls_id, patched_conf = patched_top_class(model, patched_path)
            patched_name = VOC_CLASSES[patched_cls_id] if patched_cls_id >= 0 else "None"
        else:
            patched_cls_id, patched_conf, patched_name = -1, 0.0, "None"

        # patch area %
        _, H, W = img_tensor.shape
        patch_area_pct = (patch_size * patch_size) / (H * W) * 100.0 if patch_size > 0 else 0.0

        # location for visual diff → from last history row with matching size if possible
        cell = None
        hist = summary.get("history", [])
        if hist:
            # history tuples may vary in length; we expect cell at idx 5 or 4 depending on version
            # try to find last row with size == patch_size
            cand_rows = [h for h in hist if len(h) >= 6 and h[1] == patch_size]
            row = cand_rows[-1] if cand_rows else hist[-1]
            if len(row) >= 6:
                cell = row[5]
        if cell is None:
            cell = (0, 0)

        # visual diff
        if patched_path and os.path.exists(patched_path) and patch_size > 0:
            visual_diff = compute_visual_diff_patch(img_tensor, patched_path, cell, patch_size)
        else:
            visual_diff = float("nan")

        # row
        img_id_str = (
            img_meta.get("filename", [f"img{i:03d}"])[0] if isinstance(img_meta, dict) else f"img{i:03d}"
        )
        csv_rows.append({
            "condition": cond_name,
            "img_id": img_id_str,
            "success": int(bool(summary["success"])),
            "clean_top_class": clean_name,
            "clean_top_conf": clean_conf,
            "patched_top_class": patched_name,
            "patched_top_conf": patched_conf,
            "patch_size": patch_size if summary["success"] else "",
            "queries": summary["steps"],
            "patch_area_pct": patch_area_pct if patch_size > 0 else "",
            "visual_diff_patch": visual_diff if not np.isnan(visual_diff) else "",
            "patched_path": patched_path,
        })

    # save JSON
    out_json = cond_dir / "attack_summary.json"
    save_attack_summary({"results": results, "args": {**vars(args), "condition": cond_name}}, out_json.as_posix())

    # save CSV
    df = pd.DataFrame(csv_rows)
    csv_path = cond_dir / "attack_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[✓] Metrics CSV saved → {csv_path}")

    return df


# ------------------------------------------------------------------
def load_image_records(args, *, shuffle=False):
    """
    Load up to args.num images from the dataset into memory (for reproducible runs).
    Returns list of (img_tensor, boxes, labels, meta_dict).
    """
    loader = PascalVOCLoader(
        root=".",
        year="2007",
        image_set=args.image_set,
        inp_size=416,
        num_workers=2,
        shuffle=shuffle,
    )
    n = min(args.num, len(loader))
    recs = []
    for _ in range(n):
        recs.append(loader.next())  # expected tuple (img_tensor, boxes, labels, meta)
    return recs


# ------------------------------------------------------------------
def main():
    args = parse_args()
    seed_all(args.seed)

    # Train (if needed), then load YOLO

    if args.detector == "yolo":
        print("[INFO] YOLO model...")
        train_yolo(force_train=args.force_train)
        model = YOLO(MODEL_PATH)
    else:
        print("[INFO] Fastrcnn model...")
        train_faster_r_cnn(force_train=args.force_train)
        model = FasterRCNNWrapper(FRCNN_MODEL_PATH)  


    # Always use deterministic order for ablations
    image_records = load_image_records(args, shuffle=False)
    print(f"[INFO] Loaded {len(image_records)} images for evaluation.")

    # SINGLE-RUN MODE ------------------------------------------------
    if not args.ablation:
        print("[INFO] Single-run mode.")
        base_cfg = build_attack_cfg_from_args(args)
        out_root = Path(args.save_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        _ = run_condition(
            "single",
            base_cfg,
            args=args,
            model=model,
            image_records=image_records,
            out_root=out_root,
        )
        return

    # ABLATION MODE --------------------------------------------------
    print("[INFO] Ablation mode.")
    base_cfg = build_attack_cfg_from_args(args)
    cond_cfgs = make_condition_cfgs(base_cfg)

    out_root = Path(args.save_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dfs = []
    for cond_name, cfg in cond_cfgs.items():
        # each condition writes into <save-dir>/<cond_name>
        df = run_condition(
            cond_name,
            cfg,
            args=args,
            model=model,
            image_records=image_records,
            out_root=out_root,
        )
        dfs.append(df)

    # master CSV
    master = pd.concat(dfs, ignore_index=True)
    master_path = out_root / "ablation_master.csv"
    master.to_csv(master_path, index=False)
    print(f"[✓] Ablation master CSV saved → {master_path}")

    # quick summary table
    # convert numeric columns
    for col in ["success", "queries", "patch_area_pct", "clean_top_conf", "patched_top_conf", "visual_diff_patch"]:
        master[col] = pd.to_numeric(master[col], errors="coerce")
    g = master.groupby("condition").agg({
        "success": "mean",
        "queries": "mean",
        "patch_area_pct": "mean",
        "visual_diff_patch": "mean",
    }).reset_index()
    print("\n=== Ablation Summary (means) ===")
    print(g.to_string(index=False))
    print()


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
