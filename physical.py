# physical.py
"""
Adversarial patch demo on ONE image.
• Uses Ultralytics-YOLO to auto-detect a target box.
• Exports patched image + CSV + JSON metrics.
"""

import argparse, os, random, json
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from ultralytics import YOLO

# ---- your utilities ---------------------------------------------------------
from util         import train_yolo, generate_patch_for_image, save_attack_summary, MODEL_PATH
from patch_bandit import AttackConfig
from dataloader   import VOC_CLASSES          # list of 20 Pascal-VOC names
# -----------------------------------------------------------------------------


# ---------- helpers ----------------------------------------------------------
def seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def denorm_to_uint8(img_tensor):
    mean = torch.tensor([0.485,0.456,0.406], device=img_tensor.device).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=img_tensor.device).view(3,1,1)
    img  = (img_tensor*std+mean).permute(1,2,0).clamp(0,1).cpu().numpy()
    return (img*255).astype(np.uint8)

def yolo_first_box(model, img_tensor, conf_thres=0.25):
    """Run YOLO, return (boxes_xyxy, class_ids) *on the resized tensor*."""
    im_np  = denorm_to_uint8(img_tensor)
    result = model.predict(im_np, conf=conf_thres, verbose=False)[0].boxes
    if result is None or len(result)==0:
        return None, None
    return result.xyxy.cpu(), result.cls.cpu().long()

# ---------- CLI --------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to RGB image")
    ap.add_argument("--mode", choices=["digital","physical"], default="physical")
    ap.add_argument("--save-dir", default="results_single")
    # --- patch/bandit hyper-params (same defaults you had) ---
    ap.add_argument("--patch-sizes", default="120")
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--grow-patience", type=int, default=50)
    ap.add_argument("--min-improv", type=float, default=0.0)
    ap.add_argument("--max-queries", type=int, default=250)
    ap.add_argument("--w-det",    type=float, default=1.0)
    ap.add_argument("--w-stealth",type=float, default=0.5)
    ap.add_argument("--w-print",  type=float, default=0.25)
    ap.add_argument("--seed",     type=int,   default=20)
    return ap.parse_args()
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) make sure YOLO weights exist / train if needed
    train_yolo(force_train=False)
    model = YOLO(MODEL_PATH)

    # 2) read & normalise image exactly like before
    tfm = transforms.Compose([
        transforms.Resize((416,416)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img_pil     = Image.open(args.image).convert("RGB")
    img_tensor  = tfm(img_pil)          # CHW float

    # 3) get 1st YOLO detection
    boxes_xyxy, cls_ids = yolo_first_box(model, img_tensor)
    if boxes_xyxy is None:
        raise RuntimeError("YOLO found no objects in the image. Try another image!")

    # choose the highest-confidence detection (index 0 after Ultralytics sort)
    box   = boxes_xyxy[0].view(1,-1)          # shape (1,4)
    label = cls_ids[0].view(1)                # shape (1,)

    # 4) patch config
    sizes = tuple(int(s) for s in args.patch_sizes.split(",") if s.strip())
    attack_cfg = AttackConfig(
        patch_sizes   = sizes,
        grow_patience = args.grow_patience,
        min_improv    = args.min_improv,
        max_queries   = args.max_queries,
        w_det         = args.w_det,
        w_stealth     = args.w_stealth,
        w_print       = args.w_print,
        grid_size     = args.grid_size,
        use_contextual= False,
        save_dir      = args.save_dir,
    )

    # 5) run attack
    save_prefix = f"{args.save_dir}/img000"
    summary = generate_patch_for_image(
        img_tensor = img_tensor,
        boxes      = box,
        labels     = label,
        model      = model,
        mode       = args.mode,
        attack_cfg = attack_cfg,
        save_prefix= save_prefix,
    )

    # 6) build metrics row
    clean_cls  = VOC_CLASSES[label] if 0<=label<20 else str(int(label))
    csv_row = dict(
        img_id            = Path(args.image).name,
        success           = int(bool(summary["success"])),
        patch_size        = summary["final_size"],
        queries           = summary["steps"],
        clean_top_class   = clean_cls,
    )
    # 7) write JSON + CSV
    json_path = Path(args.save_dir)/"attack_summary.json"
    csv_path  = Path(args.save_dir)/"attack_metrics.csv"
    save_attack_summary({"results":[summary],"args":vars(args)}, json_path.as_posix())
    pd.DataFrame([csv_row]).to_csv(csv_path, index=False)
    print(f"[✓] Done!  Patched image + metrics saved to: {args.save_dir}")

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# # main.py
# """
# Entry point for adaptive adversarial patch experiments + CSV metrics export.
# """
# import argparse
# import os
# from pathlib import Path

# import torch
# import numpy as np
# import random
# import pandas as pd
# from PIL import Image
# import torchvision.transforms.functional as TF

# from util import train_yolo, generate_patch_for_image, save_attack_summary, MODEL_PATH
# from dataloader import PascalVOCLoader, VOC_CLASSES
# from ultralytics import YOLO
# from patch_bandit import AttackConfig
# from torchvision import transforms

# # ------------------------------------------------------------
# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--force-train", action="store_true")
#     ap.add_argument("--mode", choices=["digital", "physical"], default="physical")
#     ap.add_argument("--image-set", default="val")
#     ap.add_argument("--num", type=int, default=1, help="#images to attack from loader")
#     ap.add_argument("--grid-size", type=int, default=8)
#     ap.add_argument("--contextual", action="store_true", help="use contextual bandit")
#     ap.add_argument("--patch-sizes", type=str, default="40,60,80,120,160")
#     ap.add_argument("--grow-patience", type=int, default=50)
#     ap.add_argument("--min-improv", type=float, default=0.0)
#     ap.add_argument("--max-queries", type=int, default=250)
#     ap.add_argument("--w-det", type=float, default=1.0)
#     ap.add_argument("--w-stealth", type=float, default=0.5)
#     ap.add_argument("--w-print", type=float, default=0.25)
#     ap.add_argument("--save-dir", type=str, default="results")
#     ap.add_argument("--save-every", type=int, default=0)
#     ap.add_argument("--seed", type=int, default=0)
#     return ap.parse_args()


# # ------------------------------------------------------------
# def seed_all(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# # ------------------------------------------------------------
# def denorm_to_uint8(img_tensor):
#     """ImageNet unnorm CHW float->uint8 HWC."""
#     mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3,1,1)
#     std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3,1,1)
#     img_unnorm = (img_tensor * std + mean).permute(1,2,0).clamp(0,1).cpu().numpy()
#     return (img_unnorm * 255).astype(np.uint8)


# # ------------------------------------------------------------
# def clean_top_class(model, img_tensor):
#     """Return (cls_id, conf, per_cls_dict) for clean image."""
#     img_np = denorm_to_uint8(img_tensor)
#     pred = model.predict(source=img_np, conf=0.001, verbose=False)[0].boxes
#     if pred is None or len(pred) == 0:
#         return -1, 0.0, {}
#     cls_ids = pred.cls.cpu().numpy().astype(int)
#     confs   = pred.conf.cpu().numpy()
#     per_cls = {}
#     for c, s in zip(cls_ids, confs):
#         per_cls[c] = max(per_cls.get(c, 0.0), float(s))
#     top_cls = max(per_cls.items(), key=lambda x: x[1])[0]
#     return top_cls, per_cls[top_cls], per_cls


# # ------------------------------------------------------------
# def patched_top_class(model, patched_path):
#     """Run detector on saved patched PNG; return (cls_id, conf)."""
#     img = Image.open(patched_path).convert("RGB")
#     img_np = np.array(img)  # HWC uint8
#     pred = model.predict(source=img_np, conf=0.001, verbose=False)[0].boxes
#     if pred is None or len(pred) == 0:
#         return -1, 0.0
#     cls_ids = pred.cls.cpu().numpy().astype(int)
#     confs   = pred.conf.cpu().numpy()
#     idx = int(np.argmax(confs))
#     return int(cls_ids[idx]), float(confs[idx])


# # ------------------------------------------------------------
# def compute_visual_diff_patch(clean_img_tensor, patched_path, cell, patch_size):
#     """
#     Mean absolute diff in [0,1] pixel space between clean crop and saved patched crop.
#     """
#     # load patched png -> CHW torch in [0,1]
#     patched_px = TF.to_tensor(Image.open(patched_path).convert("RGB"))  # [0,1]
#     # denorm clean -> [0,1]
#     mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
#     std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
#     clean_px = (clean_img_tensor.cpu() * std + mean).clamp(0,1)
#     # crop
#     x, y = cell
#     ph = pw = patch_size
#     clean_crop = clean_px[:, y:y+ph, x:x+pw]
#     patched_crop = patched_px[:, y:y+ph, x:x+pw]
#     if clean_crop.numel() == 0 or patched_crop.numel() == 0:
#         return float("nan")
#     return torch.mean(torch.abs(clean_crop - patched_crop)).item()


# # ------------------------------------------------------------
# def main():
#     args = parse_args()
#     seed_all(args.seed)

#     train_yolo(force_train=args.force_train)
#     model = YOLO(MODEL_PATH)

#     # Load a hardcoded image
#     image_path = "./physical/b1.jpg"
#     img = Image.open(image_path).convert("RGB")

#     transform = transforms.Compose([
#         transforms.Resize((416, 416)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
#     ])
#     img_tensor = transform(img)

#     # Fake bounding box and label if required
#     # Format: [x1, y1, x2, y2] in image coordinates
#     boxes = torch.tensor([[50, 50, 300, 300]])  # dummy box
#     labels = torch.tensor([5])  # dummy class ID (e.g., 5 = 'bottle' in VOC)
#     img_meta = {"filename": [image_path]}

#     # Run once
#     i = 0


#     # Attack config
#     sizes = tuple(int(s) for s in args.patch_sizes.split(",") if s.strip())
#     attack_cfg = AttackConfig(
#         patch_sizes=sizes,
#         grow_patience=args.grow_patience,
#         min_improv=args.min_improv,
#         max_queries=args.max_queries,
#         w_det=args.w_det,
#         w_stealth=args.w_stealth,
#         w_print=args.w_print,
#         grid_size=args.grid_size,
#         use_contextual=args.contextual,
#         save_dir=args.save_dir,
#         save_every=args.save_every,
#     )

#     os.makedirs(args.save_dir, exist_ok=True)

#     results = []
#     csv_rows = []

#     print(f"\n[IMAGE {i+1}/{args.num}] {img_meta}")
#     print("[INFO] Ground truth classes in the image:")
#     for lbl in labels.tolist():
#         name = VOC_CLASSES[lbl] if 0 <= lbl < len(VOC_CLASSES) else "[UNKNOWN]"
#         print(f" - {name}")

#     # --- clean detector top class ---
#     clean_cls_id, clean_conf, clean_per_cls = clean_top_class(model, img_tensor)
#     clean_name = VOC_CLASSES[clean_cls_id] if clean_cls_id >= 0 else "None"
#     print(f"[CLEAN TOP] {clean_name} ({clean_conf:.3f})")

#     # run attack
#     save_prefix = f"{args.save_dir}/img{i:03d}"
#     summary = generate_patch_for_image(
#         img_tensor=img_tensor,
#         boxes=boxes,
#         labels=labels,
#         model=model,
#         mode=args.mode,
#         attack_cfg=attack_cfg,
#         save_prefix=save_prefix,
#     )
#     results.append({"img_index": i, "img_id": str(img_meta), **summary})

#     # --- locate saved patched PNG ---
#     if summary["success"]:
#         patched_path = f"{save_prefix}_succeeded_patch_{summary['final_size']}_step_{summary['steps']}_0.png"
#     else:
#         patched_path = f"{save_prefix}_failed_patch_{summary['final_size']}_step_{summary['steps']}_0.png"
#         if not os.path.exists(patched_path):
#             cand = sorted(Path(args.save_dir).glob(f"img{i:03d}_*_patch_*.png"))
#             if cand:
#                 patched_path = str(cand[-1])

#     # patched detector top class (independent check)
#     patched_cls_id, patched_conf = patched_top_class(model, patched_path)
#     patched_name = VOC_CLASSES[patched_cls_id] if patched_cls_id >= 0 else "None"

#     # patch area %
#     _, H, W = img_tensor.shape
#     patch_size = summary["final_size"] or 0
#     patch_area_pct = (patch_size * patch_size) / (H * W) * 100.0 if patch_size > 0 else 0.0

#     # get final cell from history if available (for visual diff)
#     cell = None
#     hist = summary.get("history", [])
#     if hist and len(hist[-1]) >= 6:
#         cell = hist[-1][5]

#     if cell is not None and patch_size > 0 and os.path.exists(patched_path):
#         visual_diff = compute_visual_diff_patch(img_tensor, patched_path, cell, patch_size)
#     else:
#         visual_diff = float("nan")

#     csv_rows.append({
#         "img_id": img_meta.get("filename", ["img%03d" % i])[0] if isinstance(img_meta, dict) else f"img{i:03d}",
#         "success": int(bool(summary["success"])),
#         "clean_top_class": clean_name,
#         "clean_top_conf": clean_conf,
#         "patched_top_class": patched_name,
#         "patched_top_conf": patched_conf,
#         "patch_size": patch_size if summary["success"] else "",
#         "queries": summary["steps"],
#         "patch_area_pct": patch_area_pct if patch_size > 0 else "",
#         "visual_diff_patch": visual_diff if not np.isnan(visual_diff) else "",
#         "patched_path": patched_path,
#     })


#     # save JSON (unchanged)
#     out_json = Path(args.save_dir) / "attack_summary.json"
#     save_attack_summary({"results": results, "args": vars(args)}, out_json.as_posix())

#     # save CSV
#     df = pd.DataFrame(csv_rows)
#     csv_path = Path(args.save_dir) / "attack_metrics.csv"
#     df.to_csv(csv_path, index=False)
#     print(f"[✓] Metrics CSV saved → {csv_path}")

#     # print dataset means (success rows only)
#     if any(df["success"] == 1):
#         df_succ = df[df["success"] == 1].copy()
#         # numeric conversion for safety
#         for col in ["queries", "patch_area_pct", "clean_top_conf", "patched_top_conf", "visual_diff_patch"]:
#             df_succ[col] = pd.to_numeric(df_succ[col], errors="coerce")
#         print("\n=== Means (success only) ===")
#         print(df_succ[["queries", "patch_area_pct", "visual_diff_patch"]].mean())
#     else:
#         print("\n[INFO] No successful attacks; no averages to report.")


# # ------------------------------------------------------------
# if __name__ == "__main__":
#     main()
