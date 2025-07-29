#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_map_drop.py  (robust v3)

Evaluate detector mAP on CLEAN vs PATCHED images from a PatchBandit run.

Clean images are taken from --clean-root (e.g., VOCYOLO/images/val).
Ground-truth boxes (for AP) are loaded from --voc-root (e.g., VOCdevkit/VOC2007).
If --no-map is given, AP is skipped and only detector runs complete.

Selection flags (priority order):
  --success-only  : rows w/ success==1 AND patch file
  --subset-only   : rows w/ patch file (ignore success)
  --impute-clean  : include all rows; if no patch file, use clean image
Default when no flags given: --subset-only.

Example (strict-success subset):
    python compute_map_drop.py \
      --clean-root VOCYOLO/images/val \
      --voc-root   VOCdevkit/VOC2007 \
      --weights    runs/detect/train5/weights/best.pt \
      --run-dir    runs10_full_ctx_strict \
      --success-only \
      --out-prefix full_ctx_success

Example (intention-to-treat; clean substituted where no patch):
    python compute_map_drop.py \
      --clean-root VOCYOLO/images/val \
      --voc-root   VOCdevkit/VOC2007 \
      --weights    runs/detect/train5/weights/best.pt \
      --run-dir    runs10_full_ctx_strict \
      --impute-clean \
      --out-prefix full_ctx_all
"""

import argparse, json, sys, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Ultralytics YOLO (v8-style)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ---------------------------- VOC metadata ----------------------------
VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
]
VOC2IDX = {c:i for i,c in enumerate(VOC_CLASSES)}

# ------------------------- load VOC annotation ------------------------
def load_voc_anno(voc_root: Path, img_id: str) -> np.ndarray:
    """Return [N,5]: xmin,ymin,xmax,ymax,cls_idx for VOC img_id."""
    stem = Path(img_id).stem
    xml_path = voc_root / "Annotations" / f"{stem}.xml"
    if not xml_path.is_file():
        return np.zeros((0,5),float)
    root = ET.parse(str(xml_path)).getroot()
    rows=[]
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in VOC2IDX: 
            continue
        cls = VOC2IDX[name]
        bb  = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        rows.append((xmin,ymin,xmax,ymax,cls))
    return np.asarray(rows,float) if rows else np.zeros((0,5),float)

# ------------------------------- IoU/AP -------------------------------
def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size==0 or b.size==0:
        return np.zeros((a.shape[0],b.shape[0]),float)
    lt = np.maximum(a[:,None,:2], b[None,:,:2])
    rb = np.minimum(a[:,None,2:], b[None,:,2:])
    wh = np.clip(rb-lt,0,None)
    inter = wh[:,:,0]*wh[:,:,1]
    area_a = (a[:,2]-a[:,0])*(a[:,3]-a[:,1])
    area_b = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    union = area_a[:,None]+area_b[None,:]-inter
    return inter/np.clip(union,1e-9,None)

def voc_ap_11pt(rec,prec):
    ap=0.0
    for thr in np.linspace(0,1,11):
        p = prec[rec>=thr]
        ap += (p.max() if p.size else 0.0)/11.0
    return ap

def eval_class_ap(preds, gts, cls_idx, iou_thr=0.5):
    gt_boxes={}; npos=0
    for gid,arr in gts.items():
        sel = arr[arr[:,4]==cls_idx]
        gt_boxes[gid]=sel[:,:4]
        npos += sel.shape[0]
    if npos==0:
        return float("nan"),0,0
    if not preds:
        return 0.0,npos,0
    preds_sorted = sorted(preds,key=lambda x:x[1],reverse=True)
    tp=np.zeros(len(preds_sorted)); fp=np.zeros(len(preds_sorted))
    used={gid:np.zeros(gt_boxes[gid].shape[0],bool) for gid in gt_boxes}
    for i,(gid,score,box) in enumerate(preds_sorted):
        g=gt_boxes.get(gid,np.zeros((0,4),float))
        if g.shape[0]==0:
            fp[i]=1; continue
        ious=box_iou_xyxy(box[None,:],g)[0]
        j=ious.argmax()
        if ious[j]>=iou_thr and not used[gid][j]:
            tp[i]=1; used[gid][j]=True
        else:
            fp[i]=1
    tp_cum=np.cumsum(tp); fp_cum=np.cumsum(fp)
    rec = tp_cum/float(npos)
    prec= tp_cum/np.maximum(tp_cum+fp_cum,1e-9)
    return voc_ap_11pt(rec,prec), npos, len(preds_sorted)

# ------------------------- locate attack CSV -------------------------
def find_attack_csv(run_dir: Path) -> Path:
    c = run_dir/"single"/"attack_metrics.csv"
    if c.is_file(): return c
    c = run_dir/"attack_metrics.csv"
    if c.is_file(): return c
    found = list(run_dir.rglob("attack_metrics.csv"))
    if not found:
        sys.exit(f"[ERR] no attack_metrics.csv under {run_dir}")
    return found[0]

# ----------------------------- load CSV ------------------------------
def load_run_csv(run_dir: Path) -> Tuple[pd.DataFrame, Path]:
    csv_path = find_attack_csv(run_dir)
    df = pd.read_csv(csv_path)

    # normalise patched_top_class safely
    if "patched_top_class" in df.columns:
        df["patched_top_class"] = (
            df["patched_top_class"].astype(str).str.strip()
              .replace(["nan","NaN","","none"], "None")
        )
    else:
        df["patched_top_class"]="None"

    # numeric coercions
    for c in ["success","clean_top_conf","patched_top_conf","patch_size",
              "queries","patch_area_pct","visual_diff_patch"]:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors="coerce")
        else:
            df[c]=np.nan
    df["success"]=df["success"].fillna(0).astype(int)

    # VOC key
    if "img_id" in df.columns:
        df["img_key"]=df["img_id"].astype(str)
    elif "image_id" in df.columns:
        df["img_key"]=df["image_id"].astype(str)
    else:
        df["img_key"]=df.index.map(lambda i:f"idx_{i}.jpg")

    if "patched_path" not in df.columns:
        df["patched_path"]=""

    return df, csv_path.parent

# ---------------------- scan for patched scenes ----------------------
# Scene images: imgNNN_{succeeded|failed}_patch_SIZE_step_*.(png|jpg)
# Texture-only images end with '_patch.png' AFTER the step segment; skip.
SCENE_RE = re.compile(
    r"^img(?P<idx>\d+)_(?:succeeded|failed)_patch_(?P<size>\d+)_step_.*\.(?:png|jpg|jpeg)$",
    re.IGNORECASE,
)
TEXTURE_SUFFIX_RE = re.compile(r"_patch\.(?:png|jpg|jpeg)$", re.IGNORECASE)

def scan_scene_images(run_dir: Path) -> Dict[int, Path]:
    out={}
    for sub in [run_dir/"single", run_dir]:
        if not sub.is_dir(): continue
        for p in sub.iterdir():
            if not p.is_file(): continue
            n=p.name
            if TEXTURE_SUFFIX_RE.search(n):  # skip patch texture tiles
                continue
            m=SCENE_RE.match(n)
            if m:
                out[int(m.group("idx"))]=p.resolve()
    return out

# ---------------------- resolve patched paths ------------------------
def resolve_patch_paths(run_dir: Path, base_dir: Path, df: pd.DataFrame) -> List[Optional[Path]]:
    scene_map = scan_scene_images(run_dir)
    resolved=[]
    for idx,row in df.iterrows():
        # CSV entry?
        pstr=str(row.get("patched_path","")).strip()
        if pstr and pstr.lower() not in ("nan","none"):
            p=Path(pstr)
            if not p.is_absolute():
                p=(base_dir/p).resolve()
            if p.is_file():
                resolved.append(p); continue
        # fallback by row index
        if idx in scene_map:
            resolved.append(scene_map[idx]); continue
        resolved.append(None)
    return resolved

# --------------------------- selection set ---------------------------
def select_rows(df, patched, success_only, subset_only, impute_clean):
    rows=[]
    for img_key,succ,ppath in zip(df["img_key"], df["success"], patched):
        has_patch = ppath is not None and ppath.is_file()
        if success_only:
            if succ==1 and has_patch:
                rows.append((img_key, ppath, True))
            continue
        if subset_only:
            if has_patch:
                rows.append((img_key, ppath, True))
            continue
        if impute_clean:
            rows.append((img_key, ppath if has_patch else None, has_patch))
            continue
        # fallback => subset
        if has_patch:
            rows.append((img_key, ppath, True))
    return rows

# --------------------------- clean image -----------------------------
def find_clean(img_key: str, clean_root: Path) -> Optional[Path]:
    """Return path to clean image under clean_root (tries common extensions)."""
    p = clean_root / img_key
    if p.is_file(): return p
    stem = Path(img_key).stem
    for ext in (".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"):
        q = clean_root / f"{stem}{ext}"
        if q.is_file(): return q
    return None

# --------------------------- run detector ----------------------------
def run_detector(model, pairs: List[Tuple[Path,str]], conf_thr: float) -> Dict[str,np.ndarray]:
    out={}
    for img_path, gid in tqdm(pairs, desc="det"):
        r = model.predict(source=str(img_path), conf=conf_thr, verbose=False)[0]
        boxes = r.boxes
        if boxes is None or boxes.shape[0]==0:
            det = np.zeros((0,6),float)
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1,1)
            cls  = boxes.cls.cpu().numpy().reshape(-1,1)
            det  = np.hstack([xyxy, conf, cls])
        out[gid]=det
    return out

# ----------------------------- eval mAP ------------------------------
def eval_map(model, voc_root: Path, pairs, conf_thr=0.001, iou_thr=0.5) -> Dict[str,float]:
    gts={gid:load_voc_anno(voc_root,gid) for (_,gid) in pairs}
    dets=run_detector(model,pairs,conf_thr)
    aps={}
    for ci,cls_name in enumerate(VOC_CLASSES):
        preds=[]
        for gid,det in dets.items():
            if det.shape[0]==0: continue
            mask=det[:,5]==ci
            for row in det[mask]:
                preds.append((gid, float(row[4]), row[:4]))
        ap,_,_ = eval_class_ap(preds,gts,ci,iou_thr=iou_thr)
        aps[cls_name]=ap
    vals=[v for v in aps.values() if not np.isnan(v)]
    aps["_mAP"]=float(np.mean(vals)) if vals else float("nan")
    return aps

# ------------------------------- main --------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-root", required=True,
                    help="Dir containing CLEAN images (e.g., VOCYOLO/images/val).")
    ap.add_argument("--voc-root",   default=None,
                    help="VOC root w/ Annotations/ for GT (optional; needed for mAP).")
    ap.add_argument("--weights",    required=True, help="YOLO weights .pt")
    ap.add_argument("--run-dir",    required=True, help="PatchBandit run directory")
    ap.add_argument("--out-prefix", required=True, help="Prefix for JSON/CSV output")
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--subset-only",  action="store_true")
    ap.add_argument("--impute-clean", action="store_true")
    ap.add_argument("--no-map",       action="store_true", help="Skip AP computation; just run detector.")
    ap.add_argument("--num", type=int, default=None, help="debug limit")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou",  type=float, default=0.5)
    args=ap.parse_args()

    if YOLO is None:
        sys.exit("[ERR] ultralytics not installed; pip install ultralytics")

    clean_root=Path(args.clean_root).resolve()
    if not clean_root.is_dir():
        sys.exit(f"[ERR] clean-root not found: {clean_root}")

    voc_root=Path(args.voc_root).resolve() if args.voc_root else None
    if not args.no_map and (voc_root is None or not (voc_root/"Annotations").is_dir()):
        sys.exit("[ERR] --voc-root required (with Annotations/) unless --no-map is set.")

    run_dir=Path(args.run_dir).resolve()
    df, csv_base = load_run_csv(run_dir)
    print(f"[INFO] loading run CSV: {run_dir}/...  ({len(df)} rows)")

    patched_paths = resolve_patch_paths(run_dir, csv_base, df)
    have_patch = sum(p is not None for p in patched_paths)
    print(f"[INFO] resolved patched scene paths for {have_patch}/{len(patched_paths)} rows")

    # selection precedence
    success_only=args.success_only
    subset_only=args.subset_only if not success_only else False
    impute_clean=args.impute_clean if not (success_only or subset_only) else False
    if not (success_only or subset_only or impute_clean):
        subset_only=True

    sel = select_rows(df, patched_paths, success_only, subset_only, impute_clean)
    if args.num is not None and len(sel)>args.num:
        sel = sel[:args.num]

    if not sel:
        sys.exit("[ERR] no images selected; check flags & run contents.")

    print(f"[INFO] selected {len(sel)} rows (success_only={success_only}, subset_only={subset_only}, impute_clean={impute_clean})")

    # build CLEAN + PATCHED pairs
    clean_pairs=[]; patch_pairs=[]
    skipped=0
    for img_key, ppath, use_patch in sel:
        clean_p = find_clean(img_key, clean_root)
        if clean_p is None:
            print(f"[WARN] missing clean image for {img_key}; skipping row")
            skipped+=1
            continue
        clean_pairs.append((clean_p, img_key))
        if use_patch and ppath is not None and ppath.is_file():
            patch_pairs.append((ppath, img_key))
        else:
            # impute or fallback
            patch_pairs.append((clean_p, img_key))

    if skipped:
        print(f"[WARN] skipped {skipped} rows with missing clean images.")
    if not clean_pairs:
        sys.exit("[ERR] after filtering, no valid clean image paths.")

    model=YOLO(str(args.weights))

    # Evaluate: always run CLEAN & PATCHED detection; AP only if requested
    print("[INFO] running detector on CLEAN images...")
    det_clean = run_detector(model, clean_pairs, conf_thr=args.conf)

    print("[INFO] running detector on PATCHED images...")
    det_patch = run_detector(model, patch_pairs, conf_thr=args.conf)

    if args.no_map:
        # quick numeric summary: mean #boxes, mean conf drop of top GT class?
        n_clean = np.mean([d.shape[0] for d in det_clean.values()])
        n_patch = np.mean([d.shape[0] for d in det_patch.values()])
        print(f"[INFO] mean boxes clean: {n_clean:.2f} | patched: {n_patch:.2f}")
        mAPc=mAPp=delta=float("nan")
        aps_clean={c:float("nan") for c in VOC_CLASSES}
        aps_patch=aps_clean.copy()
    else:
        print("[INFO] computing mAP (VOC07 11-pt)...")
        # reuse eval_map but feed cached dets to avoid re-inference?
        # Simplicity: call eval_map() which will re-run quickly (images cached by OS).
        aps_clean=eval_map(model,voc_root,clean_pairs,conf_thr=args.conf,iou_thr=args.iou)
        aps_patch=eval_map(model,voc_root,patch_pairs,conf_thr=args.conf,iou_thr=args.iou)
        mAPc=aps_clean["_mAP"]; mAPp=aps_patch["_mAP"]; delta=mAPc-mAPp

        print("\n== mAP summary ==")
        print(f"mAP@{args.iou:.2f} clean : {mAPc:.3f}")
        print(f"mAP@{args.iou:.2f} patched: {mAPp:.3f}")
        print(f"Delta (clean - patched): {delta:.3f}")

    # Write JSON + CSV
    outp=Path(args.out_prefix)
    with open(outp.with_suffix(".map.json"),"w") as f:
        json.dump({
            "n_images":len(clean_pairs),
            "mAP_clean":mAPc,
            "mAP_patched":mAPp,
            "delta":delta,
            "aps_clean":{k:float(v) for k,v in aps_clean.items()},
            "aps_patched":{k:float(v) for k,v in aps_patch.items()},
            "selection":{
                "success_only":success_only,
                "subset_only":subset_only,
                "impute_clean":impute_clean,
                "no_map":args.no_map,
            },
        },f,indent=2)
    print(f"[INFO] wrote {outp.with_suffix('.map.json')}")

    rows=[]
    for cls in VOC_CLASSES:
        rows.append({
            "class":cls,
            "ap_clean":aps_clean.get(cls,np.nan),
            "ap_patch":aps_patch.get(cls,np.nan),
            "ap_drop":aps_clean.get(cls,0.0)-aps_patch.get(cls,0.0),
        })
    rows.append({"class":"mAP","ap_clean":mAPc,"ap_patch":mAPp,"ap_drop":delta})
    pd.DataFrame(rows).to_csv(outp.with_suffix(".map.csv"),index=False)
    print(f"[INFO] wrote {outp.with_suffix('.map.csv')}")

if __name__=="__main__":
    main()
