#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_qual_figs.py

Build side-by-side clean / patched figure panels from attack_metrics.csv.

Each panel shows:
  - Clean image
  - Patched image (if available)
  - Text overlays: clean_top_class/conf ; patched_top_class/conf ; patch_size ; queries

You can choose N examples (random or by filter: success only, fails only, class match, etc.)

Usage:
  python make_qual_figs.py \
      --voc-root VOCdevkit/VOC2007 \
      --run-dir runs10_full_ctx_strict \
      --out-dir figs_full_ctx \
      --num 16 \
      --success-only

"""

import argparse, os, random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Optional: try to load a default truetype font; fallback to PIL
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 12)
except Exception:
    FONT = ImageFont.load_default()

def load_csv(run_dir: Path) -> pd.DataFrame:
    hits = list(run_dir.rglob("attack_metrics.csv"))
    if not hits:
        raise FileNotFoundError(f"No attack_metrics.csv under {run_dir}")
    df = pd.read_csv(hits[0])
    df["img_key"] = df["img_id"].str.strip()
    # success int
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)
    return df

def voc_img_path(voc_root: Path, img_id: str) -> Path:
    return voc_root / "JPEGImages" / img_id

def annotate(im: Image.Image, text: str, color=(255,255,0)):
    draw = ImageDraw.Draw(im)
    draw.rectangle([(0,0),(im.width,14)], fill=(0,0,0,128))
    draw.text((2,2), text, font=FONT, fill=color)
    return im

def make_panel(clean_path: Path,
               patched_path: Path,
               clean_lbl: str,
               patched_lbl: str,
               out_path: Path,
               width: int=640):
    # load
    clean = Image.open(clean_path).convert("RGB")
    patched = Image.open(patched_path).convert("RGB") if patched_path and patched_path.exists() else None

    # resize to common height
    h = min(clean.height, patched.height if patched else clean.height)
    def resize_keep(im):
        scale = h / im.height
        return im.resize((int(im.width*scale), h), Image.BILINEAR)

    clean_r = resize_keep(clean)
    clean_r = annotate(clean_r, clean_lbl)
    if patched:
        patched_r = resize_keep(patched)
        patched_r = annotate(patched_r, patched_lbl, color=(255,128,0))
        combo = Image.new("RGB", (clean_r.width + patched_r.width, h))
        combo.paste(clean_r, (0,0))
        combo.paste(patched_r, (clean_r.width,0))
    else:
        combo = clean_r
    combo.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num", type=int, default=16,
                    help="Number of panels to produce.")
    ap.add_argument("--success-only", action="store_true",
                    help="Only include successful attacks.")
    ap.add_argument("--fails-only", action="store_true",
                    help="Only include failures.")
    ap.add_argument("--class-filter", default=None,
                    help="Comma-separated clean_top_class names to include.")
    args = ap.parse_args()

    voc_root = Path(args.voc_root)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(run_dir)

    if args.success_only and args.fails_only:
        raise ValueError("Choose at most one of --success-only or --fails-only.")

    if args.success_only:
        df = df[df["success"]==1]
    elif args.fails_only:
        df = df[df["success"]==0]

    if args.class_filter:
        keep = {c.strip() for c in args.class_filter.split(",")}
        df = df[df["clean_top_class"].isin(keep)]

    if df.empty:
        print("[WARN] no rows match filters; nothing to do.")
        return

    rows = df.sample(n=min(args.num, len(df)), random_state=0)

    for i,row in enumerate(rows.itertuples()):
        clean_path = voc_img_path(voc_root, row.img_id)
        patched_path = Path(row.patched_path) if isinstance(row.patched_path, str) and len(row.patched_path) > 0 else None
        clean_lbl = f"clean: {row.clean_top_class} ({row.clean_top_conf:.2f})"
        if patched_path:
            patched_lbl = f"patched: {row.patched_top_class} ({row.patched_top_conf:.2f})  sz={row.patch_size}px  q={row.queries}"
        else:
            patched_lbl = f"no patched image  q={row.queries}"
        out_path = out_dir / f"panel_{i:03d}_{Path(row.img_id).stem}.jpg"
        try:
            make_panel(clean_path, patched_path, clean_lbl, patched_lbl, out_path)
        except Exception as e:
            print(f"[ERR] {row.img_id}: {e}")

    print(f"[INFO] wrote panels to {out_dir}")

if __name__ == "__main__":
    main()
