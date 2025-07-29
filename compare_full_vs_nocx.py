#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_full_vs_nocx.py  (robust v2)

Pairwise comparison of PatchBandit runs:
  * Full contextual (bandit+NES+growth)
  * NoCtx (non-contextual / epsilon-greedy)
  * Random (pure uniform placement -- optional)
Prints per-run summaries and writes a merged CSV (if >=2 runs found).

Usage:
    python compare_full_vs_nocx.py \
        --full-dir runs10_full_ctx_strict \
        --nocx-dir runs10_nocx_strict \
        --rand-dir runs10_random_place \
        --out pairwise_full_nocx.csv
"""

import sys, argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

CSV_NAME = "attack_metrics.csv"

# ------------------ locate CSV ------------------
def find_csv(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None
    direct = run_dir / CSV_NAME
    if direct.is_file():
        return direct
    for sub in ["single","strict","fast","results"]:
        p = run_dir/sub/CSV_NAME
        if p.is_file():
            return p
    found = list(run_dir.rglob(CSV_NAME))
    return found[0] if found else None

# ------------------ load CSV --------------------
def load_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = find_csv(run_dir)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)

    if "patched_top_class" in df.columns:
        df["patched_top_class"] = (
            df["patched_top_class"].astype(str).str.strip()
              .replace(["nan","NaN","","none"], "None")
        )
    else:
        df["patched_top_class"] = "None"

    for c in ["success","clean_top_conf","patched_top_conf","patch_size",
              "queries","patch_area_pct","visual_diff_patch"]:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors="coerce")
        else:
            df[c]=np.nan
    df["success"]=df["success"].fillna(0).astype(int)

    # drop_proxy: if class unchanged, use patched_top_conf; else 0
    df["target_conf_proxy"] = np.where(
        df["patched_top_class"] == df.get("clean_top_class","None"),
        df["patched_top_conf"].fillna(0.0),
        0.0,
    )
    df["drop_proxy"] = df["clean_top_conf"].fillna(0.0) - df["target_conf_proxy"]
    df["drop_raw"]   = df["clean_top_conf"].fillna(0.0) - df["patched_top_conf"].fillna(0.0)

    if "img_id" in df.columns:
        df["img_key"]=df["img_id"].astype(str)
    elif "image_id" in df.columns:
        df["img_key"]=df["image_id"].astype(str)
    else:
        df["img_key"]=df.index.astype(str)

    return df

# ------------------ summary print ----------------
def print_overall(label, df):
    n=len(df)
    succ = df["success"].mean()*100.0 if "success" in df else float("nan")
    drop_med = df["drop_proxy"].median()
    drop_raw = df["drop_raw"].median()
    area_med = df["patch_area_pct"].median()
    q_med    = df["queries"].median()
    print(f"\n== {label} overall ==")
    print(f"Images: {n}")
    if not np.isnan(succ):
        print(f"Strict success: {succ:.1f}%")
    print(f"Median target drop (proxy): {drop_med:.3f}")
    print(f"Median target drop (raw):   {drop_raw:.3f}")
    print(f"Median area %: {area_med:.2f}")
    print(f"Median queries: {q_med:.1f}")

# ------------------ pairwise merge ----------------
def merge_pair(df_a, df_b, name_a, name_b):
    cols=["img_key","success","drop_proxy","patch_area_pct","queries","patch_size"]
    a=df_a[cols].copy(); b=df_b[cols].copy()
    a.columns=[c if c=="img_key" else f"{name_a}_{c}" for c in a.columns]
    b.columns=[c if c=="img_key" else f"{name_b}_{c}" for c in b.columns]
    return pd.merge(a,b,on="img_key",how="outer",suffixes=("",""))

# ------------------ main -------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--full-dir", required=True)
    ap.add_argument("--nocx-dir", default=None)
    ap.add_argument("--rand-dir", default=None)
    ap.add_argument("--out", default="pairwise_full_nocx.csv")
    args=ap.parse_args()

    print(f"[INFO] Full: loading {args.full_dir}/...")
    full_df = load_csv(Path(args.full_dir))
    if full_df is None:
        sys.exit("[ERR] full run missing attack_metrics.csv; abort.")

    nocx_df = load_csv(Path(args.nocx_dir)) if args.nocx_dir else None
    if args.nocx_dir and nocx_df is None:
        print(f"[WARN] NoCtx: no attack_metrics.csv found under {args.nocx_dir} (skipping).")

    rand_df = load_csv(Path(args.rand_dir)) if args.rand_dir else None
    if args.rand_dir and rand_df is None:
        print(f"[WARN] Random: no attack_metrics.csv found under {args.rand_dir} (skipping).")

    # always print full
    print_overall("Full", full_df)

    frames=[]
    if nocx_df is not None:
        print_overall("NoCtx", nocx_df)
        frames.append(merge_pair(full_df, nocx_df, "full","nocx"))
    if rand_df is not None:
        print_overall("Random", rand_df)
        frames.append(merge_pair(full_df, rand_df, "full","rand"))

    if frames:
        merged=frames[0]
        for extra in frames[1:]:
            merged = pd.merge(merged,extra,on="img_key",how="outer",suffixes=("",""))
        merged.to_csv(args.out,index=False)
        print(f"[INFO] wrote merged CSV: {args.out} ({len(merged)} rows)")
    else:
        print("[WARN] no pairwise merges written (only full available).")

if __name__=="__main__":
    main()
