#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize PatchBandit attack_metrics.csv outputs – per-run and pooled.

Fixes the “everything shows as one ‘single’ run” issue by detecting the
true run-root (e.g. runsPB_full_ctx_strict_s0) instead of the nested
single/ directory.

The statistical computation part is identical to the script you posted.
Only helper functions that discover runs / assign run & group names have
changed.
"""

import os, re, argparse, csv, json, statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ═════════════════  constants (same as before) ═════════════════ #
DEFAULT_LADDER = (40, 60, 80, 120, 160, 192)
DEFAULT_INP    = 416


# ═════════════════  discovery helpers (NEW / FIXED) ════════════ #
def _has_metrics(d: Path) -> bool:
    """Return True if *d* already contains any metrics file."""
    return any([
        (d / "attack_metrics.csv").exists(),
        (d / "per_image_metrics.csv").exists(),
        bool(list(d.glob("attack_summary*.json")))
    ])


def _collect_run_dirs(roots: Sequence[Path]) -> List[Path]:
    """
    Return a list of run-root directories.

    • If the dir itself has metrics ⇒ keep it.
    • Else if a child dir *one level down* has metrics
      (typical ‘single/’ sub-dir) ⇒ keep the PARENT.
    • Else search deeper and map hits back to their grand-parent.
    """
    out: set[Path] = set()

    for root in roots:
        if not root.exists():
            print(f"[WARN] path not found: {root}")
            continue

        # case 1 – root already a run-root
        if _has_metrics(root):
            out.add(root)
            continue

        # case 2 – one-level children are the run dirs we want
        child_hits = [c for c in root.iterdir() if c.is_dir() and _has_metrics(c)]
        if child_hits:
            for c in child_hits:
                out.add(c)
            continue

        # case 3 – deeper paths: bring them back two levels up
        for csv_path in root.rglob("attack_metrics.csv"):
            # <run-root>/<maybe sub>/<single>/attack_metrics.csv
            # → run-root is **two** dirs up unless single/ is directly under root
            if csv_path.parent.parent.exists():
                out.add(csv_path.parent.parent)

    return sorted(out, key=str)


def _strip_seed(name: str) -> str:
    """
    Remove suffixes like *_s0*, *_s1*, *_seed42* so multiple seeds
    aggregate under one pooled group key.
    """
    for pat in (r"^(.*)_s\d+$", r"^(.*)_seed\d+$"):
        m = re.match(pat, name)
        if m:
            return m.group(1)
    return name


# ═════════════════  data-loading helpers (unchanged) ═══════════ #
_PATCH_RE = re.compile(r"_patch_(\d+)_")

def _safe_float(x, default=None):
    try:
        return float(x) if x not in (None, "") else default
    except Exception:
        return default

def _safe_int(x, default=None):
    try:
        return int(float(x)) if x not in (None, "") else default
    except Exception:
        return default


def _find_csvs(run_root: Path) -> List[Path]:
    """
    Return *all* attack_metrics.csv & per_image_metrics.csv files
    under this run-root (any depth).
    """
    csvs = []
    for fn in ("attack_metrics.csv", "per_image_metrics.csv"):
        csvs.extend(run_root.rglob(fn))
    return csvs


def _load_one_csv(csv_path: Path,
                  ladder: Sequence[int],
                  inp_size: int,
                  run_name: str,
                  group_name: str) -> pd.DataFrame:
    """
    Read one attack_metrics/per_image_metrics CSV and return a normalised DataFrame.
    """
    df = pd.read_csv(csv_path)

    # ---------- basic columns ------------------------------------------------
    if "success" not in df.columns:
        df["success"] = 0
    else:
        df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)

    if "patch_size" not in df.columns:
        df["patch_size"] = np.nan                        # raw size (success rows only)

    # ---------- effective patch size (snap from area %) ----------------------
    eff = df["patch_size"].copy()                        # start with raw

    if "patch_area_pct" in df.columns:
        # ladder areas (percent of image)
        ladder_pct = np.array([100.0 * (s * s) / (inp_size * inp_size)
                               for s in ladder], dtype=float)

        area_vals = df["patch_area_pct"].astype(float).to_numpy()
        snapped   = np.full_like(area_vals, np.nan, dtype=float)

        for i, a in enumerate(area_vals):
            if not np.isnan(a):
                snapped[i] = ladder[np.argmin(np.abs(a - ladder_pct))]

        # overwrite only rows where eff is NaN or 0
        mask = eff.isna() | (eff == 0)
        eff.loc[mask] = snapped[mask]

    df["patch_size_eff"] = eff.fillna(0).astype(int)

    # ---------- proxy score drop (if not already present) --------------------
    if "drop_proxy" not in df.columns:
        if {"clean_top_conf", "patched_top_conf"} <= set(df.columns):
            df["drop_proxy"] = (
                pd.to_numeric(df["clean_top_conf"],  errors="coerce").fillna(0) -
                pd.to_numeric(df["patched_top_conf"], errors="coerce").fillna(0)
            )
        else:
            df["drop_proxy"] = np.nan

    # ---------- metadata -----------------------------------------------------
    df["run"]   = run_name
    df["group"] = group_name
    return df

# def _load_one_csv(csv_path: Path, ladder, inp_size, run_name, group_name) -> pd.DataFrame:
#     df = pd.read_csv(csv_path)

#     # ------ normalisation identical to your reference script ------
#     if "success" not in df.columns:
#         df["success"] = 0
#     else:
#         df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)

#     # fill/rename columns, snap size from area %, etc.
#     # (copied verbatim from your reference for brevity)
#     # --------------------------------------------------------------
#     if "patch_size" not in df.columns:
#         df["patch_size"] = np.nan
#     if "patch_area_pct" in df.columns:
#         inp2 = float(inp_size * inp_size)
#         ladder_pct = np.array([100.0 * (s * s) / inp2 for s in ladder])
#         area_vals  = df["patch_area_pct"].values
#         snap = []
#         for a in area_vals:
#             if np.isnan(a):
#                 snap.append(np.nan)
#             else:
#                 snap.append(ladder[np.argmin(np.abs(a - ladder_pct))])
#         df["patch_size_eff"] = df["patch_size"].fillna(snap).fillna(0).astype(int)
#     else:
    #     df["patch_size_eff"] = df["patch_size"].fillna(0).astype(int)

    # # proxy drop
    # if "drop_proxy" not in df.columns:
    #     if "clean_top_conf" in df.columns and "patched_top_conf" in df.columns:
    #         df["drop_proxy"] = df["clean_top_conf"] - df["patched_top_conf"]
    #     else:
    #         df["drop_proxy"] = np.nan

    # # meta
    # df["run"]   = run_name
    # df["group"] = group_name
    # return df


# ═════════════════  summarisation (same as your script) ═════════ #
def _stats(s: pd.Series) -> Dict[str, float]:
    s = s.dropna()
    if s.empty:
        return {}
    return dict(
        median=float(s.median()),
        mean=float(s.mean()),
        p25=float(s.quantile(0.25)),
        p75=float(s.quantile(0.75))
    )


def _summarize(df: pd.DataFrame, key: str) -> Dict[str, Any]:
    g = {}
    g["n"]            = int(df.shape[0])
    g["success_pct"]  = float(df["success"].mean() * 100.0)
    g["drop_stats"]   = _stats(df["drop_proxy"])
    g["area_stats"]   = _stats(df["patch_area_pct"]) if "patch_area_pct" in df else {}
    g["query_stats"]  = _stats(df["queries"])        if "queries"        in df else {}
    g["patch_hist"]   = df["patch_size_eff"].value_counts().sort_index().to_dict()
    g["name"]         = key
    return g


def _print(g):
    print(f"\n----------------- {g['name']} -----------------")
    print(f"Images:              {g['n']}")
    print(f"Strict Success (%):  {g['success_pct']:.1f}")
    if g["drop_stats"]:
        print("Target drop:         median={median:.3f}  mean={mean:.3f}".format(**g["drop_stats"]))
    if g["area_stats"]:
        print("Patch area %:        median={median:.2f}  mean={mean:.2f}".format(**g["area_stats"]))
    if g["query_stats"]:
        print("Queries:             median={median:.1f}  mean={mean:.1f}".format(**g["query_stats"]))
    if g["patch_hist"]:
        print("Patch size counts:   " + ", ".join(f"{k}px:{v}" for k,v in g["patch_hist"].items()))
    print("------------------------------------------------")


# ═══════════════════════════════════════════════════════════════ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Run dirs (e.g. runsPB_*).")
    ap.add_argument("--ladder", default="40,60,80,120,160,192")
    ap.add_argument("--inp-size", type=int, default=DEFAULT_INP)
    ap.add_argument("--out", default=None, help="(optional) write merged CSV")
    args = ap.parse_args()

    ladder = tuple(int(x) for x in args.ladder.split(","))
    roots  = [Path(r).resolve() for r in args.roots]

    run_dirs = _collect_run_dirs(roots)
    if not run_dirs:
        print("[ERROR] No run directories found.")
        return
    print(f"[INFO] Found {len(run_dirs)} run directories.")

    per_run_rows = []
    group_rows   = []

    for rd in run_dirs:
        run_name   = rd.name                       # e.g. runsPB_full_ctx_strict_s0
        group_name = _strip_seed(run_name)         # e.g. runsPB_full_ctx_strict

        csv_files = _find_csvs(rd)
        if not csv_files:
            print(f"[WARN] no metrics in {rd}")
            continue

        run_df_parts = []
        for csv_path in csv_files:
            run_df_parts.append(
                _load_one_csv(csv_path, ladder, args.inp_size, run_name, group_name)
            )
        run_df = pd.concat(run_df_parts, ignore_index=True)
        per_run_rows.append(run_df)

        # ---------- PRINT PER-RUN SUMMARY ----------
        _print(_summarize(run_df, run_name))

    # ------------- pooled group summaries -------------
    if per_run_rows:
        all_df = pd.concat(per_run_rows, ignore_index=True)
        for gname, gdf in all_df.groupby("group"):
            _print(_summarize(gdf, gname))

        if args.out:
            all_df.to_csv(args.out, index=False)
            print(f"[INFO] wrote merged CSV → {args.out} (n={len(all_df)})")


if __name__ == "__main__":
    main()
