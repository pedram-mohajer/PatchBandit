#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
size_usage_stats.py

Summarize *final* patch sizes from a PatchBandit run and compute
cumulative strict success vs. maximum allowed size (as in Fig.~size_cum).

This script uses the run's attack_metrics.csv:
  - success column (0/1)
  - patch_size column (px) is logged for successes only
  - patch_area_pct (optional; reported median per rung)
Failures typically have blank patch_size; we exclude them from the
per-rung counts but include them in denominators for cumulative success.

Outputs:
  • Terminal summary (counts, % of total, % of successes, cumulative %).
  • Optional CSV (--out) with the same data.
  • Optional bar plot (--plot) for quick inspection (PNG/PDF).

Usage:
    python size_usage_stats.py --run-dir runs10_full_ctx_strict \
        --ladder 40,60,80,120,160,192 \
        --out size_usage_full_ctx.csv \
        --plot size_usage_full_ctx.png
"""

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def find_attack_csv(run_dir: Path) -> Path:
    for cand in [run_dir/"single"/"attack_metrics.csv",
                 run_dir/"attack_metrics.csv"]:
        if cand.is_file():
            return cand
    found = list(run_dir.rglob("attack_metrics.csv"))
    if not found:
        sys.exit(f"[ERR] no attack_metrics.csv under {run_dir}")
    return found[0]


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # canonical columns we use
    if "success" not in df.columns:
        sys.exit("[ERR] CSV missing 'success' column.")
    if "patch_size" not in df.columns:
        df["patch_size"] = np.nan
    # numeric coercions
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)
    df["patch_size"] = pd.to_numeric(df["patch_size"], errors="coerce")
    if "patch_area_pct" in df.columns:
        df["patch_area_pct"] = pd.to_numeric(df["patch_area_pct"], errors="coerce")
    else:
        df["patch_area_pct"] = np.nan
    return df


def summarize(df: pd.DataFrame, ladder):
    n_total = len(df)
    df_succ = df[df["success"] == 1].copy()
    n_succ = len(df_succ)

    # counts by patch_size among successes
    # (ignore NaNs; if any success row lacks size, drop it)
    df_succ = df_succ.dropna(subset=["patch_size"])
    df_succ["patch_size"] = df_succ["patch_size"].astype(int)

    counts = df_succ["patch_size"].value_counts().sort_index()
    # ensure all ladder rungs appear (0 if none)
    counts = counts.reindex(ladder, fill_value=0)

    # % of total dataset
    pct_total = counts.values / float(n_total) * 100.0
    # % of successes
    pct_succ = counts.values / float(n_succ) * 100.0 if n_succ > 0 else np.nan

    # cumulative success up to each rung (as % of total)
    cum_counts = counts.cumsum().values
    cum_pct_total = cum_counts / float(n_total) * 100.0

    # median area per rung (successes)
    med_area = []
    for s in ladder:
        ss = df_succ[df_succ["patch_size"] == s]["patch_area_pct"].dropna()
        med_area.append(float(ss.median()) if not ss.empty else np.nan)

    out_df = pd.DataFrame({
        "patch_size": ladder,
        "n_success": counts.values,
        "pct_total": pct_total,
        "pct_of_successes": pct_succ,
        "cum_n_success": cum_counts,
        "cum_pct_total": cum_pct_total,
        "median_area_pct": med_area,
    })
    return out_df, n_total, n_succ


def plot_cum(df_stats: pd.DataFrame, out_path: Path, ymax=100):
    # bar plot of cumulative strict success (% total) vs rung
    sizes = df_stats["patch_size"].astype(str).values
    vals = df_stats["cum_pct_total"].values

    fig, ax = plt.subplots(figsize=(4.0, 1.8))  # ~one-column friendly
    ax.bar(sizes, vals)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Max Size ≤ (px)")
    ax.set_ylabel("Strict Success (%)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    # annotate bars
    for i,v in enumerate(vals):
        ax.text(i, v+1, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ladder", default="40,60,80,120,160,192",
                    help="Comma-separated patch sizes (px) in ladder order.")
    ap.add_argument("--out", default=None,
                    help="Optional CSV path to write summary.")
    ap.add_argument("--plot", default=None,
                    help="Optional PNG/PDF path for cumulative success bar plot.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    csv_path = find_attack_csv(run_dir)
    df = load_df(csv_path)

    ladder = [int(x) for x in args.ladder.split(",") if x.strip()]

    stats_df, n_total, n_succ = summarize(df, ladder)

    # ---- print summary ----
    print(f"\n[run] {run_dir.name}")
    print(f"Total images      : {n_total}")
    print(f"Strict successes  : {n_succ} ({n_succ/n_total*100:.1f}%)\n")

    print("Per-rung counts (successes only):")
    for _,row in stats_df.iterrows():
        s = int(row.patch_size)
        n = int(row.n_success)
        pt= row.pct_total
        ps= row.pct_of_successes
        ma= row.median_area_pct
        print(f"  {s:>3d}px : n={n:3d}  ({pt:5.1f}% total | {ps:5.1f}% of succ)  med area={ma:5.2f}%")

    print("\nCumulative strict success (% of total) if ladder truncated at rung:")
    for _,row in stats_df.iterrows():
        s = int(row.patch_size)
        cp = row.cum_pct_total
        print(f"  ≤{s:>3d}px : {cp:5.1f}%")

    # ---- write CSV? ----
    if args.out:
        out_path = Path(args.out)
        stats_df.to_csv(out_path, index=False)
        print(f"[INFO] wrote {out_path}")

    # ---- plot? ----
    if args.plot:
        plot_cum(stats_df, Path(args.plot))
        print(f"[INFO] wrote plot {args.plot}")


if __name__ == "__main__":
    main()
