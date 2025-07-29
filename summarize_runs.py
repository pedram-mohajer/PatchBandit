#!/usr/bin/env python
"""
summarize_runs.py   ––  Aggregate PatchBandit experiment outputs into paper-ready
summary statistics.

Usage examples
--------------
python summarize_runs.py --roots runs10_nocx_strict runs10_fixed80
python summarize_runs.py --roots runs_batch_20250716_123456  # recurse
"""

import os, re, glob, json, csv, argparse, statistics
from pathlib import Path
from typing import Dict, List, Any, Optional

# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════
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

def _group_name(run_dir_name: str) -> str:
    """
    Strip trailing `_s<seed>` or `_seed<seed>` so multi-seed runs are grouped.
    """
    for pat in (r"^(.*)_s\d+$", r"^(.*)_seed\d+$"):
        m = re.match(pat, run_dir_name)
        if m:
            return m.group(1)
    return run_dir_name

# ----------------------------------------------------------------------------- #
# NEW ▸ row normaliser so downstream code always finds the columns it expects.
# ----------------------------------------------------------------------------- #
def _normalise_row(r: Dict[str, Any]) -> None:
    """
    In-place harmonisation of column names produced by different script versions.
    Adds/renames:
        final_patch_size   – int
        queries_used       – int
        target_conf_final  – float
    """
    # patch size
    if not r.get("final_patch_size"):
        if r.get("patch_size"):
            r["final_patch_size"] = r["patch_size"]
        else:
            m = _PATCH_RE.search(str(r.get("patched_path", "")))
            if m:
                r["final_patch_size"] = m.group(1)

    # query count
    if "queries_used" not in r and "queries" in r:
        r["queries_used"] = r["queries"]

    # final target confidence
    if "target_conf_final" not in r:
        if "patched_top_conf" in r:
            r["target_conf_final"] = r["patched_top_conf"]
        elif "top_patched_conf" in r:
            r["target_conf_final"] = r["top_patched_conf"]

# ----------------------------------------------------------------------------- #

def _collect_run_dirs(roots: List[Path]) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[WARN] missing root: {root}")
            continue
        if (root / "per_image_metrics.csv").exists() or list(root.glob("attack_summary*.json")):
            out.append(root)
            continue
        for child in root.iterdir():
            if child.is_dir() and (
                (child / "per_image_metrics.csv").exists() or list(child.glob("attack_summary*.json"))
            ):
                out.append(child)
        # deep fallback
        for p in root.rglob("per_image_metrics.csv"):
            out.append(p.parent)
    return sorted(set(out), key=str)

def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)

def _collect_per_image_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    csv_path = run_dir / "per_image_metrics.csv"
    if csv_path.exists():
        for r in _load_csv(csv_path):
            r["_run_dir"] = str(run_dir)
            rows.append(r)
        return rows

    # fall back to JSON summaries
    json_files = list(run_dir.glob("attack_summary*.json")) or list(run_dir.glob("*.json"))
    for jf in json_files:
        try:
            d = _load_json(jf)
        except Exception as e:
            print(f"[WARN] bad json {jf}: {e}")
            continue
        d["_run_dir"] = str(run_dir)
        rows.append(d)
    return rows

def _compute_area_pct(r: Dict[str, Any]) -> Optional[float]:
    sz = _safe_float(r.get("final_patch_size"))
    h  = _safe_float(r.get("img_h"))
    w  = _safe_float(r.get("img_w"))
    return (sz * sz) / (h * w) * 100.0 if None not in (sz, h, w) and h > 0 and w > 0 else None

def _compute_drop(r: Dict[str, Any]) -> Optional[float]:
    c0 = _safe_float(r.get("top_clean_conf"))
    cf = _safe_float(r.get("target_conf_final"))
    return (c0 - cf) if None not in (c0, cf) else None

def _summ_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    return {
        "mean":   float(sum(vals) / len(vals)),
        "median": float(statistics.median(vals)),
        "p25":    float(statistics.quantiles(vals, n=4, method="inclusive")[0]),
        "p75":    float(statistics.quantiles(vals, n=4, method="inclusive")[2]),
        "min":    float(min(vals)),
        "max":    float(max(vals)),
        "n":      len(vals),
    }

def _summ_binary(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0)) for r in rows if r.get(key) not in (None, "")]
    return 100.0 * sum(v > 0.5 for v in vals) / len(vals) if vals else float("nan")

def _patch_hist(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    hist: Dict[int, int] = {}
    for r in rows:
        sz = _safe_int(r.get("final_patch_size"))
        if sz is None:
            continue
        hist[sz] = hist.get(sz, 0) + 1
    return dict(sorted(hist.items()))

# ══════════════════════════════════════════════════════════════════════════════
# Group-level summarisation
# ══════════════════════════════════════════════════════════════════════════════
def summarize_group(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    success_pct   = _summ_binary(rows, "success")
    top_swap_pct  = _summ_binary(rows, "top_swap")

    drop_stats    = _summ_stats([d for d in (_compute_drop(r)  for r in rows) if d is not None])
    area_stats    = _summ_stats([a for a in (_compute_area_pct(r) for r in rows) if a is not None])
    query_stats   = _summ_stats([_safe_float(r.get("queries_used"))   for r in rows if r.get("queries_used")   not in (None, "")])
    stealth_stats = _summ_stats([_safe_float(r.get("stealth"))        for r in rows if r.get("stealth")        not in (None, "")])
    tgt_stats     = _summ_stats([_safe_float(r.get("target_conf_final")) for r in rows if r.get("target_conf_final") not in (None, "")])

    return dict(
        name=name, n_rows=len(rows),
        success_pct=success_pct, top_swap_pct=top_swap_pct,
        drop_stats=drop_stats, area_stats=area_stats,
        query_stats=query_stats, stealth_stats=stealth_stats,
        tgt_conf_stats=tgt_stats, patch_hist=_patch_hist(rows)
    )

def _fmt_stats(s: Dict[str, Any], field: str, fmt: str) -> str:
    return fmt.format(**s[field]) if s[field] else "n/a"

def print_group_summary(g: Dict[str, Any]) -> None:
    print(f"\n----------------- {g['name']} -----------------")
    print(f"Images:              {g['n_rows']}")
    print(f"Strict Success (%):  {g['success_pct']:.1f}" if g['success_pct']==g['success_pct'] else "Strict Success (%):  n/a")
    if g['top_swap_pct'] == g['top_swap_pct']:
        print(f"Top-Swap  (%):       {g['top_swap_pct']:.1f}")
    if g['drop_stats']:
        print("Target drop:         median={median:.3f}  mean={mean:.3f}".format(**g['drop_stats']))
    if g['area_stats']:
        print("Patch area %:        median={median:.2f}  mean={mean:.2f}".format(**g['area_stats']))
    if g['query_stats']:
        print("Queries:             median={median:.1f}  mean={mean:.1f}".format(**g['query_stats']))
    if g['stealth_stats']:
        print("Stealth score:       median={median:.3f}  mean={mean:.3f}".format(**g['stealth_stats']))
    if g['patch_hist']:
        print("Patch size counts:   " + ", ".join(f"{k}px:{v}" for k,v in g['patch_hist'].items()))
    print("------------------------------------------------")

# ══════════════════════════════════════════════════════════════════════════════
# Optional LaTeX helper
# ══════════════════════════════════════════════════════════════════════════════
def latex_table(groups: List[Dict[str, Any]], fname: Path) -> None:
    with fname.open("w") as f:
        f.write("% Auto-generated by summarize_runs.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Attack performance summary.}\n")
        f.write("\\label{tab:auto_summary}\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Method & Strict Succ. (\\%) & Drop$_t$ & Area (\\%) & Queries \\\\\n\\midrule\n")
        for g in groups:
            succ  = g['success_pct']
            drop  = g['drop_stats'].get("median", float("nan"))
            area  = g['area_stats'].get("median", float("nan"))
            query = g['query_stats'].get("median", float("nan"))
            f.write(f"{g['name']} & {succ:.1f} & {drop:.3f} & {area:.2f} & {query:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Summarize PatchBandit runs.")
    ap.add_argument("--roots", nargs="+", required=True, help="Run dirs or batch roots.")
    ap.add_argument("--out", default="aggregate_summary.csv", help="Combined per-image CSV out.")
    ap.add_argument("--latex-out", default=None, help="If set, write LaTeX table here.")
    args = ap.parse_args()

    root_paths = [Path(r).resolve() for r in args.roots]
    run_dirs   = _collect_run_dirs(root_paths)
    if not run_dirs:
        print("[ERROR] No run directories located.")
        return
    print(f"[INFO] Found {len(run_dirs)} run directories.")

    all_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for rd in run_dirs:
        rows = _collect_per_image_rows(rd)
        if not rows:
            continue
        grp = _group_name(rd.name)
        for r in rows:
            _normalise_row(r)          # ▸ ensure required columns present
            r["_run"]   = rd.name
            r["_group"] = grp
            all_rows.append(r)
            grouped.setdefault(grp, []).append(r)

    # write aggregate CSV
    if all_rows:
        fields = sorted({k for row in all_rows for k in row.keys()})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in all_rows:
                w.writerow(row)
        print(f"[INFO] Wrote per-image aggregate: {args.out} (n={len(all_rows)})")

    # per-group summary
    summaries = []
    for gname, grows in sorted(grouped.items()):
        summ = summarize_group(gname, grows)
        summaries.append(summ)
        print_group_summary(summ)

    # LaTeX
    if args.latex_out:
        latex_table(summaries, Path(args.latex_out))
        print(f"[INFO] Wrote LaTeX table → {args.latex_out}")

if __name__ == "__main__":
    main()

###########################################################################################

# #!/usr/bin/env python
# """
# summarize_runs.py

# Aggregate PatchBandit experiment outputs into paper-ready summary stats.

# Supported inputs per run directory:
#   * per_image_metrics.csv  (preferred; one row per image)
#   * one-or-many attack_summary*.json files (fallback)

# Expected / tolerated columns (missing values handled gracefully):
#   image_id
#   success              (0/1 strict)
#   top_clean            (string)
#   top_clean_conf       (float)
#   top_patched          (string)
#   top_patched_conf     (float)
#   target_conf_final    (float)    # model conf for target at termination
#   final_patch_size     (int)
#   img_h, img_w         (int)      # to compute area%
#   queries_used         (int)
#   stealth              (float)    # L1 deviation or user metric
#   top_swap             (0/1)      # optional: fast mode success

# Grouping:
#   If your run dirs are named like "runs_full_ctx_strict" OR "runs_full_ctx_strict_s0",
#   we strip a trailing "_s<digits>" seed suffix and aggregate across seeds.

# Outputs:
#   * Summary printed to stdout in clearly delimited sections.
#   * Combined per-image table written to --out CSV (default aggregate_summary.csv).
#   * Optional LaTeX table snippet (--latex-out).

# Usage:
#   python summarize_runs.py --roots runs_full_ctx_strict runs_nocx_strict ...
#   python summarize_runs.py --roots runs_batch_20250716_123456  # will recurse
# """

# import os, re, glob, json, argparse, csv, statistics
# from pathlib import Path
# from typing import Dict, List, Any, Optional, Tuple, Callable

# # -------------------------- helpers --------------------------

# def _safe_float(x, default=None):
#     try:
#         if x is None: return default
#         return float(x)
#     except Exception:
#         return default

# def _safe_int(x, default=None):
#     try:
#         if x is None: return default
#         return int(float(x))
#     except Exception:
#         return default

# def _group_name(run_dir_name: str) -> str:
#     """
#     Strip trailing `_s<seed>` or `_seed<seed>` to group multi-seed runs.
#     """
#     m = re.match(r'^(.*)_s\d+$', run_dir_name)
#     if m:
#         return m.group(1)
#     m = re.match(r'^(.*)_seed\d+$', run_dir_name)
#     if m:
#         return m.group(1)
#     return run_dir_name

# def _collect_run_dirs(roots: List[Path]) -> List[Path]:
#     """
#     Expand each root. If root is a directory that contains subruns, include children;
#     if root itself is a run (contains per_image_metrics.csv or attack_summary*.json), include root.
#     """
#     out = []
#     for r in roots:
#         if not r.exists():
#             print(f"[WARN] root missing: {r}")
#             continue
#         # If contains metrics, treat as run
#         if (r / "per_image_metrics.csv").exists() or list(r.glob("attack_summary*.json")):
#             out.append(r)
#             continue
#         # Else scan children
#         child_runs = [p for p in r.iterdir() if p.is_dir()]
#         # Only keep children that look like runs
#         child_runs = [p for p in child_runs if (p / "per_image_metrics.csv").exists() or list(p.glob("attack_summary*.json"))]
#         if child_runs:
#             out.extend(child_runs)
#         else:
#             # Accept empty root anyway; maybe deeper nested
#             for p in r.rglob("per_image_metrics.csv"):
#                 out.append(p.parent)
#     return sorted(set(out), key=lambda p: str(p))

# def _load_csv(path: Path) -> List[Dict[str, Any]]:
#     rows = []
#     with path.open(newline="") as f:
#         rdr = csv.DictReader(f)
#         for r in rdr:
#             rows.append(dict(r))
#     return rows

# def _load_json(path: Path) -> Dict[str, Any]:
#     with path.open() as f:
#         return json.load(f)

# def _collect_per_image_rows(run_dir: Path) -> List[Dict[str, Any]]:
#     """
#     Return a list of per-image dicts for this run.
#     Priority: per_image_metrics.csv.
#     Else: collect all attack_summary*.json (or attack_summary.json if single).
#     """
#     rows: List[Dict[str, Any]] = []
#     csv_path = run_dir / "per_image_metrics.csv"
#     if csv_path.exists():
#         base_rows = _load_csv(csv_path)
#         for br in base_rows:
#             br["_run_dir"] = str(run_dir)
#             rows.append(br)
#         return rows

#     # JSON fallback
#     js_files = list(run_dir.glob("attack_summary*.json"))
#     if not js_files:
#         # Accept single summary.json style
#         js_files = list(run_dir.glob("*.json"))
#     for jf in js_files:
#         try:
#             d = _load_json(jf)
#         except Exception as e:
#             print(f"[WARN] bad json {jf}: {e}")
#             continue
#         d["_run_dir"] = str(run_dir)
#         rows.append(d)
#     return rows

# def _to_float_list(rows: List[Dict[str, Any]], key: str) -> List[float]:
#     vals = []
#     for r in rows:
#         v = _safe_float(r.get(key))
#         if v is not None:
#             vals.append(v)
#     return vals

# def _to_int_list(rows: List[Dict[str, Any]], key: str) -> List[int]:
#     vals = []
#     for r in rows:
#         v = _safe_int(r.get(key))
#         if v is not None:
#             vals.append(v)
#     return vals

# def _compute_area_pct(r: Dict[str, Any]) -> Optional[float]:
#     sz = _safe_float(r.get("final_patch_size"))
#     h  = _safe_float(r.get("img_h"))
#     w  = _safe_float(r.get("img_w"))
#     if None in (sz, h, w) or h <= 0 or w <= 0:
#         return None
#     return (sz * sz) / (h * w) * 100.0

# def _compute_drop(r: Dict[str, Any]) -> Optional[float]:
#     c0 = _safe_float(r.get("top_clean_conf"))
#     cf = _safe_float(r.get("target_conf_final"))
#     if c0 is None or cf is None:
#         return None
#     return c0 - cf

# def _summ_stats(vals: List[float]) -> Dict[str, float]:
#     if not vals:
#         return {}
#     return {
#         "mean": float(sum(vals)/len(vals)),
#         "median": float(statistics.median(vals)),
#         "p25": float(statistics.quantiles(vals, n=4, method="inclusive")[0]),
#         "p75": float(statistics.quantiles(vals, n=4, method="inclusive")[2]),
#         "min": float(min(vals)),
#         "max": float(max(vals)),
#         "n": len(vals),
#     }

# def _summ_binary(rows: List[Dict[str, Any]], key: str) -> float:
#     # return %True
#     if not rows:
#         return float('nan')
#     tot = 0; acc = 0
#     for r in rows:
#         v = r.get(key)
#         if v is None: continue
#         tot += 1
#         acc += int(float(v) > 0.5)
#     if tot == 0:
#         return float('nan')
#     return 100.0 * acc / tot

# def _patch_size_hist(rows: List[Dict[str, Any]]) -> Dict[int,int]:
#     hist = {}
#     for r in rows:
#         sz = _safe_int(r.get("final_patch_size"))
#         if sz is None:
#             continue
#         hist[sz] = hist.get(sz, 0) + 1
#     return dict(sorted(hist.items()))

# # -------------------------- summarizer per group --------------------------

# def summarize_group(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Compute summary metrics for a group (possibly multi-seed).
#     """
#     # unify keys to numeric
#     success_pct = _summ_binary(rows, "success")
#     top_swap_pct = _summ_binary(rows, "top_swap")

#     drops = [v for v in (_compute_drop(r) for r in rows) if v is not None]
#     drop_stats = _summ_stats(drops)

#     areas = [v for v in (_compute_area_pct(r) for r in rows) if v is not None]
#     area_stats = _summ_stats(areas)

#     queries = _to_float_list(rows, "queries_used")
#     query_stats = _summ_stats(queries)

#     stealth = _to_float_list(rows, "stealth")
#     stealth_stats = _summ_stats(stealth)

#     # final target conf (median)
#     tgt_conf = _to_float_list(rows, "target_conf_final")
#     tgt_conf_stats = _summ_stats(tgt_conf)

#     # patch size distribution
#     p_hist = _patch_size_hist(rows)

#     return {
#         "name": name,
#         "n_rows": len(rows),
#         "success_pct": success_pct,
#         "top_swap_pct": top_swap_pct,
#         "drop_stats": drop_stats,
#         "area_stats": area_stats,
#         "query_stats": query_stats,
#         "stealth_stats": stealth_stats,
#         "tgt_conf_stats": tgt_conf_stats,
#         "patch_hist": p_hist,
#     }

# def print_group_summary(g: Dict[str, Any]):
#     name = g["name"]
#     print(f"\n----------------- {name} -----------------")
#     print(f"Images: {g['n_rows']}")
#     print(f"Strict Success (%): {g['success_pct']:.1f}" if g['success_pct'] == g['success_pct'] else "Strict Success (%): n/a")
#     if g["top_swap_pct"] == g["top_swap_pct"]:
#         print(f"Top-Swap (%): {g['top_swap_pct']:.1f}")
#     if g["drop_stats"]:
#         print("Target score drop: median={median:.3f} mean={mean:.3f} p25={p25:.3f} p75={p75:.3f}".format(**g["drop_stats"]))
#     if g["tgt_conf_stats"]:
#         print("Final target conf: median={median:.3f} mean={mean:.3f}".format(**g["tgt_conf_stats"]))
#     if g["area_stats"]:
#         print("Patch area %: median={median:.2f} mean={mean:.2f}".format(**g["area_stats"]))
#     if g["query_stats"]:
#         print("Queries: median={median:.1f} mean={mean:.1f}".format(**g["query_stats"]))
#     if g["stealth_stats"]:
#         print("Stealth score: median={median:.3f} mean={mean:.3f}".format(**g["stealth_stats"]))
#     if g["patch_hist"]:
#         hist_str = ", ".join(f"{k}px:{v}" for k,v in g["patch_hist"].items())
#         print(f"Patch size counts: {hist_str}")
#     print("------------------------------------------")

# # -------------------------- LaTeX table helpers --------------------------

# def latex_table(groups: List[Dict[str, Any]], fname: Path):
#     """
#     Write a simple LaTeX table summarizing success, drop, area, queries.
#     """
#     with fname.open("w") as f:
#         f.write("% Auto-generated by summarize_runs.py\n")
#         f.write("\\begin{table}[t]\n\\centering\n")
#         f.write("\\caption{Attack performance summary.}\n")
#         f.write("\\label{tab:auto_summary}\n")
#         f.write("\\setlength{\\tabcolsep}{4pt}\n")
#         f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
#         f.write("Method & Strict Succ. (\\%) & Drop$_t$ & Area (\\%) & Queries \\\\\n\\midrule\n")
#         for g in groups:
#             drop_med = g["drop_stats"].get("median", float("nan")) if g["drop_stats"] else float("nan")
#             area_med = g["area_stats"].get("median", float("nan")) if g["area_stats"] else float("nan")
#             query_med = g["query_stats"].get("median", float("nan")) if g["query_stats"] else float("nan")
#             succ = g["success_pct"]
#             f.write(f"{g['name']} & {succ:.1f} & {drop_med:.3f} & {area_med:.2f} & {query_med:.0f} \\\\\n")
#         f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# # -------------------------- main --------------------------

# def main():
#     ap = argparse.ArgumentParser(description="Summarize PatchBandit runs.")
#     ap.add_argument("--roots", nargs="+", required=True,
#                     help="One or more run dirs or batch roots.")
#     ap.add_argument("--out", default="aggregate_summary.csv",
#                     help="Path to write combined per-image CSV.")
#     ap.add_argument("--latex-out", default=None,
#                     help="Optional path to write LaTeX summary table.")
#     args = ap.parse_args()

#     roots = [Path(r).resolve() for r in args.roots]
#     run_dirs = _collect_run_dirs(roots)
#     if not run_dirs:
#         print("[ERROR] No run directories found.")
#         return

#     print(f"[INFO] Found {len(run_dirs)} run directories.")
#     all_rows = []
#     grouped_rows: Dict[str, List[Dict[str, Any]]] = {}

#     for rd in run_dirs:
#         rows = _collect_per_image_rows(rd)
#         if not rows:
#             continue
#         base = _group_name(rd.name)
#         for r in rows:
#             r["_run"] = rd.name
#             r["_group"] = base
#             all_rows.append(r)
#             grouped_rows.setdefault(base, []).append(r)

#     # write combined CSV
#     if all_rows:
#         # merge fields
#         fieldset = set()
#         for r in all_rows:
#             fieldset.update(r.keys())
#         fields = sorted(fieldset)
#         with open(args.out, "w", newline="") as f:
#             w = csv.DictWriter(f, fieldnames=fields)
#             w.writeheader()
#             for r in all_rows:
#                 w.writerow(r)
#         print(f"[INFO] Wrote per-image aggregate: {args.out} ({len(all_rows)} rows)")
#     else:
#         print("[WARN] no rows to write aggregate CSV")

#     # summarize per group
#     summaries = []
#     for gname, grows in sorted(grouped_rows.items()):
#         summ = summarize_group(gname, grows)
#         summaries.append(summ)
#         print_group_summary(summ)

#     # LaTeX out
#     if args.latex_out:
#         latex_table(summaries, Path(args.latex_out))
#         print(f"[INFO] Wrote LaTeX table: {args.latex_out}")

# if __name__ == "__main__":
#     main()
