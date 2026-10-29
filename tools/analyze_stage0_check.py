#!/usr/bin/env python3
"""
Tables of run_stage0_check.sh: per cell, mean nSHD and skeleton F1 of every
configuration and of PC, the empty graph and DirectLiNGAM on the same windows,
the paired difference of each configuration against v2cap (95% bootstrap CI,
share of scenarios lower/higher) and the share of windows Stage 0 handled.

  python tools/analyze_stage0_check.py results/stage0_check
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from analyze_final import paired, fmt  # noqa: E402


def load(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    return d[d.status == "ok"].drop_duplicates("scenario_id").set_index("scenario_id")


def main(out):
    pd.set_option("display.width", 250)
    cells = [c for c in sorted(os.listdir(out)) if os.path.isdir(os.path.join(out, c)) and c != "logs"]
    M, S, P, W = {}, {}, {}, {}
    for c in cells:
        base = load(os.path.join(out, c, "v2cap", "results.csv"))
        if base is None:
            continue
        m, s, p, w = {}, {}, {}, {}
        for k, col in (("PC", "pc_mean_norm_shd"), ("empty", "empty_mean_norm_shd"), ("DirectLiNGAM", "directlingam_mean_norm_shd")):
            if col in base:
                m[k] = base[col].mean(); p[k] = fmt(paired(base.mean_norm_shd, base[col]))
                sk = col.replace("norm_shd", "skeleton_f1")
                s[k] = base[sk].mean() if sk in base else np.nan
        for cfg in sorted(os.listdir(os.path.join(out, c))):
            d = load(os.path.join(out, c, cfg, "results.csv"))
            if d is None or cfg == "logs":
                continue
            if cfg == "cdnod_fz":
                m["CD-NOD Fisher-z"] = d.mean_norm_shd.mean(); s["CD-NOD Fisher-z"] = d.mean_skeleton_f1.mean()
                p["CD-NOD Fisher-z"] = fmt(paired(base.mean_norm_shd, d.mean_norm_shd))
                m["CD-NOD vs union"] = d.union_norm_shd.mean(); s["CD-NOD vs union"] = d.union_skeleton_f1.mean()
                continue
            if cfg == "baselines":
                for k, col in (("DAGMA paper cfg", "dagma_paper_mean_norm_shd"), ("ICA-LiNGAM alone", "icalingam_mean_norm_shd")):
                    if col in d:
                        m[k] = d[col].mean(); s[k] = d[col.replace("norm_shd", "skeleton_f1")].mean()
                        p[k] = fmt(paired(base.mean_norm_shd, d[col]))
                continue
            m[cfg] = d.mean_norm_shd.mean(); s[cfg] = d.mean_skeleton_f1.mean()
            p[cfg] = "-" if cfg == "v2cap" else fmt(paired(d.mean_norm_shd, base.mean_norm_shd.reindex(d.index)))
            w[cfg] = (d.n_stage0_windows.sum() / d.n_structures.sum()) if "n_stage0_windows" in d else np.nan
        M[c], S[c], P[c], W[c] = m, s, p, w
    print("mean nSHD (lower is better)\n" + pd.DataFrame(M).round(3).fillna("-").to_string())
    print("\nskeleton F1\n" + pd.DataFrame(S).round(3).fillna("-").to_string())
    print("\npaired: config - v2cap (and v2cap - baseline for PC/empty/DirectLiNGAM): mean [95% CI] lower/higher, * CI excludes 0\n"
          + pd.DataFrame(P).fillna("-").to_string())
    print("\nshare of windows handled by Stage 0\n" + pd.DataFrame(W).round(2).fillna("-").to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "stage0_check"))
