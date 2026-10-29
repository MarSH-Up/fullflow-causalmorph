#!/usr/bin/env python3
"""
Tables of run_v3.sh (results/v3): Table 1 per grid cell and external noise family (nSHD,
skeleton F1, paired differences against the method with 95% bootstrap CIs), the ablation
rows on the 600 held-out ids, the detector table (change-point F1 with precision/recall,
confirmation F1, windows per regime) and CASTOR.

  python tools/analyze_v3.py                       # results/v3
  RESULTS_ROOT=server_results/<date> OUT=v3 python tools/analyze_v3.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.environ.get("RESULTS_ROOT") or os.path.join(os.path.dirname(__file__), ".."))
OUT = os.environ.get("OUT", "v3")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from analyze_final import paired, fmt, parse, f1 as det_f1  # noqa: E402

CELLS = ["lin_laplace", "nonlin_laplace", "lin_normal", "nonlin_normal"]
R = lambda *p: os.path.join(ROOT, "results", OUT, *p)  # noqa: E731


def load(*p):
    f = R(*p, "results.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d[d.status == "ok"].drop_duplicates("scenario_id").set_index("scenario_id")


def col(d, name):
    return None if d is None or name not in d else d[name]


def rows_for(cell, main_step="main"):
    m = load(cell, main_step); cd = load(cell, "cdnod_fz"); tuned = load(cell, "dagma_tuned"); cas = load(cell, "castor")
    rows = {"Method (CM v2 + Stage 0 + DirectLiNGAM)": (col(m, "mean_norm_shd"), col(m, "mean_skeleton_f1"))}
    for k, lab in (("directlingam", "DirectLiNGAM, same windows"), ("dagma_paper", "DAGMA paper cfg, same windows"),
                   ("pc", "PC, same windows"), ("icalingam", "ICA-LiNGAM, same windows"),
                   ("sortnregress", "sortnregress, same windows"), ("randomregress", "random-order, same windows"), ("empty", "Empty graph")):
        rows[lab] = (col(m, f"{k}_mean_norm_shd"), col(m, f"{k}_mean_skeleton_f1"))
    if tuned is not None:
        name = next((c[:-len("_mean_norm_shd")] for c in tuned if c.startswith("dagma_l") and c.endswith("_mean_norm_shd")), None)
        if name:
            rows["DAGMA tuned, same windows"] = (tuned[f"{name}_mean_norm_shd"], tuned.get(f"{name}_mean_skeleton_f1"))
    rows["CD-NOD Fisher-z"] = (col(cd, "mean_norm_shd"), col(cd, "mean_skeleton_f1"))
    rows["CD-NOD vs union graph"] = (col(cd, "union_norm_shd"), col(cd, "union_skeleton_f1"))
    rows["CASTOR (linear EM)"] = (col(cas, "mean_norm_shd"), col(cas, "mean_skeleton_f1"))
    return rows, col(m, "mean_norm_shd")


def holm(pvals):
    """Holm step-down adjusted p-values (nan kept)."""
    idx = [i for i, v in enumerate(pvals) if np.isfinite(v)]
    out = [np.nan] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i]); m = len(order); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i]); out[i] = min(running, 1.0)
    return out


def table(columns):
    """nSHD, skeleton F1 and paired tables; in the paired table '*' marks a 95% CI excluding 0 and
    '†' a Wilcoxon p below 0.05 after Holm's correction over the rows of the same column."""
    labels = list(dict.fromkeys(l for rows, _ in columns.values() for l in rows))
    M = pd.DataFrame(index=labels, columns=list(columns), dtype=object); S, P = M.copy(), M.copy()
    for c, (rows, ref) in columns.items():
        res = {}
        for l, (s, sk) in rows.items():
            if s is not None and len(s.dropna()):
                M.loc[l, c] = f"{s.mean():.3f}"
            if sk is not None and len(sk.dropna()):
                S.loc[l, c] = f"{sk.mean():.3f}"
            if not l.startswith("Method") and s is not None and ref is not None:
                res[l] = paired(ref, s)
        adj = holm([r["p"] if r else np.nan for r in res.values()])
        for (l, r), pa in zip(res.items(), adj):
            P.loc[l, c] = fmt(r) + (" †" if np.isfinite(pa) and pa < 0.05 else "")
    return M.fillna("-"), S.fillna("-"), P.fillna("-")


def detection_table():
    out = []
    steps = [("det_v1O", "v1O (previous)"), ("det_v1O_rank", "v1O + rank"), ("det_pelt", "PELT"), ("det_pelt_rank", "PELT + rank"), ("main600", "union + rank (method)")]
    for cell in CELLS:
        for step, lab in steps:
            d = load(cell, step)
            if d is None:
                continue
            tp, fp, fn = d.det_tp.sum(), d.det_fp.sum(), d.det_fn.sum()
            P, Rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            ctp, cfp, cfn = d.chow0001_tp.sum(), d.chow0001_fp.sum(), d.chow0001_fn.sum()
            cP, cR = ctp / max(ctp + cfp, 1), ctp / max(ctp + cfn, 1)
            out.append({"cell": cell, "detector": lab, "n": len(d), "det F1": round(2 * P * Rc / max(P + Rc, 1e-9), 3), "prec": round(P, 2), "rec": round(Rc, 2),
                        "conf F1": round(2 * cP * cR / max(cP + cR, 1e-9), 3), "windows/regime": round((d.n_structures / d.n_regimes).mean(), 2),
                        "nSHD": round(d.mean_norm_shd.mean(), 3)})
        cas = load(cell, "castor")
        if cas is not None:
            tp, fp, fn = cas.det_tp.sum(), cas.det_fp.sum(), cas.det_fn.sum(); P, Rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            out.append({"cell": cell, "detector": "CASTOR regimes", "n": len(cas), "det F1": round(2 * P * Rc / max(P + Rc, 1e-9), 3), "prec": round(P, 2), "rec": round(Rc, 2),
                        "conf F1": np.nan, "windows/regime": round((cas.n_regimes_found / cas.n_regimes).mean(), 2), "nSHD": round(cas.mean_norm_shd.mean(), 3)})
    return pd.DataFrame(out)


def main():
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
    grid = {c: rows_for(c) for c in CELLS}
    ext = load("external", "main")
    if ext is not None:
        info = open(R("RUN_INFO.txt")).read()
        data = info.split("external: ")[1].split(" |")[0].strip()
        noise = pd.read_csv(os.path.join(ROOT, data, "index.csv")).set_index("scenario_id").sem_type
        rows, ref = rows_for("external")
        for s in ("exp", "gumbel", "gauss"):
            keep = set(noise.index[noise == s])
            grid[f"external/{s}"] = ({l: tuple(None if x is None else x[x.index.isin(keep)] for x in v) for l, v in rows.items()}, ref[ref.index.isin(keep)])
    M, S, P = table(grid)
    print("TABLE 1: mean nSHD (lower is better)\n" + M.to_string())
    print("\nskeleton F1 (higher is better)\n" + S.to_string())
    print("\npaired: method - row (mean [95% CI] scenarios where the method is lower/higher; * = CI excludes 0; † = Wilcoxon p < 0.05 after Holm per column)\n" + P.to_string())
    app = {}
    for c in CELLS:
        m600 = load(c, "main600"); ref = col(m600, "mean_norm_shd")
        rows = {"Method (600 held-out)": (ref, col(m600, "mean_skeleton_f1"))}
        for step, lab in (("nostage0", "Without Stage 0 (CM v2 + DL), same windows"), ("oracle_seg", "With oracle segmentation"),
                          ("no_seg", "Without segmentation"), ("L1000", "With L = 1000")):
            d = load(c, step); rows[lab] = (col(d, "mean_norm_shd"), col(d, "mean_skeleton_f1"))
        app[c] = (rows, ref)
    Ma, Sa, Pa = table(app)
    print("\nABLATIONS (600 held-out ids): mean nSHD\n" + Ma.to_string())
    print("\nABLATIONS paired: method - row\n" + Pa.to_string())
    D = detection_table()
    if not D.empty:
        print("\nDETECTION (600 held-out ids)\n" + D.to_string(index=False))


if __name__ == "__main__":
    main()
