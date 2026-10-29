#!/usr/bin/env python3
"""
The pre-registered tables (Clear_paper_work/notes/2026-10-28-tabla1-prerregistro.md)
from run_all.sh: Table 1 per grid cell and external noise family, the paired
comparisons of the method against every row (95% bootstrap CI, Wilcoxon), the
detection table with chance lines by regime length, and the appendix rows.

Sources (all on standardised windows, permuted variables, L = 500):
  method, DirectLiNGAM, ICA-LiNGAM, DAGMA (paper cfg), empty   results/standardised/<cell>/main     (MAIN_N per cell)
  DAGMA tuned on calibration                                   results/review_extras/<cell>/dagma_tuned (main's windows)
  CD-NOD Fisher-z (per regime, vs union)                       results/standardised/<cell>/cdnod_fz (main's ids)
  method on the 600 held-out KCI ids                           results/review_extras/<cell>/pc       (its main row)
  PC, no segmentation, PELT, CD-NOD KCI                        600 KCI ids, paired with the row above
  external benchmark                                           results/robustness/external/*

  python tools/analyze_final.py > results/final_tables.txt
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = lambda *p: os.path.join(ROOT, "results", *p)  # noqa: E731
CELLS = ["lin_laplace", "nonlin_laplace", "lin_normal", "nonlin_normal"]
TOL = 200


def load(*p):
    f = R(*p, "results.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d[d.status == "ok"].drop_duplicates("scenario_id").set_index("scenario_id")


def col(d, name):
    return None if d is None or name not in d else d[name]


def paired(a, b, n_boot=2000, seed=0):
    """a - b on the common scenarios: mean [95% bootstrap CI], Wilcoxon p, lower/higher counts."""
    if a is None or b is None:
        return None
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 5:
        return None
    v = (j.a - j.b).to_numpy(float)
    boot = v[np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nz = v[v != 0]
    p = wilcoxon(nz).pvalue if len(nz) > 10 else float("nan")
    return dict(n=len(v), diff=v.mean(), lo=lo, hi=hi, p=p, lower=int((v < 0).sum()), higher=int((v > 0).sum()))


def fmt(r):
    if r is None:
        return "-"
    star = "" if (r["lo"] < 0 < r["hi"]) else " *"
    return f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {r['lower']}/{r['higher']} (n={r['n']}){star}"


def rows_for_grid(cell):
    """{row label: (nSHD series, skeleton F1 series)} and the method series for each scenario set."""
    main = load("standardised", cell, "main")
    tuned = load("review_extras", cell, "dagma_tuned")
    cd = load("standardised", cell, "cdnod_fz")
    pc = load("review_extras", cell, "pc")
    rows = {
        "Method (DirectLiNGAM + v2)": (col(main, "mean_norm_shd"), col(main, "mean_skeleton_f1"), "main"),
        "DirectLiNGAM, same windows": (col(main, "directlingam_mean_norm_shd"), col(main, "directlingam_mean_skeleton_f1"), "main"),
        "DAGMA tuned, same windows": (None, None, "main"),
        "CD-NOD Fisher-z": (col(cd, "mean_norm_shd"), col(cd, "mean_skeleton_f1"), "main"),
        "CD-NOD vs union graph": (col(cd, "union_norm_shd"), col(cd, "union_skeleton_f1"), "main"),
        "Empty graph": (col(main, "empty_mean_norm_shd"), None, "main"),
        "Method (600 KCI ids)": (col(pc, "mean_norm_shd"), col(pc, "mean_skeleton_f1"), "kci"),
        "PC, same windows": (col(pc, "pc_mean_norm_shd"), col(pc, "pc_mean_skeleton_f1"), "kci"),
        "No segmentation": (col(load("standardised", cell, "no_seg"), "mean_norm_shd"),
                            col(load("standardised", cell, "no_seg"), "mean_skeleton_f1"), "kci"),
        "PELT detector": (col(load("standardised", cell, "pelt"), "mean_norm_shd"),
                          col(load("standardised", cell, "pelt"), "mean_skeleton_f1"), "kci"),
        "CD-NOD KCI": (col(load("standardised", cell, "cdnod_kci"), "mean_norm_shd"),
                       col(load("standardised", cell, "cdnod_kci"), "mean_skeleton_f1"), "kci"),
    }
    if tuned is not None:
        name = next((c[:-len("_mean_norm_shd")] for c in tuned if c.startswith("dagma_l") and c.endswith("_mean_norm_shd")), None)
        if name:
            rows["DAGMA tuned, same windows"] = (tuned[f"{name}_mean_norm_shd"], tuned.get(f"{name}_mean_skeleton_f1"), "main")
    appendix = {
        "ICA-LiNGAM, same windows": (col(main, "icalingam_mean_norm_shd"), col(main, "icalingam_mean_skeleton_f1"), "main"),
        "DAGMA paper cfg, same windows": (col(main, "dagma_paper_mean_norm_shd"), col(main, "dagma_paper_mean_skeleton_f1"), "main"),
        "sortnregress": (col(main, "sortnregress_mean_norm_shd"), None, "main"),
        "random-order regression": (col(main, "randomregress_mean_norm_shd"), None, "main"),
    }
    for step, label in (("oracle_seg", "oracle segmentation"), ("oracle_L", "oracle bound L"), ("L1000", "L = 1000"),
                        ("lite", "without Stage II")):
        d = load("standardised", cell, step)
        appendix[f"With {label} (600)"] = (col(d, "mean_norm_shd"), col(d, "mean_skeleton_f1"), "kci")
    return rows, appendix, {"main": rows["Method (DirectLiNGAM + v2)"][0], "kci": rows["Method (600 KCI ids)"][0]}


def rows_for_external():
    std, dl = load("robustness", "external", "std"), load("robustness", "external", "std_dl")
    cd = load("robustness", "external", "cdnod_fz")
    rows = {
        "Method (DirectLiNGAM + v2)": (col(dl, "mean_norm_shd"), col(dl, "mean_skeleton_f1"), "ext"),
        "DirectLiNGAM, same windows": (col(std, "directlingam_mean_norm_shd"), col(std, "directlingam_mean_skeleton_f1"), "ext"),
        "DAGMA paper cfg, same windows": (col(std, "dagma_paper_mean_norm_shd"), col(std, "dagma_paper_mean_skeleton_f1"), "ext"),
        "DAGMA rescaled, same windows": (col(std, "dagma_mean_norm_shd"), col(std, "dagma_mean_skeleton_f1"), "ext"),
        "CD-NOD Fisher-z": (col(cd, "mean_norm_shd"), col(cd, "mean_skeleton_f1"), "ext"),
        "CD-NOD vs union graph": (col(cd, "union_norm_shd"), col(cd, "union_skeleton_f1"), "ext"),
        "No segmentation": (col(load("robustness", "external", "no_seg"), "mean_norm_shd"),
                            col(load("robustness", "external", "no_seg"), "mean_skeleton_f1"), "ext"),
        "PELT detector": (col(load("robustness", "external", "pelt"), "mean_norm_shd"),
                          col(load("robustness", "external", "pelt"), "mean_skeleton_f1"), "ext"),
        "Empty graph": (col(load("robustness", "external", "raw"), "empty_mean_norm_shd"), None, "ext"),
        "ICA-LiNGAM, same windows (appendix)": (col(std, "icalingam_mean_norm_shd"), col(std, "icalingam_mean_skeleton_f1"), "ext"),
    }
    return rows, rows["Method (DirectLiNGAM + v2)"][0]


def external_noise():
    info = open(R("robustness", "RUN_INFO.txt")).read()
    data = info.split("external: ")[1].split(" |")[0].strip()
    return pd.read_csv(os.path.join(ROOT, data, "index.csv")).set_index("scenario_id").sem_type


def table(columns):
    """columns: {column: (rows, method series by set)} -> nSHD table, skeleton table, paired table."""
    labels = list(dict.fromkeys(l for rows, _ in columns.values() for l in rows))
    M = pd.DataFrame(index=labels, columns=list(columns), dtype=object)
    S, P = M.copy(), M.copy()
    for c, (rows, ref) in columns.items():
        for l, (s, sk, kind) in rows.items():
            if s is not None and len(s.dropna()):
                M.loc[l, c] = f"{s.mean():.3f}"
            if sk is not None and len(sk.dropna()):
                S.loc[l, c] = f"{sk.mean():.3f}"
            if not l.startswith("Method") and s is not None:
                P.loc[l, c] = fmt(paired(ref.get(kind), s))
    return M.fillna("-"), S.fillna("-"), P.fillna("-")


def parse(s):
    return [int(x) for x in str(s).split(";") if x.strip() and x != "nan"]


def match(tc, dc, tol=TOL):
    used, tp = set(), 0
    for t in sorted(tc):
        cand = [(abs(x - t), j) for j, x in enumerate(sorted(dc)) if j not in used and abs(x - t) <= tol]
        if cand:
            used.add(min(cand)[1]); tp += 1
    return tp, len(dc) - tp, len(tc) - tp


def f1(d, get):
    TP = FP = FN = 0
    for _, r in d.iterrows():
        a, b, c = match(parse(r.true_cps), get(r)); TP += a; FP += b; FN += c
    P = TP / (TP + FP) if TP + FP else 0.0; Rc = TP / (TP + FN) if TP + FN else 0.0
    return 2 * P * Rc / (P + Rc) if P + Rc else 0.0


def detection():
    out = []
    for cell in CELLS:
        d = load("standardised", cell, "main")
        if d is None:
            continue
        L = int(d.min_regime_len_used.iloc[0])
        for n, g in [("all", d)] + list(d.groupby("samples_regime")):
            tp, fp, fn = g.chow0001_tp.sum(), g.chow0001_fp.sum(), g.chow0001_fn.sum()
            P, Rc = tp / (tp + fp), tp / (tp + fn)
            out.append({"cell": cell, "regime length": n, "n": len(g),
                        "detector": round(f1(g, lambda r: parse(r.detected_cps)), 3),
                        "confirmed": round(2 * P * Rc / (P + Rc), 3),
                        "K equispaced (K known)": round(f1(g, lambda r: [int(round(r.total_samples * k / r.n_regimes))
                                                                         for k in range(1, int(r.n_regimes))]), 3),
                        f"one cut every L={L}": round(f1(g, lambda r: list(range(L, int(r.total_samples), L))), 3)})
    return pd.DataFrame(out).set_index(["cell", "regime length"])


def main():
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
    grid, app = {}, {}
    for c in CELLS:
        rows, appendix, ref = rows_for_grid(c)
        grid[c], app[c] = (rows, ref), (appendix, ref)
    ext_rows, ext_ref = rows_for_external()
    if ext_ref is not None:
        noise = external_noise()
        for s in ("exp", "gumbel", "gauss"):
            keep = set(noise.index[noise == s])
            sub = {l: tuple(None if x is None or isinstance(x, str) else x[x.index.isin(keep)] for x in v[:2]) + (v[2],)
                   for l, v in ext_rows.items()}
            grid[f"external/{s}"] = (sub, {"ext": ext_ref[ext_ref.index.isin(keep)]})
    M, S, P = table(grid)
    print("TABLE 1: mean nSHD (lower is better)\n" + M.to_string())
    print("\nskeleton F1 (higher is better)\n" + S.to_string())
    print("\npaired: method - row (mean [95% CI] scenarios where the method is lower/higher; * = CI excludes 0)\n" + P.to_string())
    Ma, Sa, Pa = table(app)
    print("\nAPPENDIX: mean nSHD\n" + Ma.to_string())
    print("\nAPPENDIX paired: method - row\n" + Pa.to_string())
    D = detection()
    if not D.empty:
        print("\nDETECTION F1 (main, L fixed) against chance lines, by nominal regime length\n" + D.to_string())


if __name__ == "__main__":
    main()
