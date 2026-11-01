#!/usr/bin/env python3
"""
LaTeX tables and the number sheet of the CLeaR paper from results/v3 (run_v3.sh) and its
complements. Every number the text quotes comes from numbers.json, written next to the tables.

  RESULTS_ROOT=server_results/2026-10-31_v3 OUT=v3 python tools/make_v3_tables.py --tex_dir ../Clear_paper/tables

Statistics (per comparison, paired by scenario): mean difference with a 95% percentile bootstrap
CI (2,000 resamples), median difference, Hodges-Lehmann estimate, Wilcoxon signed-rank test with
Pratt's treatment of zeros (W+, W-, n, P), matched-pairs rank-biserial correlation, share of
scenarios lower/higher, Holm correction over the rows of a column.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = os.path.abspath(os.environ.get("RESULTS_ROOT") or os.path.join(os.path.dirname(__file__), ".."))
OUT = os.environ.get("OUT", "v3")
R = lambda *p: os.path.join(ROOT, "results", OUT, *p)  # noqa: E731
CELLS = ["lin_laplace", "nonlin_laplace", "lin_normal", "nonlin_normal"]
CELL_TEX = {"lin_laplace": r"\makecell{linear\\Laplace}", "nonlin_laplace": r"\makecell{cubic\\Laplace}",
            "lin_normal": r"\makecell{linear\\Gaussian}", "nonlin_normal": r"\makecell{cubic\\Gaussian}",
            "external/exp": r"\makecell{ext.\\exponential}", "external/gumbel": r"\makecell{ext.\\Gumbel}", "external/gauss": r"\makecell{ext.\\Gaussian}"}
METHOD = r"\method{}"
MAIN_LABELS = {"dagma": "DAGMA (tuned)", "cdnod": "CD-NOD (Fisher-z)", "castor": "CASTOR (subsample)", "pc": "PC", "directlingam": "DirectLiNGAM alone"}
CELL_ROW = {"lin_laplace": "linear / Laplace", "nonlin_laplace": "cubic / Laplace", "lin_normal": "linear / Gaussian", "nonlin_normal": "cubic / Gaussian",
            "external/exp": "external / exponential", "external/gumbel": "external / Gumbel", "external/gauss": "external / Gaussian"}


def load(*p):
    f = R(*p, "results.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d[d.status == "ok"].drop_duplicates("scenario_id").set_index("scenario_id")


def col(d, name):
    return None if d is None or name not in d else d[name]


def paired_stats(a, b, n_boot=2000, seed=0):
    """Statistics of a - b on the common scenarios."""
    if a is None or b is None:
        return None
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 5:
        return None
    v = (j.a - j.b).to_numpy(float); n = len(v)
    boot = v[np.random.default_rng(seed).integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nz = v[v != 0]
    if len(nz) > 10:
        r = rankdata(np.abs(v))                      # Pratt: zeros ranked, then dropped
        wp = float(r[v > 0].sum()); wm = float(r[v < 0].sum())
        p = float(wilcoxon(v, zero_method="pratt").pvalue)
        rb = (wp - wm) / (wp + wm) if wp + wm else 0.0
    else:
        wp = wm = rb = p = float("nan")
    if n <= 4000:
        walsh = (v[:, None] + v[None, :]) / 2.0
        hl = float(np.median(walsh[np.triu_indices(n)]))
    else:
        idx = np.random.default_rng(seed).choice(n, 4000, replace=False); w = v[idx]
        hl = float(np.median(((w[:, None] + w[None, :]) / 2.0)[np.triu_indices(4000)]))
    return dict(n=n, n_nonzero=int(len(nz)), mean=float(v.mean()), sd=float(v.std(ddof=1)), lo=float(lo), hi=float(hi),
                median=float(np.median(v)), hl=hl, w_plus=wp, w_minus=wm, p=p, rb=rb,
                lower=int((v < 0).sum()), higher=int((v > 0).sum()), ties=int((v == 0).sum()))


def holm(pvals):
    idx = [i for i, v in enumerate(pvals) if np.isfinite(v)]
    out = [np.nan] * len(pvals); order = sorted(idx, key=lambda i: pvals[i]); m = len(order); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i]); out[i] = min(run, 1.0)
    return out


def ms(s):
    """mean (SD) of a series."""
    s = s.dropna()
    return f"{s.mean():.3f} ({s.std(ddof=1):.3f})" if len(s) else "--"


def f1_of(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0; Rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * P * Rc / (P + Rc) if P + Rc else 0.0), P, Rc


def macro_f1(d, n_boot=2000, seed=0):
    f = np.array([f1_of(tp, fp, fn)[0] for tp, fp, fn in zip(d.det_tp, d.det_fp, d.det_fn)])
    boot = f[np.random.default_rng(seed).integers(0, len(f), (n_boot, len(f)))].mean(axis=1)
    return float(f.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ───────────────────────────── rows per column ─────────────────────────────
def grid_rows(cell):
    m = load(cell, "main"); cd = load(cell, "cdnod_fz"); tuned = load(cell, "dagma_tuned"); cas = load(cell, "castor"); diag = load(cell, "diagnostics")
    rows = {"method": (col(m, "mean_norm_shd"), col(m, "mean_skeleton_f1"))}
    name = None
    if tuned is not None:
        name = next((c[:-len("_mean_norm_shd")] for c in tuned if c.startswith("dagma_l") and c.endswith("_mean_norm_shd")), None)
    rows["dagma"] = (tuned[f"{name}_mean_norm_shd"], tuned.get(f"{name}_mean_skeleton_f1")) if name else (col(m, "dagma_paper_mean_norm_shd"), col(m, "dagma_paper_mean_skeleton_f1"))
    rows["cdnod"] = (col(cd, "mean_norm_shd"), col(cd, "mean_skeleton_f1"))
    rows["cdnod_union"] = (col(cd, "union_norm_shd"), col(cd, "union_skeleton_f1"))
    rows["pc"] = (col(m, "pc_mean_norm_shd"), col(m, "pc_mean_skeleton_f1"))
    rows["castor"] = (col(cas, "mean_norm_shd"), col(cas, "mean_skeleton_f1"))
    for k in ("directlingam", "icalingam", "dagma_paper", "sortnregress", "randomregress"):
        rows[k] = (col(m, f"{k}_mean_norm_shd"), col(m, f"{k}_mean_skeleton_f1"))
    rows["single_oracle"] = (col(diag, "nshd_single_oracle"), None)
    return rows, m


def external_rows():
    m = load("external", "main"); cd = load("external", "cdnod_fz"); cas = load("external", "castor"); tuned = load("external", "dagma_tuned"); diag = load("external", "diagnostics")
    rows = {"method": (col(m, "mean_norm_shd"), col(m, "mean_skeleton_f1"))}
    name = None
    if tuned is not None:
        name = next((c[:-len("_mean_norm_shd")] for c in tuned if c.startswith("dagma_l") and c.endswith("_mean_norm_shd")), None)
    rows["dagma"] = (tuned[f"{name}_mean_norm_shd"], tuned.get(f"{name}_mean_skeleton_f1")) if name else (col(m, "dagma_paper_mean_norm_shd"), col(m, "dagma_paper_mean_skeleton_f1"))
    rows["cdnod"] = (col(cd, "mean_norm_shd"), col(cd, "mean_skeleton_f1"))
    rows["cdnod_union"] = (col(cd, "union_norm_shd"), col(cd, "union_skeleton_f1"))
    rows["pc"] = (col(m, "pc_mean_norm_shd"), col(m, "pc_mean_skeleton_f1"))
    rows["castor"] = (col(cas, "mean_norm_shd"), col(cas, "mean_skeleton_f1"))
    for k in ("directlingam", "icalingam", "dagma_paper", "sortnregress", "randomregress"):
        rows[k] = (col(m, f"{k}_mean_norm_shd"), col(m, f"{k}_mean_skeleton_f1"))
    rows["single_oracle"] = (col(diag, "nshd_single_oracle"), None)
    info = open(R("RUN_INFO.txt")).read(); data = info.split("external: ")[1].split(" |")[0].strip()
    noise = pd.read_csv(os.path.join(ROOT, data, "index.csv")).set_index("scenario_id").sem_type
    out = {}
    for s in ("exp", "gumbel", "gauss"):
        keep = set(noise.index[noise == s])
        out[f"external/{s}"] = ({l: tuple(None if x is None else x[x.index.isin(keep)] for x in v) for l, v in rows.items()}, None)
    return out


LABELS = {"method": METHOD, "dagma": "DAGMA, tuned, same windows", "cdnod": "CD-NOD (Fisher-z), per regime", "cdnod_union": "CD-NOD (Fisher-z), vs.\\ union graph",
          "pc": "PC, same windows", "castor": "CASTOR (linear EM)", "directlingam": "DirectLiNGAM, same windows", "icalingam": "ICA-LiNGAM, same windows",
          "dagma_paper": "DAGMA, published setting, same windows", "sortnregress": "sortnregress, same windows", "randomregress": "random-order regression, same windows",
          "empty": "Empty graph", "single_oracle": "Best single DAG (oracle)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tex_dir", required=True); a = ap.parse_args()
    os.makedirs(a.tex_dir, exist_ok=True)
    numbers = {}
    columns = {c: grid_rows(c) for c in CELLS}
    ext = external_rows(); columns.update(ext)
    cols = list(columns)

    # ── paired statistics per column ──
    stats = {c: {} for c in cols}
    for c, (rows, _) in columns.items():
        ref = rows["method"][0]
        for k, (s, _) in rows.items():
            if k != "method" and s is not None:
                stats[c][k] = paired_stats(ref, s)
        adj = holm([stats[c][k]["p"] if stats[c].get(k) else np.nan for k in stats[c]])
        for k, pa in zip(list(stats[c]), adj):
            if stats[c][k]:
                stats[c][k]["p_holm"] = pa
    numbers["paired"] = stats

    # ── Table 1: mean (SD) nSHD, main rows ──
    main_rows = ["method", "dagma", "cdnod", "castor", "pc", "directlingam"]
    L = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", "Method & " + " & ".join(CELL_TEX[c] for c in cols) + r" \\", r"\midrule"]
    numbers["table1"] = {}
    for k in main_rows:
        cells = []
        for c in cols:
            s = columns[c][0][k][0]
            if s is None or not len(s.dropna()):
                cells.append("--"); continue
            v = ms(s); best = all((columns[c][0][o][0] is None or not len(columns[c][0][o][0].dropna()) or columns[c][0][o][0].mean() >= s.mean()) for o in main_rows if o != k and o != "castor") and k != "castor"
            cells.append((r"\textbf{" + v + "}") if best else v)
            numbers["table1"].setdefault(k, {})[c] = {"mean": float(s.mean()), "sd": float(s.std(ddof=1)), "median": float(s.median()), "n": int(len(s.dropna()))}
        L.append(MAIN_LABELS.get(k, LABELS[k]) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.tex_dir, "tab_main_rows.tex"), "w").write("\n".join(L) + "\n")

    # ── skeleton F1 (appendix) and extra learners (appendix) ──
    for name, keys, which in (("tab_skeleton_rows.tex", main_rows + ["cdnod_union"], 1),
                              ("tab_learners_rows.tex", ["method", "directlingam", "icalingam", "dagma_paper", "sortnregress", "randomregress", "single_oracle", "cdnod_union"], 0)):
        L = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", "Method & " + " & ".join(CELL_TEX[c] for c in cols) + r" \\", r"\midrule"]
        for k in keys:
            cells = []
            for c in cols:
                s = columns[c][0][k][which]
                cells.append(ms(s) if s is not None and len(s.dropna()) else "--")
                if s is not None and len(s.dropna()):
                    numbers.setdefault("skeleton" if which else "learners", {}).setdefault(k, {})[c] = {"mean": float(s.mean()), "sd": float(s.std(ddof=1))}
            L.append(LABELS[k] + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(a.tex_dir, name), "w").write("\n".join(L) + "\n")

    # ── full statistics table (appendix): one block per column ──
    HEAD = [r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Cell & Row & $n$ & mean diff.\ [95\% CI] & $P$ (Holm) & $r_{rb}$ & lower / higher \\", r"\midrule"]
    L = list(HEAD); L_ext = list(HEAD)
    for c in cols:
        if c.startswith("external") and L is not L_ext:
            L += [r"\bottomrule", r"\end{tabular}"]
            open(os.path.join(a.tex_dir, "tab_stats_rows.tex"), "w").write("\n".join(L) + "\n")
            L = L_ext
        for k, st in stats[c].items():
            if not st or k not in main_rows + ["single_oracle"]:
                continue
            ptxt = "$<0.001$" if st["p"] < 0.001 else f"{st['p']:.3f}"
            ptxt += " (" + ("$<0.001$" if st["p_holm"] < 0.001 else f"{st['p_holm']:.3f}") + ")"
            L.append(f"{CELL_ROW[c]} & {LABELS[k]} & {st['n']} & ${st['mean']:+.3f}$ [{st['lo']:+.3f}, {st['hi']:+.3f}] & "
                     f"{ptxt} & ${st['rb']:+.2f}$ & {st['lower']} / {st['higher']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.tex_dir, "tab_stats_ext_rows.tex"), "w").write("\n".join(L) + "\n")

    # ── ablations (600) ──
    abl = {}
    L = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule", "Configuration & " + " & ".join(CELL_TEX[c] for c in CELLS) + r" \\", r"\midrule"]
    steps = [("main600", "\\method{} (600 held-out)"), ("nostage0", "without Stage 0 (preconditioner + DirectLiNGAM), same windows"),
             ("dl600", "without Stage 0 and without preconditioning (DirectLiNGAM alone), same windows"),
             ("oracle_seg", "with the true change points"), ("no_seg", "without segmentation"), ("L1000", "$L = 1{,}000$"), ("L2000", "$L = 2{,}000$"), ("L2500", "$L = 2{,}500$")]
    D = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule", "Configuration & " + " & ".join(CELL_TEX[c] for c in CELLS) + r" \\", r"\midrule"]
    for step, lab in steps:
        if all(load(c, "main600" if step == "dl600" else step) is None for c in CELLS):
            continue
        cells = []; diffs = []; abl[step] = {}
        for c in CELLS:
            d = load(c, "main600" if step == "dl600" else step); ref = load(c, "main600")
            if step == "dl600" and d is not None:   # the learner alone on the windows of main600 (its ablation columns)
                d = d.assign(mean_norm_shd=d.directlingam_mean_norm_shd, mean_skeleton_f1=d.directlingam_mean_skeleton_f1)
            if d is None:
                cells.append("--"); diffs.append("--"); continue
            cells.append(ms(d.mean_norm_shd))
            st = paired_stats(ref.mean_norm_shd, d.mean_norm_shd) if step != "main600" else None   # method - configuration
            abl[step][c] = {"mean": float(d.mean_norm_shd.mean()), "sd": float(d.mean_norm_shd.std(ddof=1)), "skel": float(d.mean_skeleton_f1.mean()), "stats": st}
            diffs.append(f"${-st['mean']:+.3f}$ [{-st['hi']:+.3f}, {-st['lo']:+.3f}]" if st else "--")   # configuration - method
        L.append(lab + " & " + " & ".join(cells) + r" \\")
        if step != "main600":
            D.append(lab + " & " + " & ".join(diffs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]; D += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.tex_dir, "tab_ablation_rows.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(a.tex_dir, "tab_ablation_diff_rows.tex"), "w").write("\n".join(D) + "\n")
    numbers["ablation"] = abl

    # ── detection (600) ──
    det = {}
    L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Cell & Segmentation & micro F1 & P & R & macro F1 [95\% CI] & conf.\ F1 & win./regime & $\nshd$ \\", r"\midrule"]
    dsteps = [("det_v1O", "correlation detector"), ("det_v1O_rank", "correlation detector + validation"), ("det_pelt", "PELT"), ("det_pelt_rank", "PELT + validation"),
              ("main600", r"union + validation (\method{})"), ("oracle_seg", "true change points")]
    for c in CELLS:
        det[c] = {}
        for step, lab in dsteps:
            d = load(c, step)
            if d is None:
                continue
            F, P, Rc = f1_of(d.det_tp.sum(), d.det_fp.sum(), d.det_fn.sum())
            cF, _, _ = f1_of(d.chow0001_tp.sum(), d.chow0001_fp.sum(), d.chow0001_fn.sum())
            mf, mlo, mhi = macro_f1(d)
            wr = float((d.n_structures / d.n_regimes).mean())
            det[c][step] = dict(micro_f1=F, precision=P, recall=Rc, macro_f1=mf, macro_lo=mlo, macro_hi=mhi, conf_f1=cF, windows_per_regime=wr, nshd=float(d.mean_norm_shd.mean()))
            if step == "oracle_seg":
                continue
            L.append(f"{CELL_ROW[c]} & {lab} & {F:.3f} & {P:.2f} & {Rc:.2f} & {mf:.3f} [{mlo:.3f}, {mhi:.3f}] & {cF:.3f} & {wr:.2f} & {d.mean_norm_shd.mean():.3f} \\\\")
        L.append(r"\midrule")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.tex_dir, "tab_detection_rows.tex"), "w").write("\n".join(L) + "\n")
    numbers["detection"] = det
    # compact version for the main text: macro F1 (precision / recall) and nSHD per cell
    M = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule", "Segmentation & " + " & ".join(CELL_TEX[c] for c in CELLS) + r" \\", r"\midrule"]
    for step, lab in dsteps:
        if step in ("oracle_seg", "det_v1O_rank", "det_pelt"):   # the full table (appendix) keeps every configuration
            continue
        cells = []
        for c in CELLS:
            r = det[c].get(step)
            cells.append(f"{r['macro_f1']:.3f} [{r['macro_lo']:.3f}, {r['macro_hi']:.3f}]" if r else "--")
        M.append(lab + " & " + " & ".join(cells) + r" \\")
    M += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.tex_dir, "tab_detection_main.tex"), "w").write("\n".join(M) + "\n")

    # ── routing shares, transform failures ──
    routing = {}
    for c in cols:
        m = columns[c][1] if c in CELLS else None
        if c in CELLS:
            m = load(c, "main")
        else:
            m = load("external", "main")
            info = open(R("RUN_INFO.txt")).read(); data = info.split("external: ")[1].split(" |")[0].strip()
            noise = pd.read_csv(os.path.join(ROOT, data, "index.csv")).set_index("scenario_id").sem_type
            m = m[noise.reindex(m.index) == c.split("/")[1]]
        W = m.n_structures.sum()
        routing[c] = {"stage0": float(m.n_stage0_windows.sum() / W), "pc": float(m.n_pc_windows.sum() / W),
                      "lingam": float(1 - (m.n_stage0_windows.sum() + m.n_pc_windows.sum()) / W), "windows": int(W),
                      "transform_failed": int(m.n_transform_failed.sum()), "gate": float(m.mean_gate_share.mean()), "curvature": float(m.mean_curvature.mean()),
                      "n": int(len(m))}
    numbers["routing"] = routing
    L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Cell & windows & LiNGAM path & Stage 0 (ANM) & CPDAG \\", r"\midrule"]
    for c in cols:
        r = routing[c]
        L.append(f"{CELL_ROW[c]} & {r['windows']:,} & {100 * r['lingam']:.0f}\\% & {100 * r['stage0']:.0f}\\% & {100 * r['pc']:.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.tex_dir, "tab_routing_rows.tex"), "w").write("\n".join(L) + "\n")

    # ── diagnostics (benchmark) ──
    diag = {}
    for c in cols:
        d = load(c, "diagnostics") if c in CELLS else load("external", "diagnostics")
        if d is None:
            continue
        if c not in CELLS:
            info = open(R("RUN_INFO.txt")).read(); data = info.split("external: ")[1].split(" |")[0].strip()
            noise = pd.read_csv(os.path.join(ROOT, data, "index.csv")).set_index("scenario_id").sem_type
            d = d[noise.reindex(d.index) == c.split("/")[1]]
        diag[c] = {k: {"mean": float(d[k].mean()), "sd": float(d[k].std(ddof=1))} for k in ("varsortability", "r2sortability", "cubic_r2_gain", "density", "nshd_empty", "nshd_single_oracle")}
        diag[c]["edge_flips"] = int(d.edge_flips.sum()); diag[c]["n"] = int(len(d))
    numbers["diagnostics"] = diag
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Cell & varsortability & $R^2$-sortability & cubic $R^2$ gain & density & best single DAG $\nshd$ \\", r"\midrule"]
    for c in cols:
        if c not in diag:
            continue
        r = diag[c]; f = lambda k: f"{r[k]['mean']:.3f} ({r[k]['sd']:.3f})"
        L.append(f"{CELL_ROW[c]} & {f('varsortability')} & {f('r2sortability')} & {f('cubic_r2_gain')} & {f('density')} & {f('nshd_single_oracle')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.tex_dir, "tab_diagnostics_rows.tex"), "w").write("\n".join(L) + "\n")

    # ── CASTOR on its ids ──
    cas = {}
    for c in CELLS + ["external"]:
        d = load(c, "castor"); m = load(c, "main")
        if d is None or m is None:
            continue
        ids = m.index.intersection(d.index)
        cas[c] = {"n": int(len(ids)), "castor": float(d.mean_norm_shd.reindex(ids).mean()), "method": float(m.mean_norm_shd.reindex(ids).mean()),
                  "castor_skel": float(d.mean_skeleton_f1.reindex(ids).mean()),
                  "regimes_found_ratio": float((d.n_regimes_found / d.n_regimes).reindex(ids).mean()),
                  "det_f1": f1_of(d.det_tp.reindex(ids).sum(), d.det_fp.reindex(ids).sum(), d.det_fn.reindex(ids).sum())[0],
                  "timeouts": int((pd.read_csv(R(c, "castor", "results.csv")).status == "timeout").sum())}
    numbers["castor"] = cas
    L = [r"\begin{tabular}{lrrccccr}", r"\toprule",
         r"Cell & $n$ & t/o & CASTOR & \method{} & diff.\ [95\% CI] & skel.\ F1 & cp F1 \\", r"\midrule"]
    for c in CELLS + ["external"]:
        if c not in cas:
            continue
        r = cas[c]
        if c == "external":
            st = [stats[k]["castor"] for k in cols if k.startswith("external") and stats[k].get("castor")]
            dtxt = ", ".join(f"${x['mean']:+.3f}$" for x in st)
        else:
            x = stats[c].get("castor"); dtxt = f"${x['mean']:+.3f}$ [{x['lo']:+.3f}, {x['hi']:+.3f}]" if x else "--"
        L.append(f"{CELL_ROW.get(c, 'external (all)')} & {r['n']} & {r['timeouts']} & {r['castor']:.3f} & {r['method']:.3f} & {dtxt} & {r['castor_skel']:.2f} & {r['det_f1']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.tex_dir, "tab_castor_rows.tex"), "w").write("\n".join(L) + "\n")

    json.dump(numbers, open(os.path.join(a.tex_dir, "numbers.json"), "w"), indent=1, default=float)
    print("tables written to", a.tex_dir)
    print(json.dumps({k: numbers["table1"][k] for k in numbers["table1"]}, indent=0)[:3000])


if __name__ == "__main__":
    main()
