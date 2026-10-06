#!/usr/bin/env python3
"""
Early-validation analysis for docs/2026-10-06_improvement_plan.md.

Reads the rows written by run_hardening_validation.sh, restricts the October-5
full-grid results to the same scenarios as the "before" reference, and prints:

  1. detection: micro P/R/F1 and the recall-vs-tolerance curve per detector
     row, before vs after, with under-segmentation and zero-detection rates
  2. structure ladder: before (oracle anchor, old consensus, old pruning) vs
     L2/L3 cold vs the controls, paired on scenario_id against L3_cold
  3. the error decomposition of plan §3.2: delta_det = L3 - C2,
     delta_anchor = L3 - C1, and the bootstrap ablation L3 - L2 (cold) next to
     the old winning - baseline on the same scenarios
  4. consensus recall by number of extraction windows (the n <= 2 check)
  5. the plan's §4 predictions next to what was observed

Usage:
    python analyze_hardening_validation.py [--val results/hardening_validation]
                                           [--before results/results_october_5]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

HEAD = [("mean_struct_precision", "per-regime precision", True),
        ("mean_struct_recall",    "per-regime recall",    True),
        ("mean_norm_shd",         "mean nSHD",            False),
        ("windowed_nshd_weighted","windowed nSHD",        False),
        ("transition_shd",        "transition SHD",       False),
        ("dynamic_score",         "dynamic score",        False),
        ("consensus_precision",   "consensus precision",  True),
        ("consensus_recall",      "consensus recall",     True),
        ("consensus_f1",          "consensus F1",         True)]

_num = re.compile(r"-?\d+(?:\.\d+)?")


def parse(s):
    return [] if pd.isna(s) else [int(float(x)) for x in _num.findall(str(s))]


def load(path, ids=None):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d.status == "ok"].drop_duplicates("scenario_id", keep="first").set_index("scenario_id")
    if ids is not None:
        d = d[d.index.isin(ids)]
    return d


# ── detection ─────────────────────────────────────────────────────────────────

def match(tc, dc, tol):
    tc, dc = sorted(tc), sorted(dc); used = set(); tp = 0
    for t in tc:
        best, bd = None, tol + 1
        for j, x in enumerate(dc):
            if j not in used and abs(x - t) <= tol and abs(x - t) < bd:
                best, bd = j, abs(x - t)
        if best is not None:
            used.add(best); tp += 1
    return tp, len(dc) - tp, len(tc) - tp


def detection_table(rows):
    out = []
    for name, d in rows.items():
        if d is None:
            continue
        rec = {"row": name, "n": len(d)}
        for tol in (25, 50, 100, 200):
            TP = FP = FN = 0
            for tc, dc in zip(d.true_cps.map(parse), d.detected_cps.map(parse)):
                a, b, c = match(tc, dc, tol); TP += a; FP += b; FN += c
            P = TP / (TP + FP) if TP + FP else 0.0
            R = TP / (TP + FN) if TP + FN else 0.0
            rec[f"R@{tol}"] = R
            if tol == 200:
                rec["P@200"] = P
                rec["F1@200"] = 2 * P * R / (P + R) if P + R else 0.0
        rec["zero_det"] = (d.n_detected_cps == 0).mean()
        rec["under_seg"] = (d.n_structures < d.n_regimes).mean()
        rec["over_seg"] = (d.n_structures > d.n_regimes).mean()
        if "n_cps_merged" in d:
            rec["merged/scn"] = d.n_cps_merged.mean()
        out.append(rec)
    t = pd.DataFrame(out).set_index("row")
    cols = ["n", "P@200", "R@25", "R@50", "R@100", "R@200", "F1@200", "zero_det", "under_seg", "over_seg"]
    cols += [c for c in ("merged/scn",) if c in t]
    return t[cols]


# ── structure ─────────────────────────────────────────────────────────────────

def paired(ref, other, col, hib):
    j = ref[[col]].join(other[[col]], rsuffix="_o", how="inner").dropna()
    if len(j) == 0:
        return None
    a, b = j[col].astype(float), j[f"{col}_o"].astype(float)
    diff = b - a
    gain = diff if hib else -diff
    w, l = int((gain > 0).sum()), int((gain < 0).sum())
    p = stats.wilcoxon(b, a).pvalue if w + l else float("nan")
    return dict(n=len(j), ref=a.mean(), other=b.mean(),
                rel=100 * gain.mean() / abs(a.mean()) if a.mean() else float("nan"),
                wins=w, losses=l, p=p)


def ladder_table(rows, ref_name="L3_cold"):
    ref = rows[ref_name]
    lines = []
    for name, d in rows.items():
        if d is None or name == ref_name:
            continue
        for col, label, hib in HEAD:
            if col not in d or col not in ref:
                continue
            r = paired(ref, d, col, hib)
            if r is None:
                continue
            lines.append(dict(row=name, metric=label, **r))
    return pd.DataFrame(lines)


def print_block(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", default="results/hardening_validation")
    ap.add_argument("--before", default="results/results_october_5")
    ap.add_argument("--ladder_suffix", default="",
                    help="suffix of the ladder/control rows to analyse, e.g. _v1K (detection rows are unaffected)")
    args = ap.parse_args()

    ids = {ln.strip() for ln in open(os.path.join(args.val, "sample_ids.txt")) if ln.strip()}
    before_w = load(os.path.join(args.before, "v1h_full_winning/results.csv"), ids)
    before_b = load(os.path.join(args.before, "v1h_full_baseline/results.csv"), ids)
    det_names = ["D_v1H", "D_v1H_adaptive", "D_v1H_fixedW500", "D_v1J"]
    lad_names = ["L2_cold", "C3_no_seg", "L3_cold", "C1_oracle_anchor", "C2_oracle_seg", "L4_soft_chain"]
    sweep = sorted(n for n in os.listdir(args.val)
                   if n.startswith(("D_v1J_g", "D_v1K_g", "D_v1L_g")) and os.path.exists(os.path.join(args.val, n, "results.csv")))
    names = det_names + [n for n in sweep if n not in det_names] + lad_names
    rows = {n: load(os.path.join(args.val, n, "results.csv")) for n in names if n.startswith("D_")}
    # ladder rows are keyed by their base name so the rest of the script is suffix-agnostic
    for n in lad_names:
        rows[n] = load(os.path.join(args.val, n + args.ladder_suffix, "results.csv"))
    if args.ladder_suffix:
        print(f"ladder rows: *{args.ladder_suffix}")
    done = [n for n, d in rows.items() if d is not None and len(d)]
    print("rows available:", ", ".join(f"{n}({len(rows[n])})" for n in done))
    print(f"before (Oct-5) on the same scenarios: winning={len(before_w)} baseline={len(before_b)}")

    # 1. detection ────────────────────────────────────────────────────────────
    print_block("1. DETECTION  (micro; before = Oct-5 run on the same scenarios, old post-processing)")
    det_rows = {"before_v1H(old)": before_w}
    det_rows.update({n: rows[n] for n in names if n.startswith("D_") and rows[n] is not None})
    print(detection_table(det_rows).round(3).to_string())

    # precision / recall by p for every detector row: tells whether one gamma
    # works across p or the null needs a p-dependent margin
    print("\n  P@200 / R@200 by p:")
    for name, d in det_rows.items():
        if d is None:
            continue
        cells = []
        for p_, g in d.groupby("p"):
            TP = FP = FN = 0
            for tc, dc in zip(g.true_cps.map(parse), g.detected_cps.map(parse)):
                a, b, c = match(tc, dc, 200); TP += a; FP += b; FN += c
            P = TP / (TP + FP) if TP + FP else 0.0; R = TP / (TP + FN) if TP + FN else 0.0
            cells.append(f"p{p_}:{P:.2f}/{R:.2f}")
        print(f"    {name:<18} " + "  ".join(cells))

    # recall by position of the true change point: the regime-0-calibrated
    # threshold made this decay 0.77 -> 0.35 on long regimes; the adaptive
    # null exists to flatten it.
    print("\n  recall by change-point position (R@200), all scenarios / only R>=5 & n=2500:")
    for name, d in det_rows.items():
        if d is None:
            continue
        for tag, sub in (("all", d), ("R>=5,n=2500", d[(d.n_regimes >= 5) & (d.samples_regime == 2500)])):
            hits = {}
            for tc, dc in zip(sub.true_cps.map(parse), sub.detected_cps.map(parse)):
                for k, t in enumerate(sorted(tc)):
                    hits.setdefault(k, []).append(any(abs(x - t) <= 200 for x in dc))
            if not hits:
                continue
            line = "  ".join(f"pos{k}={np.mean(v):.2f}(n={len(v)})" for k, v in sorted(hits.items()) if len(v) >= 10)
            print(f"    {name:<18} [{tag:<12}] {line}")

    if rows["D_v1H"] is not None:
        d = rows["D_v1H"]
        print("\n  D_v1H recall by samples/regime (R@200):")
        for n_, g in d.groupby("samples_regime"):
            TP = FP = FN = 0
            for tc, dc in zip(g.true_cps.map(parse), g.detected_cps.map(parse)):
                a, b, c = match(tc, dc, 200); TP += a; FP += b; FN += c
            print(f"    n={n_:>5}: R={TP/(TP+FN):.3f}  P={TP/(TP+FP) if TP+FP else 0:.3f}")

    # 2. ladder ───────────────────────────────────────────────────────────────
    if rows["L3_cold"] is not None:
        print_block("2. STRUCTURE: every row vs L3_cold (paired; rel>0 = that row is better than L3_cold)")
        lad = dict(rows); lad["before_winning(oracle,old)"] = before_w; lad["before_baseline(oracle,old)"] = before_b
        lad = {k: v for k, v in lad.items() if v is not None and not k.startswith("D_")}
        t = ladder_table(lad)
        if len(t):
            piv = t.pivot(index="metric", columns="row", values="rel").round(1)
            order = [m for _, m, _ in HEAD if m in piv.index]
            print(piv.loc[order].to_string())
            print("\n  (per-metric means)")
            means = pd.DataFrame({k: {m: v[c].mean() for c, m, _ in HEAD if c in v} for k, v in lad.items()})
            means["L3_cold"] = {m: rows["L3_cold"][c].mean() for c, m, _ in HEAD if c in rows["L3_cold"]}
            print(means.loc[order].round(3).to_string())

        # 3. decomposition ────────────────────────────────────────────────────
        print_block("3. DECOMPOSITION  (plan §3.2; mean paired differences on the headline metrics)")
        L3 = rows["L3_cold"]
        pairs = [("delta_anchor = L3_cold - C1_oracle_anchor  (size of the leak)", "C1_oracle_anchor"),
                 ("delta_det    = L3_cold - C2_oracle_seg     (cost of detection)", "C2_oracle_seg"),
                 ("ablation     = L3_cold - L2_cold           (bootstrap+overlap, cold)", "L2_cold"),
                 ("premise      = L3_cold - C3_no_seg         (vs ignoring non-stationarity)", "C3_no_seg"),
                 ("soft chain   = L4_soft_chain - L3_cold", "L4_soft_chain")]
        for label, other in pairs:
            o = rows.get(other)
            if o is None:
                continue
            print(f"\n  {label}")
            for col, m, hib in HEAD:
                if col not in o or col not in L3:
                    continue
                if other == "L4_soft_chain":
                    r = paired(L3, o, col, hib)
                else:
                    r = paired(o, L3, col, hib)
                if r:
                    sign = "+" if r["rel"] >= 0 else ""
                    print(f"    {m:<22} {r['ref']:.3f} -> {r['other']:.3f}  {sign}{r['rel']:.1f}%  "
                          f"win/loss {r['wins']}/{r['losses']}  p={r['p']:.1e}")
        if before_w is not None and before_b is not None and rows["L2_cold"] is not None:
            print("\n  old ablation on the same scenarios (oracle anchor, old consensus, old pruning): winning - baseline")
            for col, m, hib in HEAD:
                r = paired(before_b, before_w, col, hib)
                if r:
                    print(f"    {m:<22} {r['ref']:.3f} -> {r['other']:.3f}  {r['rel']:+.1f}%  win/loss {r['wins']}/{r['losses']}")

        # 4. consensus by window count ────────────────────────────────────────
        print_block("4. CONSENSUS RECALL BY NUMBER OF EXTRACTION WINDOWS  (recall == 1.000 at n<=2 was the bug)")
        for name, d in (("before_winning", before_w), ("L3_cold", L3)):
            if d is None:
                continue
            nw = d["n_windows_ext"] if "n_windows_ext" in d else d["n_structures"]
            g = d.groupby(nw.clip(upper=4)).agg(n=("consensus_recall", "size"),
                                                recall=("consensus_recall", "mean"),
                                                precision=("consensus_precision", "mean"),
                                                f1=("consensus_f1", "mean")).round(3)
            g.index.name = "windows(<=4)"
            print(f"\n  {name}\n{g.to_string()}")

        # 5. predictions ──────────────────────────────────────────────────────
        print_block("5. PLAN §4 PREDICTIONS vs OBSERVED (sample)")
        obs = {}
        dt = detection_table({"D": rows["D_v1H"]}) if rows["D_v1H"] is not None else None
        if dt is not None:
            obs["detection recall (micro)"] = ("0.465", "0.65–0.75", f"{dt.loc['D','R@200']:.3f}")
            obs["detection precision (micro)"] = ("0.739", "0.65–0.70", f"{dt.loc['D','P@200']:.3f}")
            obs["under-segmented scenarios"] = ("59%", "≈35%", f"{dt.loc['D','under_seg']:.0%}")
        n2 = L3[(L3.n_windows_ext <= 2)] if "n_windows_ext" in L3 else L3[L3.n_structures <= 2]
        if len(n2):
            obs["consensus recall, n<=2 windows"] = ("1.000", "≈0.41", f"{n2.consensus_recall.mean():.3f}")
            obs["consensus precision, n<=2"] = ("0.34", "≈0.48", f"{n2.consensus_precision.mean():.3f}")
        w0 = []
        for s in L3.cm_norm_shd_per_regime.dropna():
            v = [float(x) for x in str(s).split(";") if x not in ("", "nan")]
            if v: w0.append(v[0])
        if w0:
            obs["window-0 nSHD"] = ("0.136", "≈0.20", f"{np.mean(w0):.3f}")
        if rows["L2_cold"] is not None:
            r = paired(rows["L2_cold"], L3, "mean_struct_precision", True)
            obs["bootstrap ablation, per-regime precision"] = ("+9.4%", "+8 to +11%", f"{r['rel']:+.1f}%")
        if "directlingam_mean_norm_shd" in L3:
            a, b = L3.mean_norm_shd, L3.directlingam_mean_norm_shd
            m = a.notna() & b.notna()
            wr = (b[m] > a[m]).sum() / max(((b[m] != a[m]).sum()), 1)
            obs["L3 vs DirectLiNGAM, nSHD win-rate"] = ("99.4%", "85–95%", f"{wr:.1%}")
        t = pd.DataFrame(obs, index=["before", "predicted", "observed"]).T
        print(t.to_string())


if __name__ == "__main__":
    main()
