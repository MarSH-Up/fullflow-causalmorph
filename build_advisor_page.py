#!/usr/bin/env python3
"""
Fill docs/advisor_page_template.html with the final figures (as data URIs)
and the before/after tables computed from the validation rows, and write the
page to publish.

Usage:
    python build_advisor_page.py --before_row C0_legacy_v2 --det_row D_v1L_g2.5 \
        --after_row LB_v1L25_v2 --after_method icalingam \
        --figures results/hardening_validation/figures --out results/hardening_validation/advisor_page.html
"""
import argparse
import base64
import os
import subprocess

import pandas as pd

from analyze_hardening_validation import load, paired, HEAD
from plot_before_after import remap_learner, micro

LABELS = {"icalingam": "v1L change points, cold start, no bootstrap, no overlap, ICA-LiNGAM per window",
          "directlingam": "v1L change points, cold start, no bootstrap, no overlap, DirectLiNGAM per window",
          "pipeline": "v1L change points, cold start, no bootstrap, no overlap, CausalMorph + chain prior"}


def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def detection_table(before, after):
    rows = []
    for tol in (25, 50, 100, 200):
        pb, rb, fb = micro(before, tol); pa, ra, fa = micro(after, tol)
        rows.append((f"tolerance {tol}", pb, pa, rb, ra, fb, fa))
    h = "<tr><th></th><th class='num'>precision before</th><th class='num'>after</th><th class='num'>recall before</th><th class='num'>after</th><th class='num'>F1 before</th><th class='num'>after</th></tr>"
    body = "".join(f"<tr><td>{n}</td><td class='num'>{pb:.3f}</td><td class='num {'good' if pa>=pb else 'bad'}'>{pa:.3f}</td>"
                   f"<td class='num'>{rb:.3f}</td><td class='num {'good' if ra>=rb else 'bad'}'>{ra:.3f}</td>"
                   f"<td class='num'>{fb:.3f}</td><td class='num {'good' if fa>=fb else 'bad'}'>{fa:.3f}</td></tr>"
                   for n, pb, pa, rb, ra, fb, fa in rows)
    shares = []
    for lab, fn in (("under-segmented", lambda d: (d.n_structures < d.n_regimes).mean()),
                    ("over-segmented", lambda d: (d.n_structures > d.n_regimes).mean()),
                    ("no change point detected", lambda d: (d.n_detected_cps == 0).mean())):
        shares.append(f"<tr><td>{lab}</td><td class='num'>{fn(before):.1%}</td><td class='num'>{fn(after):.1%}</td><td colspan='4'></td></tr>")
    return f"<table>{h}{body}{''.join(shares)}</table>"


def structure_table(before, after):
    h = "<tr><th>metric</th><th class='num'>before</th><th class='num'>after</th><th class='num'>paired change</th><th class='num'>win / loss</th></tr>"
    body = []
    for col, m, hib in HEAD:
        if col not in before or col not in after:
            continue
        r = paired(before, after, col, hib)
        if not r:
            continue
        cls = "good" if r["rel"] >= 0 else "bad"
        body.append(f"<tr><td>{m} {'↑' if hib else '↓'}</td><td class='num'>{r['ref']:.3f}</td><td class='num {cls}'>{r['other']:.3f}</td>"
                    f"<td class='num {cls}'>{r['rel']:+.1f}%</td><td class='num'>{r['wins']} / {r['losses']}</td></tr>")
    return f"<table>{h}{''.join(body)}</table>"


def sweep_table(val, ref_name, rows):
    ref = load(os.path.join(val, ref_name, "results.csv"))
    h = "<tr><th>extraction, v1L windows, paired vs no bootstrap / no overlap</th><th class='num'>per-regime precision</th><th class='num'>per-regime recall</th><th class='num'>windowed nSHD</th><th class='num'>consensus F1</th></tr>"
    body = []
    for label, name in rows:
        d = load(os.path.join(val, name, "results.csv"))
        if d is None:
            continue
        cells = []
        for col, hib in (("mean_struct_precision", True), ("mean_struct_recall", True),
                         ("windowed_nshd_weighted", False), ("consensus_f1", True)):
            r = paired(ref, d, col, hib)
            v = r["rel"] if r else float("nan")
            cells.append(f"<td class='num {'good' if v >= 0 else 'bad'}'>{v:+.1f}%</td>")
        body.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    return f"<table>{h}{''.join(body)}</table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="results/hardening_validation")
    ap.add_argument("--before_row", required=True); ap.add_argument("--det_row", required=True)
    ap.add_argument("--after_row", required=True); ap.add_argument("--after_method", default="icalingam")
    ap.add_argument("--figures", default="results/hardening_validation/figures")
    ap.add_argument("--template", default="docs/advisor_page_template.html")
    ap.add_argument("--out", default="results/hardening_validation/advisor_page.html")
    a = ap.parse_args()

    before = load(os.path.join(a.val, a.before_row, "results.csv"))
    det = load(os.path.join(a.val, a.det_row, "results.csv"))
    after = remap_learner(load(os.path.join(a.val, a.after_row, "results.csv")), a.after_method)
    common = set(before.index) & set(det.index) & set(after.index)
    before, det, after = before[before.index.isin(common)], det[det.index.isin(common)], after[after.index.isin(common)]

    pb, rb, fb = micro(before); pa, ra, fa = micro(det)
    f1_b = before.mean_struct_f1.mean(); f1_a = after.mean_struct_f1.mean()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    subs = {
        "AFTER_LABEL": LABELS[a.after_method],
        "DET_R_BEFORE": f"{rb:.2f}", "DET_R_AFTER": f"{ra:.2f}", "DET_P_BEFORE": f"{pb:.2f}", "DET_P_AFTER": f"{pa:.2f}",
        "F1_BEFORE": f"{f1_b:.2f}", "F1_AFTER": f"{f1_a:.2f}",
        "WN_BEFORE": f"{before.windowed_nshd_weighted.mean():.3f}", "WN_AFTER": f"{after.windowed_nshd_weighted.mean():.3f}",
        "TABLE_DETECTION": detection_table(before, det),
        "TABLE_STRUCTURE": structure_table(before, after),
        "TABLE_SWEEP": sweep_table(a.val, "L2_cold_v1L25_v2", [
            ("window overlap 0.15 alone", "X_overlap_only_v1L25_v2"),
            ("bootstrap 20 @ 0.30 + overlap", "X_boot_t030_v1L25_v2"),
            ("bootstrap 20 @ 0.40 + overlap", "X_boot_t040_v1L25_v2"),
            ("bootstrap 20 @ 0.50 + overlap", "X_boot_t050_v1L25_v2"),
            ("bootstrap 20 @ 0.60 + overlap (October)", "L3_cold_v1L25_v2"),
            ("bootstrap 20 @ 0.60, no overlap", "X_boot_t060_noov_v1L25_v2")]),
        "HEAD_SHA": head,
    }
    for key, fname in (("FIG_A", "A_detection_before_after.png"), ("FIG_B", "B_detection_tolerance.png"),
                       ("FIG_C", "C_segmentation_outcome.png"), ("FIG_D", "D_structure_before_after.png"),
                       ("FIG_E", "E_consensus_by_windows.png"), ("FIG_F", "F_decomposition.png")):
        subs[key] = data_uri(os.path.join(a.figures, fname))

    html = open(a.template, encoding="utf-8").read()
    for k, v in subs.items():
        html = html.replace("{{" + k + "}}", v)
    left = [t for t in ("{{",) if t in html]
    if left:
        raise SystemExit("unfilled placeholders remain in the page")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(html)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
