#!/usr/bin/env python3
"""
Would a segment-level criterion remove the detector's false alarms? For every scenario of a
run (its detected_cps and true_cps), each detected cut is labelled true/false (within the 200-
sample tolerance) and scored by the two segments it separates, bounded by its neighbouring
cuts: the BIC gain of splitting under the Gaussian mean-and-covariance cost PELT uses
(positive = the split pays its penalty), the same with correlation matrices only (segments
standardised), and the Frobenius distance between the two correlation matrices. The report
gives the detection F1 after keeping only the cuts whose gain exceeds a threshold, per cell.

  python tools/cut_validation_probe.py results/stage0_check v2_auto name=dir ...
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from run_experiments import load_scenario  # noqa: E402

TOL = 200


def parse(s):
    return [int(x) for x in str(s).split(";") if x.strip() and x != "nan"]


def logdet_cost(S):
    n = len(S)
    C = np.cov(S, rowvar=False) + 1e-6 * np.eye(S.shape[1])
    return n * np.linalg.slogdet(C)[1]


def corr_cost(S):
    Z = (S - S.mean(0)) / (S.std(0) + 1e-12)
    return logdet_cost(Z)


def rank_gain(A, B, pen):
    """BIC gain of the split under the Gaussian cost on the joint ranks of the two segments (heavy tails flattened)."""
    from scipy.stats import rankdata, norm
    AB = np.vstack([A, B]); n = len(AB)
    Z = norm.ppf(rankdata(AB, axis=0) / (n + 1.0))
    return logdet_cost(Z) - logdet_cost(Z[:len(A)]) - logdet_cost(Z[len(A):]) - pen


def label_cuts(true, det):
    """true/false label of every detected cut: greedy one-to-one matching within TOL."""
    used, lab = set(), []
    for c in sorted(det):
        cand = [(abs(c - t), j) for j, t in enumerate(true) if j not in used and abs(c - t) <= TOL]
        if cand:
            used.add(min(cand)[1]); lab.append(True)
        else:
            lab.append(False)
    return lab


def main(out, cfg, cells):
    rows = []
    for c in cells:
        name, ds = c.split("=", 1)
        d = pd.read_csv(os.path.join(out, name, cfg, "results.csv")); d = d[d.status == "ok"]
        index = pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).set_index("scenario_id")
        for _, r in d.iterrows():
            X, names, true_cps, _, _, _ = load_scenario(r.scenario_id, ds, index.loc[r.scenario_id], permute_variables=True)
            det = sorted(parse(r.detected_cps)); T, p = X.shape
            if not det:
                rows.append({"cell": name, "scenario_id": r.scenario_id, "cut": np.nan, "true": np.nan, "n_true": len(true_cps)})
                continue
            labels = label_cuts(true_cps, det)
            bounds = [0] + det + [T]
            pen = (p + p * (p + 1) / 2) * np.log(T)
            for k, c in enumerate(det):
                A, B = X[bounds[k]:c], X[c:bounds[k + 2]]
                AB = X[bounds[k]:bounds[k + 2]]
                if len(A) <= p + 2 or len(B) <= p + 2:
                    gain = gain_c = gain_r = np.nan
                else:
                    gain = logdet_cost(AB) - logdet_cost(A) - logdet_cost(B) - pen
                    gain_c = corr_cost(AB) - corr_cost(A) - corr_cost(B) - (p * (p - 1) / 2) * np.log(T)
                    gain_r = rank_gain(A, B, pen)
                RA, RB = np.corrcoef(A.T), np.corrcoef(B.T)
                rows.append({"cell": name, "scenario_id": r.scenario_id, "cut": c, "true": labels[k], "n_true": len(true_cps),
                             "gain": gain, "gain_corr": gain_c, "gain_rank": gain_r, "frob": float(np.linalg.norm(np.nan_to_num(RA - RB)))})
    D = pd.DataFrame(rows); D.to_csv(os.path.join(out, f"cut_validation_{cfg}.csv"), index=False)
    pd.set_option("display.width", 220)
    cuts = D.dropna(subset=["cut"])
    print("detected cuts: BIC gain of the split (median) for true and false cuts, and share of each above 0")
    g = cuts.groupby(["cell", "true"])
    print(pd.DataFrame({"n": g.size(), "gain med": g.gain.median().round(1), "gain>0": (cuts.gain > 0).groupby([cuts.cell, cuts["true"]]).mean().round(2),
                        "rank gain med": g.gain_rank.median().round(1), "rank gain>0": (cuts.gain_rank > 0).groupby([cuts.cell, cuts["true"]]).mean().round(2),
                        "frob med": g.frob.median().round(2)}).to_string())
    print("\ndetection F1 (precision/recall) keeping only cuts with gain > thr; thr=-inf is the detector as it is")
    res = []
    n_true = D.groupby("scenario_id").n_true.first()
    for col, thrs in (("gain", (-np.inf, 0)), ("gain_rank", (0, 2 * 1e2, 5 * 1e2)), ("gain_corr", (0,))):
        for thr in thrs:
            r = {"criterion": col, "thr": thr}
            for name in dict.fromkeys(D.cell):
                x = cuts[cuts.cell == name]
                keep = x[(x[col] > thr) | (x[col].isna())] if np.isfinite(thr) else x
                tp = int(keep["true"].sum()); fp = len(keep) - tp
                fn = int(n_true[D[D.cell == name].scenario_id.unique()].sum()) - tp
                P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
                r[name] = f"{(2 * P * R / (P + R) if P + R else 0):.3f} ({P:.2f}/{R:.2f})"
            res.append(r)
    print(pd.DataFrame(res).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
