#!/usr/bin/env python3
"""
Detection-only comparison of the correlation detector's variants (v1L Pearson
vs v1M Spearman) over a gamma grid, with the same window sizes, refractory
period, post-processing (detection_cps) and 200-sample tolerance as
run_experiments.py. The v1G wavelet fallback (used only when the correlation
detector finds nothing) is not run; such scenarios are counted as `empty`.

--combos gP:gS adds the "both" rule: a change point is kept only when Pearson
(at gamma gP) and Spearman (at gamma gS) each find one within one window W of
it; it is placed where Pearson put it ("both_p") or where Spearman did ("both_s").
--unions gP:gS adds the "union" rule (v1O): every Pearson change point, plus the
Spearman ones with no Pearson change point within W.

    python tools/detector_sweep.py --dataset_dir datasets_v2_nonlin_laplace \
        --scenario_ids_file results/hardening_validation/sample_ids.txt \
        --out results/hardening_validation/detector_sweep_nonlin_laplace.csv
"""
import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "NSD_Wavelets", "src"), os.path.join(ROOT, "causalmorph")):
    if path not in sys.path:
        sys.path.insert(0, path)

VARIANTS = {"v1L": False, "v1M": True}   # name -> rank


def _both(cps_a, cps_b, radius):
    """Change points of cps_a that have a change point of cps_b within radius."""
    return [c for c in cps_a if any(abs(c - d) <= radius for d in cps_b)]


def _union(cps_a, cps_b, radius):
    """cps_a plus the change points of cps_b that have none of cps_a within radius."""
    return sorted(list(cps_a) + [c for c in cps_b if not any(abs(c - d) <= radius for d in cps_a)])


def _job(sid, index_row, dataset_dir, gammas, tolerance, detectors, combos=(), unions=()):
    import warnings as _w
    _w.filterwarnings("ignore")
    from detectors.detectors_correlation import detect_correlation_changes
    from evaluation.metrics import evaluate_detection, simplify_transitions
    from run_experiments import load_scenario

    X, _, true_cps, _, regime_sizes, _ = load_scenario(sid, dataset_dir, index_row)
    T = len(X)
    min_regime_len = min(regime_sizes) if regime_sizes else T // max(int(index_row["n_regimes"]), 1)
    refractory = max(150, min_regime_len // 4)
    window = max(50, min_regime_len // 5)
    cache = {}

    def cps_for(name, g):
        if (name, g) not in cache:
            cache[(name, g)], _, _ = detect_correlation_changes(
                X, window=window, refractory_period=refractory, edge_margin=window,
                statistic="max", gamma=g, studentize=True, refine_scale=True, rank=VARIANTS[name])
        return cache[(name, g)]

    rows = []

    def score(name, g, cps):
        det = simplify_transitions(sorted(cps), min_distance=160)
        ev = evaluate_detection(true_cps, det, tolerance=tolerance)
        rows.append(dict(scenario_id=sid, p=int(index_row["p"]), samples_regime=index_row.get("samples_regime"),
                         detector=name, gamma=g, n_true=len(true_cps), n_det=len(det), empty=int(not det),
                         tp=ev.true_positives, fp=ev.false_positives, fn=ev.false_negatives))

    for name in detectors:
        for g in gammas:
            score(name, g, cps_for(name, g))
    for gp, gs in combos:
        pear, spear = cps_for("v1L", gp), cps_for("v1M", gs)
        score("both_p", f"{gp}:{gs}", _both(pear, spear, window))
        score("both_s", f"{gp}:{gs}", _both(spear, pear, window))
    for gp, gs in unions:
        score("union", f"{gp}:{gs}", _union(cps_for("v1L", gp), cps_for("v1M", gs), window))
    return rows


def summarise(df):
    g = df.groupby(["detector", "gamma"]).agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                                              n_det=("n_det", "sum"), n_true=("n_true", "sum"),
                                              empty=("empty", "sum"))
    exact = df.assign(ok=df.n_det == df.n_true).groupby(["detector", "gamma"]).ok.mean()
    g["precision"] = g.tp / (g.tp + g.fp)
    g["recall"] = g.tp / (g.tp + g.fn)
    g["f1"] = 2 * g.precision * g.recall / (g.precision + g.recall)
    g["exact_count"] = exact
    g["det_per_true"] = g.n_det / g.n_true
    return g[["precision", "recall", "f1", "exact_count", "det_per_true", "empty"]].round(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--scenario_ids_file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gammas", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5])
    ap.add_argument("--detectors", nargs="*", choices=list(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--combos", nargs="*", default=[], help="gP:gS pairs for the 'both' rule, e.g. 2.5:2.0")
    ap.add_argument("--unions", nargs="*", default=[], help="gP:gS pairs for the 'union' rule, e.g. 2.5:2.0")
    ap.add_argument("--tolerance", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    combos = [tuple(float(v) for v in c.split(":")) for c in a.combos]
    unions = [tuple(float(v) for v in c.split(":")) for c in a.unions]

    index = pd.read_csv(os.path.join(a.dataset_dir, "index.csv"), dtype=str)
    if a.scenario_ids_file:
        keep = {ln.strip() for ln in open(a.scenario_ids_file) if ln.strip()}
        index = index[index.scenario_id.isin(keep)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_job, r.scenario_id, r, a.dataset_dir, a.gammas, a.tolerance, a.detectors, combos, unions) for _, r in index.iterrows()]
        rows = [row for f in futs for row in f.result()]
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"{a.dataset_dir}  ({df.scenario_id.nunique()} scenarios)")
    print(summarise(df).to_string())


if __name__ == "__main__":
    main()
