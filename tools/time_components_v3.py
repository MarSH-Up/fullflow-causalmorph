#!/usr/bin/env python3
"""
Per-component wall-clock of the revised method on one core (OMP/BLAS threads = 1): the
correlation detector (v1O), PELT, the rank-based cut validation, the extraction with Stage 0
routing (standardised windows, CausalMorph v2, DirectLiNGAM), the same extraction without
Stage 0, the structural confirmation, and DAGMA (paper configuration) on the same windows.
Sample: the 16 scenarios of results/timing/ids.txt (p in {3,6,9,12}, nominal regime length in
{500,2500}, two seeds) in lin_laplace (datasets/) and nonlin_normal (results/timing/ds_nonlin_normal).

  OMP_NUM_THREADS=1 python tools/time_components_v3.py          # writes results/timing_v3/timing.csv
"""
import os
import sys
import time
import warnings

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p_ in (ROOT, os.path.join(ROOT, "NSD_Wavelets", "src"), os.path.join(ROOT, "causalmorph")):
    sys.path.insert(0, p_)
from run_experiments import load_scenario, detection_cps, extraction_windows, min_window_for, chow_confirmed_cps  # noqa: E402
from full_pipeline import (detect_change_points, pelt_change_points, bic_filter_cps, extract_causal_structures,  # noqa: E402
                           _fit_window_learner, _standardize_window)

L = 500
OUT = os.path.join(ROOT, "results", "timing_v3"); os.makedirs(OUT, exist_ok=True)
ids = [l.strip() for l in open(os.path.join(ROOT, "results", "timing", "ids.txt")) if l.strip()]
rows = []
for cell, ds in (("lin_laplace", os.path.join(ROOT, "datasets")), ("nonlin_normal", os.path.join(ROOT, "results", "timing", "ds_nonlin_normal"))):
    index = pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).set_index("scenario_id")
    for sid in ids:
        X, names, true_cps, true_adjs, _, _ = load_scenario(sid, ds, index.loc[sid], permute_variables=True)
        T, p = X.shape
        r = {"cell": cell, "scenario_id": sid, "p": p, "regime_length": int(index.loc[sid, "samples_regime"]), "T": T}
        t0 = time.time(); det = detect_change_points(X, baseline_end=max(150, L // 4), min_regime_len=L, seed=0, detector_version="v1O",
                                                    corr_gamma=2.5, corr_gamma_rank=1.5); v1o = detection_cps(det); r["t_detector_v1O"] = time.time() - t0
        t0 = time.time(); pelt = pelt_change_points(X, min_size=L); r["t_pelt"] = time.time() - t0
        cand = sorted(set(v1o) | {c for c in pelt if not any(abs(c - d) < 160 for d in v1o)})
        t0 = time.time(); cps = bic_filter_cps(X, cand, "rank"); r["t_rank_validation"] = time.time() - t0
        r["n_candidates"], r["n_cuts"] = len(cand), len(cps)
        cps_ext, _ = extraction_windows(cps, T, min_window_for(p, 40))
        common = dict(X=X, detected_change_points=cps_ext, variable_names=names, window_overlap=0.0, verbose=False, prior_mode="cold",
                      hybrid=False, bootstrap_n=0, learner="directlingam", seed=0, transform="v2cap", standardize=True)
        t0 = time.time(); S = extract_causal_structures(stage0="auto", stage0_output="stage0", **common); r["t_extraction_stage0"] = time.time() - t0
        r["n_windows"] = len(S); r["n_stage0"] = sum(s.stage0_used for s in S); r["n_pc"] = sum(s.route == "pc" for s in S)
        t0 = time.time(); S0 = extract_causal_structures(stage0="off", **common); r["t_extraction_nostage0"] = time.time() - t0
        t0 = time.time(); chow_confirmed_cps(np.asarray(X, float), S, 0.001); r["t_confirmation"] = time.time() - t0
        bounds = [0] + list(cps_ext) + [T]; t0 = time.time()
        for a, b in zip(bounds[:-1], bounds[1:]):
            _fit_window_learner("dagma_paper", _standardize_window(pd.DataFrame(X[a:b], columns=names)), names, 0)
        r["t_dagma_paper"] = time.time() - t0
        r["t_total_method"] = r["t_detector_v1O"] + r["t_pelt"] + r["t_rank_validation"] + r["t_extraction_stage0"] + r["t_confirmation"]
        rows.append(r); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "timing.csv"), index=False)
pd.set_option("display.width", 220)
print(D.groupby(["p", "regime_length"])[[c for c in D if c.startswith("t_")]].mean().round(2).to_string())
