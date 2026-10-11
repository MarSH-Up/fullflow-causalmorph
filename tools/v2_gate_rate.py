# tools/v2_gate_rate.py
"""Share of non-root nodes whose residual the v2 gate maps (Jarque-Bera p >= alpha),
on the true regime windows of the first K sampled scenarios of each benchmark cell.
Run: cd Full-Flow && OMP_NUM_THREADS=1 python tools/v2_gate_rate.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "causalmorph"))
from run_experiments import load_scenario
from full_pipeline import _make_learner, _from_lingam, _to_causalmorph
from core.causalmorph_v2 import causalmorph_v2

CELLS = {"lin_laplace": "datasets", "nonlin_laplace": "datasets_v2_nonlin_laplace",
         "lin_normal": "datasets_v2_lin_normal", "nonlin_normal": "datasets_v2_nonlin_normal"}
K = int(os.environ.get("K", "20"))
ids = [l.strip() for l in open(os.path.join(ROOT, "results/hardening_validation/sample_ids.txt")) if l.strip()][:K]

print(f"{'cell':<16}{'nodes':>7}{'gated':>7}{'rate':>7}")
for cell, ds in CELLS.items():
    index = pd.read_csv(os.path.join(ROOT, ds, "index.csv"), dtype=str).set_index("scenario_id")
    nodes = gated = 0
    for sid in ids:
        X, names, cps, adjs, sizes, _ = load_scenario(sid, os.path.join(ROOT, ds), index.loc[sid])
        bounds = [0] + list(cps) + [len(X)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            W = pd.DataFrame(X[a:b], columns=names)
            m = _make_learner("directlingam", 0).fit(W)
            est = pd.DataFrame((np.abs(_from_lingam(m.adjacency_matrix_)) > 0.05).astype(float), index=names, columns=names)
            _, det = causalmorph_v2(W, list(m.causal_order_), _to_causalmorph(est), return_details=True)
            for rec in det.values():
                if rec["skipped"] is None:
                    nodes += 1; gated += int(rec["gated"])
    print(f"{cell:<16}{nodes:>7}{gated:>7}{(gated / nodes if nodes else float('nan')):>7.2f}")
