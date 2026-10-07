# tools/kbs_replication.py
"""
Stationary replication of the KBS experiment (corrected convention) with
four preconditioning arms: none, v1 (published, seeded), lite, v2 — each
with ICA-LiNGAM and DirectLiNGAM, with the TRUE graph as prior and with the
learner's own estimate as prior. Also sweeps the v2 gate threshold
(gate_alpha) with the true prior.

Run: cd Full-Flow && W=4 OMP_NUM_THREADS=1 python tools/kbs_replication.py
"""
import itertools, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "causalmorph"))
sys.path.insert(0, "/Users/mdlsh/Desktop/Research/PhD/Code/IYCC/IYCC/src")
from components.scenarios.simple_scenario import causal_graph_simple_scenario_v25
from full_pipeline import _apply_transform, _from_lingam, _to_causalmorph, compute_shd, _make_learner
from core.causalmorph_v2 import causalmorph_v2
from concurrent.futures import ProcessPoolExecutor, as_completed

ARMS = ("none", "v1", "lite", "v2")
ALPHAS = (0.001, 0.01, 0.05)


def binar(B, thr=0.05):
    return (np.abs(np.asarray(B)) > thr).astype(float)


def one(cfg):
    A, X = causal_graph_simple_scenario_v25(p=cfg["p"], pconn=0.3, dist=[cfg["dist"]] * cfg["p"], deviation=0.5,
            signal_strength=2.0, nsamples=cfg["n"], seed=cfg["seed"], mode=cfg["mode"], nonlin_func=cfg["func"],
            nonlinearity=cfg["nl"], max_attempts=20, min_std=1e-6, min_edges=1)
    X = X.astype(float); A = A.astype(float)
    out = dict(cfg)
    for alg in ("icalingam", "directlingam"):
        m0 = _make_learner(alg, cfg["seed"]).fit(X)
        est0 = pd.DataFrame(binar(_from_lingam(m0.adjacency_matrix_)), index=A.index, columns=A.columns)
        order0 = list(m0.causal_order_)
        for prior_name, prior_adj in (("true", A), ("est", est0)):
            for arm in ARMS:
                try:
                    T = _apply_transform(arm, X, order0, prior_adj, cfg["seed"])
                    m = _make_learner(alg, cfg["seed"]).fit(T)
                    est = pd.DataFrame(binar(_from_lingam(m.adjacency_matrix_)), index=A.index, columns=A.columns)
                    out[f"{alg}_{prior_name}_{arm}"] = compute_shd(est, A)["F1"]
                except Exception as exc:  # noqa: BLE001
                    out[f"{alg}_{prior_name}_{arm}"] = float("nan"); out["error"] = repr(exc)[:120]
        for alpha in ALPHAS:
            try:
                T = causalmorph_v2(X, order0, _to_causalmorph(A), gate_alpha=alpha)
                m = _make_learner(alg, cfg["seed"]).fit(T)
                est = pd.DataFrame(binar(_from_lingam(m.adjacency_matrix_)), index=A.index, columns=A.columns)
                out[f"{alg}_true_v2_a{alpha}"] = compute_shd(est, A)["F1"]
            except Exception as exc:  # noqa: BLE001
                out[f"{alg}_true_v2_a{alpha}"] = float("nan"); out["error"] = repr(exc)[:120]
    return out


def make_cfgs():
    cfgs = []
    for p, n, r in itertools.product([5, 10, 20], [500, 1000, 2000], range(5)):
        cfgs.append(dict(p=p, n=n, seed=1000 * p + n + r, mode="linear", func="tanh", nl=0.0, dist="laplace"))
    for p, func, r in itertools.product([5, 10], ["tanh", "square"], range(3)):
        cfgs.append(dict(p=p, n=1000, seed=7000 + 100 * p + r, mode="nonlinear", func=func, nl=0.5, dist="laplace"))
    for p, n, r in itertools.product([5, 10], [1000, 2000], range(3)):
        cfgs.append(dict(p=p, n=n, seed=9000 + 100 * p + n + r, mode="linear", func="tanh", nl=0.0, dist="normal"))
    return cfgs


def main():
    cfgs = make_cfgs()
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("W", "4"))) as ex:
        futs = [ex.submit(one, c) for c in cfgs]
        for f in as_completed(futs):
            rows.append(f.result())
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results", "kbs_replication"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "kbs_replication", "kbs_replication_arms.csv"), index=False)
    for (mode, dist), d in df.groupby(["mode", "dist"]):
        print(f"\n{mode} / {dist}   n runs={len(d)}   mean F1")
        print(f"  {'':24}" + "".join(f"{a:>8}" for a in ARMS))
        for alg in ("icalingam", "directlingam"):
            for prior in ("true", "est"):
                print(f"  {alg + ' / prior ' + prior:<24}" + "".join(f"{d[f'{alg}_{prior}_{a}'].mean():>8.3f}" for a in ARMS))
        print(f"  gate sweep (v2, true prior)  mean F1")
        print(f"  {'':24}" + "".join(f"{'a=' + str(a):>8}" for a in ALPHAS))
        for alg in ("icalingam", "directlingam"):
            print(f"  {alg:<24}" + "".join(f"{d[f'{alg}_true_v2_a{a}'].mean():>8.3f}" for a in ALPHAS))
    if "error" in df.columns:
        print("\nerrors:", df["error"].dropna().value_counts().to_dict())
    else:
        print("\nerrors: none")


if __name__ == "__main__":
    main()
