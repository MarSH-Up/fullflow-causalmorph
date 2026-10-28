"""
Reproduce Table 1 of Bello, Aragam and Ravikumar (NeurIPS 2022, Appendix C.1.1)
with the `dagma` package, to check the DAGMA baseline configuration.

Stationary linear SEMs, ER4 and SF4, Gaussian/exponential/Gumbel noise, n = 1000,
10 repetitions, d in {20, 30, 50}; SHD as in the paper (dagma.utils.count_accuracy:
extra + missing + reversed, a reversal counted once). Two configurations:

  paper     raw data, the hyperparameters of Appendix C.1.1
            (lambda1 0.05, T 4, s {1,.9,.8,.7}, 2e4/7e4 iterations, threshold 0.3)
  pipeline  the per-window baseline of full_pipeline.py (centred, divided by the
            mean column sd, package defaults with lambda1 0.03, threshold 0.3)

  python tools/reproduce_dagma_table1.py   -> results/dagma_benchmark/reproduction_table1.csv
"""
import contextlib
import io
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from dagma import utils as du
from dagma.linear import DagmaLinear

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "dagma_benchmark", "reproduction_table1.csv")
# Table 1 of the paper: mean SHD averaged over ER4/SF4 and the three noises (95% CI, 10 repetitions)
REPORTED = {20: (6.78, 1.64), 30: (11.05, 2.50), 50: (12.03, 3.42)}


def fit(X, config):
    X = X - X.mean(axis=0)
    with contextlib.redirect_stderr(io.StringIO()):
        if config == "paper":
            return DagmaLinear(loss_type="l2").fit(X, lambda1=0.05, w_threshold=0.3, T=4, mu_init=1.0,
                                                   mu_factor=0.1, s=[1.0, 0.9, 0.8, 0.7],
                                                   warm_iter=20000, max_iter=70000, lr=3e-4)
        X = X / X.std(axis=0).mean()
        return DagmaLinear(loss_type="l2").fit(X, lambda1=0.03, w_threshold=0.3)


def one(indexed_job):
    seed, (d, graph, sem, rep) = indexed_job  # one seed per job
    du.set_random_seed(seed)
    B = du.simulate_dag(d, 4 * d, graph)
    X = du.simulate_linear_sem(du.simulate_parameter(B), 1000, sem)
    row = {"d": d, "graph": f"{graph}4", "sem": sem, "rep": rep}
    for config in ("paper", "pipeline"):
        W = fit(X, config)
        row[f"shd_{config}"] = du.count_accuracy(B, (W != 0).astype(int))["shd"]
    return row


def main():
    jobs = list(itertools.product(REPORTED, ("ER", "SF"), ("gauss", "exp", "gumbel"), range(10)))
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 12))) as ex:
        rows = list(ex.map(one, enumerate(jobs)))
    df = pd.DataFrame(rows)
    df["shd_reported"] = df.d.map(lambda d: REPORTED[d][0])
    df["ci_reported"] = df.d.map(lambda d: REPORTED[d][1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.groupby("d")[["shd_reported", "shd_paper", "shd_pipeline"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
