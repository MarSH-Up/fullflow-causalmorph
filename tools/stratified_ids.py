#!/usr/bin/env python3
"""
Write a stratified sample of scenario ids: equal share per (p, n_regimes,
samples_regime) stratum as far as the strata allow, deterministic (seed),
optionally excluding ids (e.g. the tuning sample) and restricted to a subset.

    python tools/stratified_ids.py --index datasets/index.csv --n 600 \
        --exclude results/hardening_validation/sample_ids.txt --out results/full_v2/kci_ids.txt
"""
import argparse

import numpy as np
import pandas as pd


def stratified(index: pd.DataFrame, n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    groups = {k: list(g.scenario_id) for k, g in index.groupby(["p", "n_regimes", "samples_regime"])}
    for k in groups:
        rng.shuffle(groups[k])
    keys = sorted(groups)
    picked, round_ = [], 0
    while len(picked) < n and any(len(groups[k]) > round_ for k in keys):
        order = list(keys); rng.shuffle(order)
        for k in order:
            if len(picked) >= n:
                break
            if len(groups[k]) > round_:
                picked.append(groups[k][round_])
        round_ += 1
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--exclude")
    ap.add_argument("--within")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    index = pd.read_csv(a.index, dtype=str)
    if a.exclude:
        try:
            ex = {l.strip() for l in open(a.exclude) if l.strip()}
            index = index[~index.scenario_id.isin(ex)]
        except FileNotFoundError:
            pass
    if a.within:
        keep = {l.strip() for l in open(a.within) if l.strip()}
        index = index[index.scenario_id.isin(keep)]
    ids = stratified(index, a.n, a.seed)
    with open(a.out, "w") as f:
        f.write("\n".join(ids) + ("\n" if ids else ""))
    print(f"{len(ids)} stratified ids -> {a.out}")


if __name__ == "__main__":
    main()
