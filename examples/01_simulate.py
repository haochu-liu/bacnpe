"""
Example 1: simulate gene alignments under the ClonalOrigin model.

1. Simulate a clonal genealogy of 12 genomes (coalescent).
2. Simulate one gene of 1,000 bp on it and print its 30 summary statistics.
3. Simulate a small training set of 200 genes in parallel.

Run from anywhere:

    python examples/01_simulate.py

Outputs are written to examples/output/simulations/ (a few seconds with 4 workers).
"""
from pathlib import Path

import numpy as np

from bacnpe import ClonalOrigin_seq_sim, ClonalTree, parallel_sim
from bacnpe.summary_stats import STAT_NAMES

OUTPUT = Path(__file__).resolve().parent / "output" / "simulations"


def main():
    # 1. Clonal genealogy: simulations use NumPy's global random state
    np.random.seed(1)
    tree = ClonalTree(12)
    print(f"{tree}: height {tree.height:.3f}, total branch length {tree.length:.3f}")

    # 2. One gene: rho_site/2 and theta_site/2 are the rates per site and unit of
    #    branch length, delta is the mean recombination tract length (bp)
    z = ClonalOrigin_seq_sim(tree, rho_site=0.02, theta_site=0.05, L=1000, delta=300.0)
    print("\nSummary statistics of one simulated gene:")
    for name, value in zip(STAT_NAMES, z):
        print(f"  {name:<24s} {value:.6g}")

    # 3. Training set: (rho_site, theta_site, L) drawn from uniform priors, one
    #    simulation per draw, run by 4 worker processes in chunks of 50.
    #    The paper uses L up to 10,000 bp; 2,000 keeps this example quick.
    params_path, stats_path = parallel_sim(
        tree,
        rho_range=(0.0, 0.1),
        theta_range=(0.0, 0.1),
        L_range=(100, 2000),          # both ends included
        delta=300.0,
        n_sims=200,
        save_dir=OUTPUT,
        chunk_size=50,
        max_workers=4,
        seed=1,
        overwrite=True,
    )

    theta = np.loadtxt(params_path, delimiter=",")   # (200, 3): rho_site, theta_site, L
    x = np.loadtxt(stats_path, delimiter=",")        # (200, 30): summary statistics
    print(f"\nSaved {params_path.name} {theta.shape} and {stats_path.name} {x.shape} in {OUTPUT}")


# Needed for parallel_sim: worker processes re-import this file on Windows and macOS
if __name__ == "__main__":
    main()
