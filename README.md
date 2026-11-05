# bacnpe

Simulation of bacterial sequences under the ClonalOrigin approximation of the
ancestral recombination graph, with a 30-dimensional vector of summary
statistics for neural posterior estimation (NPE) of local recombination and
mutation rates.

## Install

```bash
pip install .            # or: pip install -e ".[test]" for development
pip install ".[npe]"     # PyTorch + sbi, needed for bacnpe.npe
```

Requires Python ≥ 3.9, NumPy, SciPy, pandas and tqdm.

## Usage

```python
import numpy as np
from bacnpe import ARG, ClonalTree, ClonalOrigin_seq_sim, add_mutation, segment_summary_stats
from bacnpe.summary_stats import STAT_NAMES

np.random.seed(1)                      # simulations use NumPy's global random state
tree = ClonalTree(20)                  # clonal genealogy with 20 leaves
stats = ClonalOrigin_seq_sim(tree, rho_site=0.02, theta_site=0.05, L=200, delta=30.0)

dict(zip(STAT_NAMES, stats))           # 30 named summary statistics

# The two simulation steps can also be run separately:
arg = ARG(tree, rho_site=0.02, L=200, delta=30.0)   # ARG with recombination
node_site = add_mutation(arg, theta_site=0.05)       # bool, (n_nodes, L); first n rows are the leaves
stats = segment_summary_stats(tree, node_site[:tree.n])  # 30 statistics from the leaf sequences
```

### Simulating a training set in parallel

`parallel_sim` draws `rho_site`, `theta_site` and `L` from uniform
priors and runs `ClonalOrigin_seq_sim` for each draw in a pool of worker
processes. It writes `params.csv` (`n_sims` × 3) and `stats.csv`
(`n_sims` × 30) to `save_dir`, and row *i* of the two files belongs to the
same simulation.

```python
from bacnpe import ClonalTree, parallel_sim

if __name__ == "__main__":             # required on Windows/macOS when max_workers > 1
    tree = ClonalTree(27)
    params_path, stats_path = parallel_sim(
        tree,
        rho_range=(0, 0.2), theta_range=(0, 0.1), L_range=(400, 10000),  # L: both ends included
        delta=542.0, n_sims=10000, save_dir="sims/ecoli",
        chunk_size=100, max_workers=12, seed=1,
    )
```

- Each simulation has its own seed derived from `seed`, so the output
  depends only on `seed`, not on `max_workers` or `chunk_size`.
- `stats.csv` is written `chunk_size` rows at a time, so an interrupted run
  keeps its finished chunks.
- Existing files are not replaced unless `overwrite=True`.
- Both files start with `#` comment lines giving the column names and seed.
  `np.loadtxt(path, delimiter=",")` skips them.

### Training NPE

`bacnpe.npe` trains sbi's `NPE_C` on the output of `parallel_sim`.
It needs the `npe` extra. `import bacnpe` alone does not load torch.

```python
import numpy as np
import torch
from bacnpe.npe import LeaveLengthOut_NN, build_prior, train_npe, save_npe, load_npe

theta = np.loadtxt("sims/ecoli/params.csv", delimiter=",")   # (n_sims, 3)
x = np.loadtxt("sims/ecoli/stats.csv", delimiter=",")        # (n_sims, 30)

prior = build_prior(rho_range=(0, 0.2), theta_range=(0, 0.1), L_range=(400, 10000))
result = train_npe(theta, x, prior, seed=100)    # no embedding: flow conditioned on all 30 statistics

# Notebook setting: LeaveLengthOut_NN embedding (create it after seeding for reproducible weights)
torch.manual_seed(100)
embedding = LeaveLengthOut_NN(input_dim=30, num_hiddens=128, num_hidden_layers=1, num_outputs=2)
result = train_npe(theta, x, prior, embedding_net=embedding, seed=100)

# Training and validation losses
from sbi.analysis import plot_summary
plot_summary(result.inference, tags=["training_loss", "validation_loss"], figsize=(8, 4))

save_npe(result, "sims/ecoli/trained_npe_posterior.pt")

posterior = load_npe("sims/ecoli/trained_npe_posterior.pt")["posterior"]   # sbi DirectPosterior
```

- Use the same ranges in `build_prior` as in `parallel_sim`.
- `train_npe` drops simulations with NaN/inf statistics first. The number
  dropped is in `result.settings["n_dropped"]`.
- `prior` and `embedding_net` may be `None`. The flow and training settings
  default to the notebook values. All settings are stored in the checkpoint
  with the sbi, torch and bacnpe versions.
- `plot_summary` reads the TensorBoard logs that sbi writes to `sbi-logs/`
  in the working directory. The losses are also in
  `result.inference.summary["training_loss"]` and `["validation_loss"]`.
- `load_npe` also reads checkpoints saved by the old notebooks, without the
  `pysimARG` folder. It unpickles Python objects, so only load files you trust.

| Argument     | Type  | Allowed values |
|--------------|-------|----------------|
| `n` (tree)   | int   | ≥ 1 (≥ 3 for `ClonalOrigin_seq_sim`) |
| `rho_site`   | float | ≥ 0, finite |
| `theta_site` | float | ≥ 0, finite |
| `L`          | int   | ≥ 1 (≥ 3 for summary statistics) |
| `delta`      | float | ≥ 1, finite |

Wrong types raise `TypeError`; out-of-range values raise `ValueError`.
NumPy scalars (`np.int64`, `np.float32`, …) are accepted.

## Layout

```
bacnpe/
├── __init__.py              public API: ClonalTree, ARG, add_mutation, ClonalOrigin_seq_sim,
│                            segment_summary_stats, parallel_sim
├── parallel.py              parallel_sim: prior draws + parallel simulations to CSV
├── _validation.py           input checking shared by the entry points
├── npe/                     neural posterior estimation (needs torch + sbi)
│   ├── prior.py             build_prior
│   ├── distributions.py     DiscreteUniform (prior on L)
│   ├── embedding.py         LeaveLengthOut_NN (embedding network)
│   ├── train.py             train_npe, NPEResult, build_density_estimator
│   └── checkpoint.py        save_npe, load_npe
├── tree/                    tree structures
│   ├── base.py              _Tree (base class, class name "Tree")
│   └── clonal_genealogy.py  ClonalTree
├── simulation/              simulators
│   ├── arg.py               ARG, recombination events, graph assembly
│   ├── mutation.py          add_mutation
│   └── seq_sim.py           ClonalOrigin_seq_sim
└── summary_stats/           summary statistics
    ├── linkage.py           LD (D, D', r²), four-gamete test, Hudson's Rm, Kelly's ZnS
    ├── diversity.py         Watterson's θ, Tajima's π (+ Wakeley's π²), Tajima's D
    ├── homoplasy.py         homoplasy index (Fitch parsimony on the clonal tree)
    ├── wall.py              Wall's B and Q
    ├── regression.py        exponential decay fit of r² on distance
    └── segment_stats.py     segment_summary_stats (the 30-entry vector), STAT_NAMES
```

Everything except `ClonalTree`, `ARG`, `add_mutation`, `ClonalOrigin_seq_sim`,
`parallel_sim`, `segment_summary_stats`, `STAT_NAMES` and the names in
`bacnpe.npe.__all__` is an internal helper (underscore prefix) and may
change without notice.

## Tests

```bash
pytest
```

The suite checks output types, shapes and value ranges, the handling of
unpermitted inputs, the internal helpers on hand-made examples, and
"golden" vectors produced by the original implementation with fixed seeds.
`tests/test_npe.py` is skipped when torch or sbi is not installed.

See `CHANGES.md` for what changed relative to the original scripts.
