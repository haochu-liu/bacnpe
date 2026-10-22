# bacnpe

Simulation of bacterial sequences under the ClonalOrigin approximation of the
ancestral recombination graph, with a 30-dimensional vector of summary
statistics for neural posterior estimation (NPE) of local recombination and
mutation rates.

## Install

```bash
pip install .            # or: pip install -e ".[test]" for development
```

Requires Python ≥ 3.9, NumPy, SciPy and pandas.

## Usage

```python
import numpy as np
from bacnpe import ClonalTree, ClonalOrigin_seq_sim
from bacnpe.summary_stats import STAT_NAMES

np.random.seed(1)                      # simulations use NumPy's global random state
tree = ClonalTree(20)                  # clonal genealogy with 20 leaves
stats = ClonalOrigin_seq_sim(tree, rho_site=0.02, theta_site=0.05, L=200, delta=30.0)

dict(zip(STAT_NAMES, stats))           # 30 named summary statistics
```

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
├── __init__.py              public API: ClonalTree, ClonalOrigin_seq_sim
├── _validation.py           input checking shared by the entry points
├── tree/                    tree structures
│   ├── base.py              _Tree (base class)
│   └── clonal_genealogy.py  ClonalTree
├── simulation/              simulators
│   ├── arg.py               _ARG, recombination events, graph assembly
│   ├── mutation.py          _add_mutation
│   └── seq_sim.py           ClonalOrigin_seq_sim
└── summary_stats/           summary statistics
    ├── linkage.py           LD (D, D', r²), four-gamete test, Hudson's Rm, Kelly's ZnS
    ├── diversity.py         Watterson's θ, Tajima's π (+ Wakeley's π²), Tajima's D
    ├── homoplasy.py         homoplasy index (Fitch parsimony on the clonal tree)
    ├── wall.py              Wall's B and Q
    ├── regression.py        exponential decay fit of r² on distance
    └── segment_stats.py     the 30-entry vector, STAT_NAMES
```

Everything except `ClonalTree` and `ClonalOrigin_seq_sim` is an internal
helper (underscore prefix) and may change without notice.

## Tests

```bash
pytest
```

The suite checks output types, shapes and value ranges, the handling of
unpermitted inputs, the internal helpers on hand-made examples, and
"golden" vectors produced by the original implementation with fixed seeds.

See `CHANGES.md` for what changed relative to the original scripts.
