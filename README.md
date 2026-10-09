# bacnpe

`bacnpe` provides two public callables:

- `ClonalTree(n)` simulates a Kingman coalescent genealogy.
- `ClonalOrigin_seq_sim(tree, rho_site, theta_site, L, delta)` simulates a
  ClonalOrigin-style ancestral recombination graph and returns 46 summary
  statistics.

All implementation helpers have leading underscores and are intentionally not
part of the public API.

## Installation

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
pytest
```

## Example

```python
import numpy as np
from bacnpe import ClonalTree, ClonalOrigin_seq_sim

np.random.seed(123)
tree = ClonalTree(20)
statistics = ClonalOrigin_seq_sim(
    tree,
    rho_site=0.001,
    theta_site=0.01,
    L=100,
    delta=20,
)
assert statistics.shape == (46,)
```

The implementation uses NumPy's global random state, so calling
`numpy.random.seed(...)` before simulation makes a run reproducible.
