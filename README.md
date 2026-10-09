# bacnpe

`bacnpe` provides two public callables:

- `ClonalTree(n)` simulates a Kingman coalescent genealogy.
- `ClonalOrigin_seq_sim(tree, rho_site, theta_site, L, delta)` simulates a
  ClonalOrigin-style ancestral recombination graph and returns 46 summary
  statistics.

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
    rho_site=0.02,
    theta_site=0.03,
    L=1000,
    delta=200,
)
```
