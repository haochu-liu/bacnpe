# Changes relative to the original scripts

## Class names

- `ClonalTree` and `ARG` now keep their own class names (`type(x).__name__`
  and `repr`). Before, `_Tree.__init__` overwrote `self.__class__.__name__`
  with `"Tree"` on every instance, which renamed the subclass itself: every
  `ARG` reported its class as `Tree`, and `ClonalTree` was renamed to `Tree`
  and back each time a tree was built.
- The base class `_Tree` is named `Tree`. This is set once when the class is
  defined, not on each instance. Its `__qualname__` is still `_Tree`, so
  pickling works.

## Public API

- `segment_summary_stats(tree, seg_mat)` is public. It can be imported from
  `bacnpe` and from `bacnpe.summary_stats`, and it is listed in both
  `__all__` lists.

## Parallel simulation (`bacnpe/parallel.py`)

`parallel_sim(clonal_tree, rho_range, theta_range, L_range, delta, n_sims,
save_dir, *, chunk_size=100, max_workers=None, seed=None,
overwrite=False, progress=True)` replaces `likelihood-free/python/example.py`.
Differences from that script:

- **Seeds:** each simulation gets its own seed, derived from `seed`. Before,
  every worker drew from NumPy's global random state. With "fork" (Linux),
  all workers started from the same state and reused the same random
  numbers. With "spawn" (Windows/macOS), a run could not be reproduced.
  Results now depend only on `seed`, and `max_workers=1` gives the same
  output as a parallel run.
- **Inputs:** `delta` is an argument (it was hard-coded to 542). The tree,
  prior ranges and output directory are arguments instead of fixed E. coli
  paths. All inputs are checked before any file is written.
- **L prior:** `L_range=(low, high)` includes both ends.
  `np.random.randint(400, 10000)` excluded 10000.
- **Output files:** `params.csv` (was `theta1.csv`) and `stats.csv` (was
  `x1.csv`) start with `#` lines giving the column names and seed.
  `np.loadtxt` skips these lines. `stats.csv` is no longer appended to an
  existing file, so a second run cannot silently double its rows. Existing
  files raise `FileExistsError` unless `overwrite=True`.
- **Progress bar:** `tqdm` is a required dependency. The bar is shown by
  default and turned off with `progress=False`.
