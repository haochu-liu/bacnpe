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

- `_check_range` moved to `bacnpe/_validation.py`, so that `parallel_sim`
  and `bacnpe.npe.build_prior` share it.

## NPE (`bacnpe/npe/`)

Replaces `pysimARG/discrete_uniform.py`, `pysimARG/LeaveLengthOut_NN.py` and
these notebook sections in `local_recomb_mutate_bacillus.ipynb` and
`local_recomb_mutate_ClonalOrigin.ipynb`: "Create prior to pass range
knowledge to NPE", "Using all simulations" and "Save the trained NPE".
torch and sbi are required dependencies. So is `six`: `sbi.analysis` (for
example `plot_summary`) imports it in sbi 0.27, but sbi does not declare it.
The minimum Python version is now 3.10, because sbi requires it.

- **Prior:** `build_prior(rho_range, theta_range, L_range)` takes the same
  ranges as `parallel_sim`. The notebooks built a prior but never passed it
  to `NPE_C`, so sbi used an empirical prior instead (the saved checkpoints
  contain `ImproperEmpirical`). `train_npe` passes the prior. Training is
  unchanged, but posterior samples outside the prior are now rejected by
  default.
- **Optional prior and embedding:** `train_npe(theta, x, prior=None,
  embedding_net=None)`. With no embedding network, the flow is conditioned
  on the full z-scored `x`. To get the notebook setup, pass
  `LeaveLengthOut_NN(30, 128, 1, 2)`.
- **Seeds:** `train_npe(seed=...)` seeds torch and NumPy before building and
  training the flow. It restores the caller's random states afterwards.
  An embedding network passed in is initialised when you create it, so seed
  before creating it. The notebooks created it before `torch.manual_seed`,
  so its starting weights were not reproducible.
- **Invalid simulations:** `train_npe` drops rows with NaN/inf and records
  how many in `settings["n_dropped"]`. The ClonalOrigin notebook did this
  by hand. The bacillus notebook relied on sbi doing it silently.
- **Column drop:** the notebooks deleted columns 16–31 from older 46-column
  simulation files. `parallel_sim` writes the 30 columns directly, so there
  is nothing to delete.
- **`DiscreteUniform`:**
  - Bounds are checked: integer-valued, `low <= high`.
  - Sampling uses each element's own bounds. Before, batched bounds used the
    overall min and max.
  - `log_prob` works on any device.
  - Values between integers still count as inside the support, because NPE
    treats `L` as continuous.
- **`LeaveLengthOut_NN`:** behaviour and attribute names are unchanged; the
  docstring is corrected.
- **Checkpoints:**
  - `save_npe` stores the posterior, the training settings, and the sbi,
    torch and bacnpe versions. It does not overwrite an existing file unless
    `overwrite=True`. `checkpoint["posterior"]` works as before.
  - `load_npe` maps the old module names (`LeaveLengthOut_NN`,
    `discrete_uniform`) to `bacnpe.npe`. Old checkpoints therefore load
    without `pysimARG`. The bacillus checkpoint was checked to give
    identical samples.
  - `load_npe` warns when the sbi version differs.
