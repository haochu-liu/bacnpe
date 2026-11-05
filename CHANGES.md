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

## Estimated clonal trees (`bacnpe/tree/`)

- **Loading without simulating:** `ClonalTree.from_newick`, `from_csv` and
  `from_arrays` build a tree directly. Before, `example.py` simulated a
  `ClonalTree(n=27)` and then overwrote `edge`, `node_height`, `height` and
  `length`. The arrays are checked: binary tree, leaves at height 0,
  internal nodes numbered by height, edge lengths equal to height
  differences. Edges are sorted by parent. Simulations on a loaded tree
  are identical to those on the overwritten tree.
- **No R needed:** `from_newick` ports `R/simARG/*_clonal_tree.R`:
  - `ape::read.tree`;
  - `ape::cophenetic.phylo`;
  - `phangorn::upgma`, which is `hclust(method = "average")` +
    `as.phylo`;
  - the rescaling, renumbering and sorting.

  Arithmetic and tie-breaking follow R. On the klebsiella, ecoli and staph
  ClonalFrameML trees and the SimBac tree, topology and node numbering are
  identical to the R CSV files, and lengths agree to within 5e-15.
- **Root scaling:** the R scripts rescale the standardized (UPGMA) tree t2
  by `e = dist.nodes(t2)[Ntip(t2), 1] / 2`. On t2 this is half the
  distance between tips 1 and n, i.e. the height of their common ancestor
  in t2. That ancestor is the root of t2 only when the two tips lie on
  opposite sides of it. The R CSV files confirm this: the node with height
  `2(1 - 1/n)` is always the common ancestor of tips 1 and n.
  - ecoli and SimBac: it is the root.
  - klebsiella: it is not, so the root height is 2.246 instead of 1.98.
  - staph: it is not, so the root height is 4.313 instead of 1.982.

  `from_newick` scales the root of t2 to `2(1 - 1/n)`, which the R comment
  `#current root height` suggests was intended. `r_compatible=True`
  reproduces the R trees.
- **`root_height`** can be set to another height, or to `None` to keep the
  UPGMA heights.
- **Tip names:** stored on the tree as `tip_labels` (None for simulated
  trees), and included in `to_dict()`.
- **Saving:** `to_csv(directory)` writes the files of the R scripts
  (`clonal_edge.csv`, `clonal_node_height.csv`, `tip_names.csv`) at full
  precision. `to_newick()` returns a Newick string.

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
- **Posterior samples for observed genes:**
  `sample_posteriors(posterior, x_obs, num_samples=5000)` replaces the
  per-gene `posterior.sample` loop of the `local_recomb_mutate_*`
  notebooks.
  - It returns `(n_genes, num_samples, 3)` floats, ready for
    `detect_hotspots`.
  - Rows with NaN/inf get NaN samples and a warning. Before, sbi raised an
    error on them.
  - `seed` gives reproducible draws without changing torch's global random
    state.
  - `reject_outside_prior` defaults to True, as in Algorithm 2 of the
    paper. The bacillus and staph notebooks used False, which gave the
    non-positive draws that `detect_hotspots` clips. The ecoli notebook
    used True.

## Hotspot detection (`bacnpe/detection/`)

Replaces `pysimARG/hotspot_DoG.py` and the "Plot bar plots by position"
section of `hotspots_bacillus.ipynb`, `hotspots_ecoli.ipynb`,
`hotspots_klebsiella.ipynb` and `hotspots_staph.ipynb`.

- **Same results:** the detector's numerical code is unchanged. On synthetic
  genomes, `detect_hotspots` gives bit-for-bit the same per-gene arrays,
  segments and hotspot table as the notebook cells using `hotspot_DoG.py`.
  The cases checked include unsorted genes, negative posterior samples,
  subsampled draws and a hotspot that wraps around the origin.
- **`detect_hotspots(gene_starts, posterior_samples, chromosome_length_bp,
  gene_ends=...)`** replaces each notebook cell. The notebooks sorted genes
  first, transposed the samples, clipped values `<= 0` to machine epsilon and
  built the detector. Genes no longer need to be sorted, and the input array
  is not modified. Defaults are the notebook values (10 kb / 1 Mb, threshold
  0.9, `random_state=10`, `max_gap_bp=800_000`).
- **`max_gap_bp`** can be set once on the detector. `segments`,
  `to_dataframe` and `plot` use it unless told otherwise. Before, it had to
  be repeated in every call (four times per notebook cell).
- **`result.to_dataframe()`** replaces the per-row `pd.DataFrame` filling.
  - Columns have proper dtypes instead of `object`.
  - Two columns are added: `median_probability` and `wraps_origin`.
  - `end_position_bp` is the gene end of the last gene, as in the notebooks.
    It is looked up through `sort_order`, so it is correct whatever the input
    order.
  - In `hotspots_bacillus.ipynb` the mutation table was created but never
    filled, so its Excel sheet would have been empty.
- **`result.summary()`** returns the segment report the notebooks printed.
- **`result.segment_gene_indices()`** gives the input-order indices of the
  genes in each hotspot, replacing the manual `gene_info` slicing in
  `hotspots_staph.ipynb`.
- **`save_hotspot_tables`** replaces the commented-out `pd.ExcelWriter`
  block. It writes `.xlsx` (one sheet per table) or a folder of CSV files,
  and does not overwrite existing files unless `overwrite=True`.
- **Plot:** `result.plot(...)` / `plot_dog_hotspots(result, ...)` draw the
  same figure as before.
- **Input checks:** inputs are checked for type as well as value. For
  example, a string bandwidth raises `TypeError`.
- **Dependencies:** `matplotlib` and `openpyxl` (for Excel) are now required.
