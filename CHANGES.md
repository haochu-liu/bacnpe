# Changes relative to the original scripts

**Outputs are unchanged.** With the same `np.random.seed`, the refactored
package reproduces the original trees, ARGs, mutated sequences and the
30-entry summary vector **bit for bit** (checked on 1,800 seeded runs covering
n ∈ {3, 4, 8, 15, 30}, ρ ∈ {0, 0.005, 0.05, 0.3}, θ ∈ {0, 0.01, 0.1},
L ∈ {1, 2, 7, 60, 250}, δ ∈ {1, 3, 40}, plus larger runs up to n = 100,
L = 2000). Typical runs are about 3–10× faster.

## Bug fixes

**Inputs that could only crash or give meaningless output are now
rejected up front with a clear message:**
   - `delta < 1`: `np.random.binomial(..., 1 - 1/delta)` gets a negative
     probability and crashed for any `L ≥ 2`. Now `ValueError` (δ is the mean
     length of a geometric segment, so δ ≥ 1).
   - `ClonalOrigin_seq_sim` with fewer than 3 leaves: the four-gamete test
     crashed for n = 2, and Watterson's θ / Tajima's D divided by zero.
     Now `ValueError`.
   - negative, NaN or infinite rates, `L < 1`: previously failed deep inside
     NumPy or produced NaN/inf. Now `ValueError`.

## Public API and naming

- Package name `bacnpe`; public API is only `ClonalTree` and
  `ClonalOrigin_seq_sim` (`from bacnpe import ...`).
- Everything else has an underscore: `_Tree` (was `tree`), `_ARG` (was
  `ARG`), `_add_mutation`, `_clonal_origin_nodes`, `_ld`, `_g4_test`,
  `_hudson_rm`, `_kelly_z`, `_watterson_theta`, `_tajima_pi`, `_tajima_d`,
  `_wall_bq`, `_homoplasy_index`, `_exp_regression`, `_shifted_exponential`,
  `_segment_summary_stats`.
- New: `bacnpe.summary_stats.STAT_NAMES`, the names of the 30 output entries.

## Input types

- All functions have type hints.
- Public entry points validate and normalise inputs: integers (`n`, `L`)
  accept Python/NumPy integers and are converted to `int`; rates accept any
  real number and are converted to `float`. `bool`, strings, arrays and
  floats-for-integers are rejected with `TypeError`; bad values with
  `ValueError`. **Note:** the original raised `ValueError` for a wrong type
  (e.g. `L=100.0`); it is now `TypeError`.
- `_segment_summary_stats` accepts a boolean or 0/1 matrix and converts it to
  `bool`; other values raise.
- `_ARG.node_mat` / `edge_mat` are now always `bool` (they were `float64`
  holding 0/1 when recombination occurred, `bool` otherwise).
  `ClonalTree.height/length` and `_ARG.height/length` are plain `float`.

## Combined / restructured code

- `ClonalOrigin_ARG.py` + `seq_simulator.py` + `ClonalOrigin_nodes.py` →
  `simulation/arg.py`. The simulator is split into
  `_simulate_recombination_events` (all random draws, same order as before)
  and `_build_arg_graph` (deterministic assembly).
- `LD.py` + `G4_test.py` + `Hudson_Rm.py` + `Kelly_Z.py` →
  `summary_stats/linkage.py`, built on one vectorised pairwise routine
  `_pairwise_ld_g4`. `_segment_summary_stats` reuses it instead of keeping
  its own copy of the Rm and ZnS logic.
- `Watterson_theta.py` + `Tajima_pi.py` + `Tajima_D.py` →
  `summary_stats/diversity.py`. `_watterson_theta` and `_tajima_d` now take
  the number of sequences `n` instead of the whole matrix (they only used its
  row count).
- `_ARG(tree, rho_site, L, delta, k=None)`: `k` now defaults to `L`
  (the sequence simulator always passed `k = L`).

## Performance improvements (same results)

- Pairwise LD / four-gamete statistics computed for all site pairs at once
  instead of two Python function calls per pair. Sums over pairs use a
  left-to-right accumulation (`np.cumsum`) so they match the original
  `+=` loop exactly.
- Tajima's π: pairwise Hamming distances by one matrix product instead of a
  double loop.
- Homoplasy index: Fitch parsimony for all sites at once (state sets as bit
  masks) instead of one dictionary pass per site; children of each node are
  looked up once instead of `np.where` per node per site.
- Wall's B/Q vectorised.
- ARG assembly: dictionary look-ups replace repeated linear `np.argmax`
  searches over all nodes (quadratic → linear).
- Mutations: kept/discarded in one indexing step; flipped sites are
  pre-computed per edge instead of filtering all mutations for every edge.

## Other clean-ups

- `segment_summary_stats` called `warnings.filterwarnings('ignore',
  category=RuntimeWarning)` at import time, silencing RuntimeWarnings in the
  user's whole program. Warnings are now suppressed only locally, where
  expected (Tajima's D with S = 0, the curve fit).
- Removed the dead `if not incompatible_intervals: s_vec[24] = 0` branch
  (overwritten right after) and other unused variables.
- Docstrings corrected: `ClonalOrigin_seq_sim` returns a **30**-dimensional
  vector (docstring said 46); `_add_mutation` processes edges **top-down**
  (root to leaves; the comment said bottom-up); the ARG simulates `k` sites
  (docstring said "two sites").
- Grouping of clonal edges by parent now keeps edge order explicitly
  instead of relying on `np.argsort`'s (unstable) default sort.

## Things deliberately **not** changed (worth considering)

- Tajima's D is `NaN` when there is no segregating site (0/0). An NPE
  pipeline may want to map this to 0.
- Randomness still comes from NumPy's global state (`np.random.seed`).
  Passing a `numpy.random.Generator` would make parallel simulation safer,
  but changes the random stream and therefore the outputs.
- `pandas` is kept only for the `groupby` mean in the r² decay fit, because
  its summation is what the original results depend on.
