# bacnpe

**bacnpe** infers gene-specific recombination and mutation rates from an
alignment of bacterial genomes, and detects regions where these rates are
locally elevated (hotspots). It accompanies the paper

> Liu H., Everitt R. G., Didelot X. *Detection of local recombination and
> mutation hotspots in bacterial genomes using neural posterior estimation.*

The method has three steps:

1. **Simulate** gene alignments under the ClonalOrigin model, conditional
   on a fixed clonal genealogy, and summarise each one by 29 summary
   statistics plus its length.
2. **Train** a neural posterior estimator (NPE-C with a neural spline flow
   and a length-preserving compression network) on these simulations. One
   trained model then gives the posterior of each observed gene.
3. **Detect hotspots** with the circular posterior-propagated
   Difference-of-Gaussians (ppDoG) detector, which contrasts a local and a
   regional smoother of the posterior log rates along the chromosome.

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Workflow](#workflow)
   1. [Clonal genealogy](#1-clonal-genealogy)
   2. [ClonalOrigin simulation](#2-clonalorigin-simulation)
   3. [Summary statistics](#3-summary-statistics)
   4. [Simulating a training set](#4-simulating-a-training-set)
   5. [Training NPE](#5-training-npe)
   6. [Posteriors for observed genes](#6-posteriors-for-observed-genes)
   7. [Hotspot detection (ppDoG)](#7-hotspot-detection-ppdog)
4. [Paper sections and code](#paper-sections-and-code)
5. [Package layout](#package-layout)
6. [Public API and input checks](#public-api-and-input-checks)
7. [Tests](#tests)
8. [Citation and license](#citation-and-license)

## Installation

```bash
pip install .            # or: pip install -e ".[test]" for development
```

Requires Python ≥ 3.10, NumPy, SciPy, pandas, tqdm, PyTorch, sbi,
matplotlib and openpyxl. It also needs `six`, which `sbi.analysis` imports
without declaring it. R is not needed.

## Quick start

Simulate one gene of 2,000 bp on a simulated genealogy of 30 genomes, and
compute its summary statistics:

```python
import numpy as np
from bacnpe import ClonalTree, ClonalOrigin_seq_sim
from bacnpe.summary_stats import STAT_NAMES

np.random.seed(1)                      # simulations use NumPy's global random state
tree = ClonalTree(30)                  # clonal genealogy simulated by the coalescent
z = ClonalOrigin_seq_sim(tree, rho_site=0.02, theta_site=0.05, L=2000, delta=300.0)
dict(zip(STAT_NAMES, z))               # 29 summary statistics + the length L
```

## Workflow

The steps below follow the analyses of the paper. The comments show the
B. cereus, S. aureus and E. coli settings where they differ.

### 1. Clonal genealogy

The ClonalOrigin model is conditional on a fixed clonal genealogy 𝒯
(paper, Section 2.1). It is either simulated, as in the simulation study
(Section 3.1), or estimated from the data, e.g. by ClonalFrame or
ClonalFrameML (Sections 3.2–3.4).

```python
from bacnpe import ClonalTree

tree = ClonalTree(30)                                      # simulated (coalescent)
tree = ClonalTree.from_newick("data/staph/cfml.newick")    # estimated phylogeny
tree.tip_labels[:3]                                        # genome names of leaves 1..n
```

`from_newick` turns an estimated phylogeny into a clonal genealogy in
coalescent units:

- it makes the tree ultrametric by UPGMA on its patristic distances;
- it rescales the root to the coalescent expectation `2(1 - 1/n)`.

This is a Python port of the R scripts (`ape` + `phangorn`) used before.
Further options:

- `root_height=` sets another root height, or `None` keeps the UPGMA
  heights (half the patristic distances).
- `r_compatible=True` reproduces the R output exactly. The R scripts scale
  the common ancestor of the first and last tip to `2(1 - 1/n)`. That node
  is not always the root: for the S. aureus tree, the R root is 2.18×
  taller.

A genealogy can be saved and reloaded:

```python
tree.to_csv("data/staph")              # clonal_edge.csv, clonal_node_height.csv, tip_names.csv
tree = ClonalTree.from_csv("data/staph")
tree = ClonalTree.from_arrays(edge, node_height, tip_labels=names)
tree.to_newick()
```

Leaves are numbered `1..n` in the order of `tip_labels`. Observed
alignments must use this row order (see [step 6](#6-posteriors-for-observed-genes)).

### 2. ClonalOrigin simulation

`ClonalOrigin_seq_sim(tree, rho_site, theta_site, L, delta)` runs
Algorithm 1 of Supplementary S1 and returns the 30-entry vector
z = (s₁, …, s₂₉, ℓ):

- Recombination tracts have geometric lengths with mean `delta`. The
  ClonalOrigin process is simulated site by site (Equations 1–6), and gives
  the local genealogy at every site.
- Mutations are added on the local genealogies at rate `theta_site / 2`
  per site and unit of branch length (Equation 7).

The two steps can also be run separately:

```python
from bacnpe import ARG, add_mutation, segment_summary_stats

arg = ARG(tree, rho_site=0.02, L=2000, delta=300.0)      # recombination events + local genealogies
node_site = add_mutation(arg, theta_site=0.05)           # bool (n_nodes, L); first n rows are the leaves
z = segment_summary_stats(tree, node_site[:tree.n])      # the same 30-entry vector
```

`rho_site` and `theta_site` are ρ_s and θ_s of the paper: ρ_s/2 and θ_s/2
are the rates per site and unit of branch length.

### 3. Summary statistics

`segment_summary_stats(tree, seg_mat)` computes the statistics of
Supplementary S2 for a binary alignment `seg_mat` of shape `(n, ℓ)`. The
names are in `bacnpe.summary_stats.STAT_NAMES`:

| Index | Statistics | Paper |
|---|---|---|
| 0–7 | Exact LD and G4 features: sums of \|D\|, D′, r² and four-gamete failures over near (d < ℓ/2) and far pairs of segregating sites | S10–S14 |
| 8–15 | Proportional LD and G4 features: the same quantities averaged over pairs | S14 |
| 16 | Homoplasy index (Fitch parsimony on the clonal genealogy) | S8 |
| 17 | Proportion of segregating sites | S7 |
| 18, 19, 20 | Watterson's θ, Tajima's π, Wakeley's π | S1, S2, S9 |
| 21 | Tajima's D | S3–S6 |
| 22, 23 | Wall's B and Q | S16, S17 |
| 24 | Hudson's R_M | — |
| 25 | Kelly's Z_nS | S15 |
| 26–28 | a, b, c of the fit r² = a·exp(−b·d) + c | S18 |
| 29 | Length ℓ | Eq. 9 |

### 4. Simulating a training set

`parallel_sim` draws ρ_s, θ_s and ℓ independently from uniform priors.
ℓ is discrete uniform, with both ends included. It then runs the simulator
for each draw in a pool of worker processes. It writes two files to
`save_dir`, and row *i* of both belongs to the same simulation:

- `params.csv`, `n_sims` × 3;
- `stats.csv`, `n_sims` × 30.

```python
from bacnpe import ClonalTree, parallel_sim

if __name__ == "__main__":             # required on Windows/macOS when max_workers > 1
    tree = ClonalTree.from_csv("data/ecoli")
    parallel_sim(
        tree,
        rho_range=(0, 0.2), theta_range=(0, 0.1), L_range=(400, 10000),   # E. coli priors
        delta=542.0, n_sims=50000, save_dir="sims/ecoli",
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

### 5. Training NPE

`bacnpe.npe` trains sbi's `NPE_C` (Section 2.2, Algorithm 2 of S3) with a
neural spline flow. The compression network `LeaveLengthOut_NN`
(Section 2.3) passes the whole input z through an MLP, then appends ℓ to
its output (Equation 10). `import bacnpe` alone does not load torch; it is
loaded by `import bacnpe.npe`.

The configuration selected in Supplementary S8 (Table S3) is:

- compression network: 1 hidden layer of 128 units, 2 + 1 outputs;
- NSF: 8 transforms, 30 hidden features, 5 bins;
- training: learning rate 10⁻⁴, batch size 100, patience 20, no clipping.

The flow and training values are the defaults of `train_npe`.

```python
import numpy as np
import torch
from bacnpe.npe import LeaveLengthOut_NN, build_prior, train_npe, save_npe, load_npe

theta = np.loadtxt("sims/ecoli/params.csv", delimiter=",")   # (n_sims, 3): rho, theta, L
x = np.loadtxt("sims/ecoli/stats.csv", delimiter=",")        # (n_sims, 30)

prior = build_prior(rho_range=(0, 0.2), theta_range=(0, 0.1), L_range=(400, 10000))
torch.manual_seed(100)                 # reproducible embedding weights
embedding = LeaveLengthOut_NN(input_dim=30, num_hiddens=128, num_hidden_layers=1, num_outputs=2)
result = train_npe(theta, x, prior, embedding_net=embedding, seed=100)

save_npe(result, "sims/ecoli/trained_npe_posterior.pt")
posterior = load_npe("sims/ecoli/trained_npe_posterior.pt")["posterior"]   # sbi DirectPosterior
```

- Use the same ranges in `build_prior` as in `parallel_sim`. The posterior
  then discards draws outside the prior support (Algorithm 2, line 12).
- `embedding_net=None` trains the flow directly on all 30 entries. This is
  the reference model of S8 and the setting of the benchmark in S7.
- `train_npe` drops simulations with NaN/inf statistics first. The number
  dropped is in `result.settings["n_dropped"]`.
- All settings are stored in the checkpoint, with the sbi, torch and
  bacnpe versions.
- `load_npe` also reads checkpoints saved by the old notebooks. It
  unpickles Python objects, so only load files you trust.
- The training and validation losses are in
  `result.inference.summary["training_loss"]` and `["validation_loss"]`.
  `sbi.analysis.plot_summary(result.inference, tags=["training_loss",
  "validation_loss"])` plots them from the TensorBoard logs in `sbi-logs/`.

### 6. Posteriors for observed genes

An observed gene is summarised exactly like a simulation. Its alignment
must be a binary matrix whose rows follow `tree.tip_labels`.
`sample_posteriors` then draws M posterior samples per gene (Algorithm 2,
lines 10–13):

```python
import numpy as np
from bacnpe import segment_summary_stats
from bacnpe.npe import load_npe, sample_posteriors

# genomes_bool: (n, genome length) binary alignment, rows ordered as tree.tip_labels
x_obs = np.stack([segment_summary_stats(tree, genomes_bool[:, start - 1:end])
                  for start, end in zip(gene_starts, gene_ends)])          # (n_genes, 30)

posterior = load_npe("sims/ecoli/trained_npe_posterior.pt")
theta_post = sample_posteriors(posterior, x_obs, num_samples=5000, seed=1)  # (n_genes, 5000, 3)
```

- **Prior support:** draws outside the prior support are discarded and
  redrawn, as in Algorithm 2. This needs a posterior trained with a prior
  from `build_prior`. `reject_outside_prior=False` keeps them, as the
  B. cereus and S. aureus notebooks did, and can give non-positive rates.
- **Invalid genes:** rows of `x_obs` with NaN or inf get NaN samples, with
  a warning.
- **Genes without segregating sites** (`prop_segregating_sites == 0`)
  carry little information about the rates. The notebooks removed them
  before sampling.
- **`posterior`** may be the sbi posterior, the `NPEResult` from
  `train_npe`, or the output of `load_npe`.
- **`seed`** makes the draws reproducible without changing torch's global
  random state.
- **Length check:** the inferred ℓ (`theta_post[:, :, 2]`) should match
  the observed lengths, as in Figure 1C and Figures S5–S7.

### 7. Hotspot detection (ppDoG)

`bacnpe.detection` implements the ppDoG detector (Section 2.5,
Supplementary S6), separately for ρ_s and θ_s. For every posterior draw:

1. Take the log rates, Λ = log λ.
2. Smooth them along the circular chromosome with a narrow (h_N) and a
   broad (h_B) Gaussian kernel. The weights use circular distances and the
   precision 1/(v_j + τ²) of each gene's log-rate posterior.
3. Compute the score D = N − B. exp(D) is the local-to-background fold
   enrichment.

A gene is a candidate when P(D > log F_min) ≥ 0.9. Runs of neighbouring
candidates, including runs across the origin, are merged into candidate
hotspot regions.

```python
from bacnpe.detection import detect_hotspots, save_hotspot_tables

common = dict(chromosome_length_bp=2_902_619, gene_ends=gene_ends,
              narrow_bandwidth_bp=10_000, broad_bandwidth_bp=1_000_000)
recomb = detect_hotspots(gene_starts, theta_post[:, :, 0], min_fold_enrichment=1.2, **common)
mutate = detect_hotspots(gene_starts, theta_post[:, :, 1], min_fold_enrichment=40, **common)

print(recomb.summary())                                   # candidate regions
recomb.plot(rate_label="Recombination rate",
            title_text="S. aureus Recombination Rate with Detected Candidate Hotspots",
            save_path="staph_recomb_ppdog.pdf")           # three-panel figure (Figures 4–9)
gene_info.iloc[recomb.segment_gene_indices()[0]]          # genes in the first region

save_hotspot_tables({"Recombination": recomb.to_dataframe(),     # Tables S4–S6
                     "Mutation": mutate.to_dataframe()}, "staph_hotspots.xlsx")
```

These are the settings of the paper (Table S1). In all cases τ = 0.2
(`residual_log_sd`, the default) and the probability threshold is 0.9.

| Dataset | Rate | h_N (bp) | h_B (bp) | F_min |
|---|---|---|---|---|
| B. cereus | recombination, mutation | 10,000 | 1,000,000 | 1.5 |
| S. aureus | recombination | 10,000 | 1,000,000 | 1.2 |
| S. aureus | mutation | 10,000 | 1,000,000 | 40 |
| E. coli | recombination | 100,000 | 1,000,000 | 4 |
| E. coli | mutation | 100,000 | 1,000,000 | 100 |

- **Defaults** of `detect_hotspots`:
  - h_N = 10 kb, h_B = 1 Mb, τ = 0.2;
  - probability threshold 0.9;
  - at most 5,000 posterior draws, chosen with `random_state=10`;
  - candidates further apart than `max_gap_bp=800_000` are not merged.
- **Positions:** genes are placed at their start positions and need not be
  sorted.
- **Tables:** `to_dataframe()` gives one row per region:
  - start and end (`end_position_bp` is the end of the last gene);
  - number of genes;
  - peak probability;
  - peak fold enrichment, the largest posterior-median fold enrichment;
  - median probability;
  - whether the region wraps the origin.

  `save_hotspot_tables` writes an Excel workbook (one sheet per table) when
  the path ends in `.xlsx`, and otherwise a folder of CSV files.
- **Non-positive draws** are replaced by machine epsilon before taking
  logs. With a prior from `build_prior` there are normally none.

## Paper sections and code

| Paper | Code |
|---|---|
| 2.1 ClonalOrigin model, S1 Algorithm 1 | `bacnpe.simulation`: `ARG`, `add_mutation`, `ClonalOrigin_seq_sim` |
| Clonal genealogy 𝒯 (simulated or estimated) | `bacnpe.tree`: `ClonalTree`, `ClonalTree.from_newick` |
| S2 Summary statistics | `bacnpe.summary_stats`: `segment_summary_stats`, `STAT_NAMES` |
| 3.x Training simulations | `bacnpe.parallel_sim` |
| 2.2 NPE-C, S3 Algorithm 2, S8 configuration | `bacnpe.npe`: `build_prior`, `DiscreteUniform`, `train_npe`, `save_npe`, `load_npe`, `sample_posteriors` |
| 2.3 Compression network | `bacnpe.npe.LeaveLengthOut_NN` |
| 2.5, S6 ppDoG detector, Tables S4–S6, Figures 4–9 | `bacnpe.detection`: `detect_hotspots`, `plot_dog_hotspots`, `save_hotspot_tables` |

## Package layout

```
bacnpe/
├── __init__.py              public API: ClonalTree, ARG, add_mutation, ClonalOrigin_seq_sim,
│                            segment_summary_stats, parallel_sim
├── tree/                    clonal genealogy
│   ├── clonal_genealogy.py  ClonalTree (simulate, from_newick/from_csv/from_arrays, to_csv/to_newick)
│   ├── newick.py            Newick reader and writer
│   ├── upgma.py             estimated phylogeny -> clonal genealogy (UPGMA + rescaling)
│   └── base.py              _Tree (base class, class name "Tree")
├── simulation/              ClonalOrigin simulator
│   ├── arg.py               ARG: recombination events and local genealogies
│   ├── mutation.py          add_mutation
│   └── seq_sim.py           ClonalOrigin_seq_sim
├── summary_stats/           summary statistics
│   ├── linkage.py           LD (D, D', r²), four-gamete test, Hudson's R_M, Kelly's Z_nS
│   ├── diversity.py         Watterson's θ, Tajima's π, Wakeley's π, Tajima's D
│   ├── homoplasy.py         homoplasy index (Fitch parsimony on the clonal genealogy)
│   ├── wall.py              Wall's B and Q
│   ├── regression.py        exponential decay fit of r² on distance
│   └── segment_stats.py     segment_summary_stats (the 30-entry vector), STAT_NAMES
├── parallel.py              parallel_sim: prior draws + parallel simulations to CSV
├── npe/                     neural posterior estimation (needs torch + sbi)
│   ├── prior.py             build_prior
│   ├── distributions.py     DiscreteUniform (prior on ℓ)
│   ├── embedding.py         LeaveLengthOut_NN (compression network)
│   ├── train.py             train_npe, NPEResult, build_density_estimator
│   ├── checkpoint.py        save_npe, load_npe
│   └── sample.py            sample_posteriors (posterior draws for every observed gene)
├── detection/               ppDoG hotspot detection
│   ├── dog.py               detect_hotspots, DoGHotspotDetector, DoGHotspotResult
│   ├── plot.py              plot_dog_hotspots (three-panel figure)
│   └── export.py            save_hotspot_tables (Excel / CSV)
└── _validation.py           input checking shared by the entry points
```

## Public API and input checks

The public names are:

- from `bacnpe`: `ClonalTree`, `ARG`, `add_mutation`,
  `ClonalOrigin_seq_sim`, `segment_summary_stats` and `parallel_sim`;
- `STAT_NAMES`;
- the names in `bacnpe.npe.__all__` and `bacnpe.detection.__all__`.

Everything else is an internal helper (underscore prefix) and may change
without notice.

| Argument     | Type  | Allowed values |
|--------------|-------|----------------|
| `n` (tree)   | int   | ≥ 1 (≥ 3 for `ClonalOrigin_seq_sim`) |
| `rho_site`   | float | ≥ 0, finite |
| `theta_site` | float | ≥ 0, finite |
| `L`          | int   | ≥ 1 (≥ 3 for summary statistics) |
| `delta`      | float | ≥ 1, finite |

Wrong types raise `TypeError`; out-of-range values raise `ValueError`.
NumPy scalars (`np.int64`, `np.float32`, …) are accepted.

## Tests

```bash
pytest
```

The suite checks:

- output types, shapes and value ranges;
- the handling of unpermitted inputs;
- the internal helpers on hand-made examples;
- "golden" vectors produced by the original implementation with fixed seeds;
- the conversion of estimated trees against the output of the original R
  scripts.

See `CHANGES.md` for what changed relative to the original scripts.

## Citation and license

If you use bacnpe, please cite the paper above. The code is released under
the MIT License (`LICENSE.txt`).
