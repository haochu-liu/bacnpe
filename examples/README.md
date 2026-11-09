# Examples

These examples show the bacnpe workflow of the paper, from simulation to
hotspot detection. They need bacnpe installed (see the main README). The
notebooks also need Jupyter (`pip install notebook`, or the Jupyter
extension of VS Code).

| File | What it does
|---|---|
| [`01_simulate.py`](01_simulate.py) | Simulates a clonal genealogy of 12 genomes and one gene of 1,000 bp under ClonalOrigin, prints its 30 summary statistics, then simulates a small training set of 200 genes with `parallel_sim` (4 workers, chunks of 50).
| [`02_clonal_tree.py`](02_clonal_tree.py) | Turns the estimated *E. coli* genealogy `data/cf.nwk` into a clonal genealogy (UPGMA + rescaling to coalescent units) and saves `clonal_edge.csv`, `clonal_node_height.csv` and `tip_names.csv`.
| [`03_ecoli_npe_hotspots.ipynb`](03_ecoli_npe_hotspots.ipynb) | The *E. coli* analysis of the paper: trains NPE on 50,000 simulations, samples the posterior of every alignment block, and detects recombination and mutation hotspots with ppDoG.
| [`04_bacillus_npe_hotspots.ipynb`](04_bacillus_npe_hotspots.ipynb) | The same analysis for the *B. cereus* group: 1,277 alignment blocks of 13 genomes (Section 3.2 and Table S4 of the paper).
| [`05_staph_npe_hotspots.ipynb`](05_staph_npe_hotspots.ipynb) | The same analysis for *S. aureus*: 1,982 core genes of 110 genomes (Section 3.3 and Table S5 of the paper).

Run the scripts from the repository root or from this folder:

```bash
python examples/01_simulate.py
python examples/02_clonal_tree.py
```

Open the notebooks with `examples` as the working directory. Each one loads
its trained posterior from `data/` if the file exists, and otherwise trains
NPE and saves the posterior there.

## Data

| File | Content |
|---|---|
| `data/cf.nwk` | Clonal genealogy of 27 *E. coli* genomes, estimated by ClonalFrame (Newick). |
| `data/ecoli_theta.csv` | Parameters $(ρ_s, θ_s, ℓ)$ of 50,000 simulations on the *E. coli* genealogy, with δ = 542 bp. |
| `data/ecoli_x.csv` | The 30 summary statistics of the same simulations, row by row. |
| `data/ecoli_block_info.csv` | The 836 observed alignment blocks: length and start/end positions on *E. coli* 536. |
| `data/ecoli_block_summary_stats.csv` | Observed summary statistics of the blocks. |
| `data/ecoli_trained_npe.pt` | Trained posterior for *E. coli*. |
| `data/bacillus_theta.csv`, `data/bacillus_x.csv` | Parameters and summary statistics of 50,000 simulations on the *B. cereus* genealogy, with δ = 236 bp. |
| `data/bacillus_block_info.csv` | The 1,277 observed alignment blocks: length and start/end positions on *B. anthracis* Ames 0581. |
| `data/bacillus_block_summary_stats.csv` | Observed summary statistics of the blocks. |
| `data/bacillus_trained_npe.pt` | Trained posterior for *B. cereus*. |
| `data/staph_theta.csv`, `data/staph_x.csv` | Parameters and summary statistics of 50,000 simulations on the *S. aureus* genealogy, with δ = 454 bp. |
| `data/staph_core_gene_info.csv` | The 1,982 core genes: length and start/end positions on *S. aureus* MRSA252. |
| `data/staph_core_gene_summary_stats.csv` | Observed summary statistics of the genes. |
| `data/staph_trained_npe.pt` | Trained posterior for *S. aureus*. |

## Output

Results are written to `examples/output/`, which git ignores:

- `simulations/`: `params.csv` and `stats.csv` from example 1;
- `ecoli_tree/`: the clonal genealogy from example 2;
- `ecoli_hotspots.xlsx`, `bacillus_hotspots.xlsx` and `staph_hotspots.xlsx`:
  the hotspot tables from the notebooks, with one sheet per rate.

The notebooks show the hotspot figures inline.

## Differences from the paper

Results from these examples may differ from those reported in the paper.

- **Posterior results.** Neural posterior estimation is stochastic: the
  simulations, the initial network weights, the training batches and the
  posterior draws are all random. Seeds are set in the examples, but the same
  seed can give different numbers with another version of PyTorch or sbi, or
  on other hardware. If you retrain NPE, or generate new simulations, the
  posterior of each gene can differ from the paper, so the rates, intervals
  and hotspot probabilities can differ too.
- **Running time.** The times reported in the paper were measured on specific
  machines. Simulation, training and sampling times depend on your hardware,
  the number of workers and the number of epochs until early stopping, which
  itself varies between training runs.
- **`min_fold_enrichment` for ppDoG.** The values in the notebooks (Table S1
  of the paper) were chosen for the posteriors of the paper. They are not
  universal constants. Because fold enrichment is computed from the posterior
  draws, a different posterior can change its range, and the same threshold
  can then give more, fewer or different hotspot regions. If you retrain the
  model, inspect the fold-enrichment panel of the figures and adjust
  `min_fold_enrichment` if needed, as described in Supplementary Text S6, and
  check that the main regions remain stable under moderate changes of it and
  of the narrow bandwidth.
