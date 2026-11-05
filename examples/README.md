# Examples

These examples show the bacnpe workflow of the paper, from simulation to
hotspot detection. They need bacnpe installed (see the main README). The
notebook also needs Jupyter (`pip install notebook`, or the Jupyter
extension of VS Code).

| File | What it does | Run time |
|---|---|---|
| [`01_simulate.py`](01_simulate.py) | Simulates a clonal genealogy of 12 genomes and one gene of 1,000 bp under ClonalOrigin, prints its 30 summary statistics, then simulates a small training set of 200 genes with `parallel_sim` (4 workers, chunks of 50). | seconds |
| [`02_clonal_tree.py`](02_clonal_tree.py) | Turns the estimated *E. coli* genealogy `data/cf.nwk` into a clonal genealogy (UPGMA + rescaling to coalescent units) and saves `clonal_edge.csv`, `clonal_node_height.csv` and `tip_names.csv`. | seconds |
| [`03_ecoli_npe_hotspots.ipynb`](03_ecoli_npe_hotspots.ipynb) | The *E. coli* analysis of the paper: trains NPE on 50,000 simulations, samples the posterior of every alignment block, and detects recombination and mutation hotspots with ppDoG. | ~25 min training on a CPU, then a few minutes |

Run the scripts from the repository root or from this folder:

```bash
python examples/01_simulate.py
python examples/02_clonal_tree.py
```

Open the notebook with `examples` as the working directory, which is the
default in Jupyter and VS Code. The trained posterior is saved, and later
runs load it instead of training again.

## Data

| File | Content |
|---|---|
| `data/cf.nwk` | Clonal genealogy of 27 *E. coli* genomes, estimated by ClonalFrame (Newick). |
| `data/ecoli_theta.csv` | Parameters (ρ_s, θ_s, ℓ) of 50,000 simulations on the *E. coli* genealogy, with δ = 542 bp. |
| `data/ecoli_x.csv` | The 30 summary statistics of the same simulations, row by row. |
| `data/ecoli_block_info_updated.csv` | The 836 observed alignment blocks: length and start/end positions on *E. coli* 536. |
| `data/ecoli_block_summary_stats.csv` | Observed summary statistics of the blocks, in the older 46-column format. The notebook drops columns 16–31 to get the 30 used here. |

## Output

All results are written to `examples/output/`, which git ignores:

- `simulations/`: `params.csv` and `stats.csv` from example 1;
- `ecoli_tree/`: the clonal genealogy from example 2;
- from the notebook:
  - the trained posterior, `ecoli_trained_npe.pt`;
  - the hotspot figures, `ecoli_local_recomb_dog.pdf` and `ecoli_local_mutate_dog.pdf`;
  - the hotspot tables, `ecoli_hotspots.xlsx`.
