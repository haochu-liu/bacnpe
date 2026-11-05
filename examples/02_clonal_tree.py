"""
Example 2: turn an estimated phylogeny into a clonal genealogy.

data/cf.nwk is the clonal genealogy of 27 E. coli genomes estimated by
ClonalFrame. ClonalTree.from_newick makes it ultrametric by UPGMA and rescales
its root to the coalescent expectation 2(1 - 1/n), so it can be used by the
simulator. The result is saved as

    clonal_edge.csv         [parent, child, length] for each edge
    clonal_node_height.csv  height of nodes 1..2n-1 (leaves 1..n at height 0)
    tip_names.csv           genome name of leaves 1..n

Run from anywhere:

    python examples/02_clonal_tree.py
"""
from pathlib import Path

import numpy as np

from bacnpe import ClonalTree

HERE = Path(__file__).resolve().parent
NEWICK = HERE / "data" / "cf.nwk"
OUTPUT = HERE / "output" / "ecoli_tree"


def main():
    tree = ClonalTree.from_newick(NEWICK)
    print(f"{tree} from {NEWICK.name}")
    print(f"  root height {tree.height:.4f} (2(1 - 1/n) = {2 * (1 - 1 / tree.n):.4f})")
    print(f"  total branch length {tree.length:.4f}")
    print(f"  first tips: {tree.tip_labels[:5]}")
    print("  first edges [parent, child, length]:")
    print(tree.edge[:4])

    tree.to_csv(OUTPUT, overwrite=True)
    print(f"\nSaved {sorted(p.name for p in OUTPUT.iterdir())} in {OUTPUT}")

    # The saved tree loads back identically and can be passed to the simulator
    reloaded = ClonalTree.from_csv(OUTPUT)
    assert np.array_equal(reloaded.edge, tree.edge)
    assert reloaded.tip_labels == tree.tip_labels


if __name__ == "__main__":
    main()
