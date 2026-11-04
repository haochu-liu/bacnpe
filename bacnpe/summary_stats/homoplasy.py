"""Homoplasy index of an incidence matrix on the clonal tree (Fitch parsimony)."""
from __future__ import annotations

import numpy as np

from ..tree.clonal_genealogy import ClonalTree


def _homoplasy_index(tree: ClonalTree, node_site: np.ndarray) -> float:
    """
    Calculate the homoplasy index for an ARG with mutations.

    The homoplasy index measures the amount of homoplasy (parallel or convergent
    evolution) in the data. It is calculated as 1 - m/s, where m is the minimum
    number of changes required (1 per segregating site), and s is the number
    of changes needed on the clonal tree (Fitch parsimony score).

    Parameters
    ----------
    tree : ClonalTree
        A clonal tree object representing the clonal genealogy for the ARG.
    node_site : np.ndarray
        Node-site incidence matrix (boolean), typically from add_mutation().
        Shape: (n_nodes, n_sites); only the first ``tree.n`` rows (the leaves)
        are used.

    Returns
    -------
    float
        The homoplasy index, ranging from 0 (no homoplasy) to 1 (maximum homoplasy).
    """
    n_leaf = tree.n
    leaf_states = np.asarray(node_site[:n_leaf], dtype=bool)
    n_site = leaf_states.shape[1]

    # Minimum number of changes: one per segregating site
    m_sum = int(np.count_nonzero(leaf_states.any(axis=0) & ~leaf_states.all(axis=0)))

    # Compute actual changes using Fitch algorithm, for all sites at once.
    # State sets are encoded as bit masks: {0} -> 0b01, {1} -> 0b10, {0, 1} -> 0b11
    state_sets = np.empty((2 * n_leaf - 1, n_site), dtype=np.uint8)
    # Initialize leaf nodes
    state_sets[:n_leaf] = np.where(leaf_states, 2, 1)

    # Children of every internal node (node ids are 1-indexed)
    children = {}
    for parent, child in tree.edge[:, :2].astype(int):
        children.setdefault(parent, []).append(child)

    # Process internal nodes (children always have smaller ids than parents)
    s_sum = 0
    for parent_node in range(n_leaf + 1, 2 * n_leaf):
        child_1, child_2 = children[parent_node]
        child_1_states = state_sets[child_1 - 1]
        child_2_states = state_sets[child_2 - 1]

        intersec = child_1_states & child_2_states
        # Intersection is empty -> mutation occurred, take the union;
        # otherwise no mutation, keep the intersection
        empty = intersec == 0
        state_sets[parent_node - 1] = np.where(empty, child_1_states | child_2_states, intersec)
        s_sum += int(np.count_nonzero(empty))

    # Calculate homoplasy index
    if m_sum == 0 and s_sum == 0:
        return 0.0

    return 1 - m_sum / s_sum
