"""Mutation process on an ARG (infinite-site style, uniform over edges and sites)."""
from __future__ import annotations

import numpy as np

from .._validation import _check_real
from .arg import ARG


def add_mutation(arg: ARG, theta_site: float) -> np.ndarray:
    """
    Add mutations uniformly onto the edges of ARG with infinite site assumption.

    The root sequence is drawn uniformly at random, the number of mutations
    is Poisson with mean ``theta_site * L * arg.length / 2``, each mutation
    falls on an edge with probability proportional to its length and on a
    uniformly chosen site. Mutations on sites that are not ancestral material
    of their edge are discarded. States are then propagated from the root
    down to the leaves.

    Parameters
    ----------
    arg : ARG
        An ARG object to add mutations to.
    theta_site : float
        The mutation rate per site.

    Returns
    -------
    node_site : np.ndarray, shape (n_nodes, L), dtype bool
        Node-site incidence matrix (allelic state of every node at every site).

    Raises
    ------
    TypeError
        If `arg` is not an ARG or `theta_site` is not a real number.
    ValueError
        If `theta_site` is negative, NaN or infinite.
    """
    if not isinstance(arg, ARG):
        raise TypeError("`arg` must be of class 'ARG'.")
    theta_site = _check_real(theta_site, "theta_site", minimum=0.0)

    n_nodes, n_sites = arg.node_mat.shape
    theta = theta_site * n_sites
    n_mutations = np.random.poisson(theta * arg.length / 2)

    # Initialize node_site matrix (boolean) with a random root sequence
    root_node = np.random.choice([True, False], size=n_sites)
    node_site = np.tile(root_node, (n_nodes, 1))

    # If there is no mutation
    if n_mutations == 0:
        return node_site

    # If there are mutations
    # Sample edges with probability proportional to edge length
    edge_probs = np.clip(arg.edge[:, 2], a_min=0, a_max=None)
    edge_probs = edge_probs / np.sum(edge_probs)
    mutate_edge = np.random.choice(len(arg.edge), n_mutations, replace=True, p=edge_probs)
    # Sample sites uniformly (0-indexed)
    mutate_site = np.random.choice(n_sites, n_mutations, replace=True)

    # Ignore mutations not in the edge material
    keep_mutation = arg.edge_mat[mutate_edge, mutate_site].astype(bool)
    mutate_edge = mutate_edge[keep_mutation]
    mutate_site = mutate_site[keep_mutation]

    if mutate_edge.size == 0:
        return node_site

    # Pre-compute, for every edge carrying mutations, which sites are flipped
    # (a site hit an even number of times on the same edge is not flipped).
    flips = {}
    order = np.argsort(mutate_edge, kind="stable")
    edges_sorted = mutate_edge[order]
    unique_edges, starts = np.unique(edges_sorted, return_index=True)
    for e, sites in zip(unique_edges, np.split(mutate_site[order], starts[1:])):
        flips[int(e)] = np.bincount(sites, minlength=n_sites) % 2 == 1

    # Simulate the mutations at every node.
    # Edges are stored from the bottom of the ARG to the top, so going from
    # the last edge to the first processes parents before their children
    # (top-down, from the root to the leaves).
    for i in range(len(arg.edge) - 1, -1, -1):
        parent_idx = int(arg.edge[i, 0]) - 1
        child_idx = int(arg.edge[i, 1]) - 1

        parent_seq = node_site[parent_idx, :]
        if i in flips:
            parent_seq = np.logical_xor(parent_seq, flips[i])

        # Only the ancestral material carried by this edge is inherited
        material_range = arg.edge_mat[i, :]
        node_site[child_idx, material_range] = parent_seq[material_range]

    return node_site
