"""Public sequence-simulation entry point and internal mutation helper."""

from __future__ import annotations

from numbers import Real

import numpy as np

from ._arg import _ARG
from ._clonal_tree import ClonalTree
from ._statistics import _segment_summary_stats


def _validate_rate(name: str, value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return value


def _add_mutation(arg: _ARG, theta_site: float) -> np.ndarray:
    theta = theta_site * arg.node_mat.shape[1]
    mutation_count = np.random.poisson(theta * arg.length / 2)
    root_state = np.random.choice([True, False], size=arg.node_mat.shape[1])
    node_site = np.tile(root_state, (arg.node_mat.shape[0], 1))
    if mutation_count == 0:
        return node_site

    lengths = np.clip(arg.edge[:, 2], 0, None)
    total_length = float(np.sum(lengths))
    if total_length <= 0:
        return node_site
    mutated_edges = np.random.choice(
        len(arg.edge), mutation_count, replace=True, p=lengths / total_length
    )
    mutated_sites = np.random.choice(arg.node_mat.shape[1], mutation_count, replace=True)
    keep = arg.edge_mat[mutated_edges, mutated_sites].astype(bool)
    mutated_edges, mutated_sites = mutated_edges[keep], mutated_sites[keep]

    for edge_index in range(len(arg.edge) - 1, -1, -1):
        edge_mutations = mutated_sites[mutated_edges == edge_index]
        parent = int(arg.edge[edge_index, 0]) - 1
        child = int(arg.edge[edge_index, 1]) - 1
        parent_sequence = node_site[parent].copy()
        flips = np.bincount(edge_mutations, minlength=len(parent_sequence)) % 2 == 1
        parent_sequence = np.logical_xor(parent_sequence, flips)
        material = arg.edge_mat[edge_index].astype(bool)
        node_site[child, material] = parent_sequence[material]
    return node_site


def ClonalOrigin_seq_sim(
    tree: ClonalTree,
    rho_site: Real,
    theta_site: Real,
    L: int,
    delta: Real,
) -> np.ndarray:
    """Simulate sequence data and return 46 summary statistics.

    Parameters
    ----------
    tree
        A :class:`ClonalTree` with at least two leaves.
    rho_site
        Non-negative recombination rate per site.
    theta_site
        Non-negative mutation rate per site.
    L
        Positive integer sequence length.
    delta
        Mean recombinant tract length. It must be at least one site.
    """
    if not isinstance(tree, ClonalTree):
        raise TypeError("`tree` must be a bacnpe.ClonalTree instance.")
    if tree.n < 2:
        raise ValueError("`tree` must contain at least two leaves.")
    if isinstance(L, (bool, np.bool_)) or not isinstance(L, (int, np.integer)):
        raise TypeError("`L` must be a positive integer.")
    if L < 1:
        raise ValueError("`L` must be a positive integer.")

    rho = _validate_rate("rho_site", rho_site)
    theta = _validate_rate("theta_site", theta_site)
    if isinstance(delta, (bool, np.bool_)) or not isinstance(delta, Real):
        raise TypeError("`delta` must be a real number.")
    delta = float(delta)
    if not np.isfinite(delta) or delta < 1:
        raise ValueError("`delta` must be finite and at least 1.")

    arg = _ARG(tree, rho, int(L), delta, int(L))
    node_site = _add_mutation(arg, theta)
    leaf_matrix = node_site[: tree.n] != node_site[0]
    return _segment_summary_stats(tree, leaf_matrix)
