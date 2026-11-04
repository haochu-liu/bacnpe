"""
Approximated ancestral recombination graph (ARG) under the ClonalOrigin model.

This module merges the original ``ClonalOrigin_ARG.py``, ``seq_simulator.py``
and ``ClonalOrigin_nodes.py``. The simulation is split into two steps:

1. :func:`_simulate_recombination_events` - draw the recombination edges
   (all random number generation happens here).
2. :func:`_build_arg_graph` - deterministically assemble the full ARG
   (edges, node heights and ancestral material) from those events.

Node / edge conventions (1-indexed, as in the original R implementation)
------------------------------------------------------------------------
* Clonal edges are numbered ``1..2n-2`` (row order of ``ClonalTree.edge``);
  edge ``2n-1`` denotes the virtual edge above the clonal root.
* ``recomb_edge`` has one row per recombination event with columns
  ``[b_edge, b_height, a_edge, a_height, x, y]``: the recombinant lineage
  leaves clonal edge ``b_edge`` at time ``b_height`` and coalesces back onto
  clonal edge ``a_edge`` at time ``a_height``; it carries sites ``x..y``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .._validation import _check_integer, _check_real
from ..tree.base import _Tree
from ..tree.clonal_genealogy import ClonalTree


class ARG(_Tree):
    """
    An approximated ancestral recombination graph (ARG) using ClonalOrigin algorithm.

    Simulate coalescent and recombination events by the ClonalOrigin algorithm
    along a sequence of ``L`` sites, conditioned on a given clonal genealogy.

    Attributes
    ----------
    edge : np.ndarray, shape (n_edges, 3), dtype float64
        Edge matrix with columns [node1 (parent), node2 (child), length].
        Node ids are 1-indexed rows of ``node_height`` / ``node_mat``.
    edge_mat : np.ndarray, shape (n_edges, L), dtype bool
        Edge material matrix: which sites are ancestral on each edge.
    node_height : np.ndarray, shape (n_nodes,), dtype float64
        Node heights, sorted in increasing order.
    node_mat : np.ndarray, shape (n_nodes, L), dtype bool
        Node material matrix: which sites are ancestral at each node.
    node_clonal : np.ndarray, shape (n_nodes,), dtype bool
        Boolean array indicating clonal nodes.
    rho : float
        Total recombination rate (L * rho_site).
    L : int
        Number of sites.
    delta : float
        Mean recombinant segment length.
    height : float
        Height of the ARG.
    length : float
        Total length of the ARG.
    """

    def __init__(self, tree_obj: ClonalTree, rho_site: float, L: int,
                 delta: float):
        """
        Initialize an ARG by simulating recombination on a clonal tree.

        Parameters
        ----------
        tree_obj : ClonalTree
            The clonal genealogy.
        rho_site : float
            The recombination parameter per site (>= 0).
        L : int
            The number of sites (>= 1).
        delta : float
            The mean of recombinant segment length (>= 1).
        """
        # Input validation
        if not isinstance(tree_obj, ClonalTree):
            raise TypeError("`tree_obj` must be of class 'ClonalTree'.")
        rho_site = _check_real(rho_site, "rho_site", minimum=0.0)
        L = _check_integer(L, "L", minimum=1)
        delta = _check_real(delta, "delta", minimum=1.0)

        super().__init__(tree_obj.n)

        self.rho = L * rho_site
        self.L = L
        self.delta = delta

        self._simulate(tree_obj, rho_site, delta, L)

    def _simulate(self, tree_obj: ClonalTree, rho_site: float, delta: float, k: int) -> None:
        """Simulate the ARG using ClonalOrigin algorithm and store the result."""
        n = self.n
        clonal_edge = tree_obj.edge.copy()
        clonal_node_height = tree_obj.node_height.copy()

        recomb_edge = _simulate_recombination_events(
            clonal_edge, clonal_node_height, n, rho_site, delta, k
        )

        if recomb_edge.shape[0] == 0:
            # Handle case with no recombination: the ARG is the clonal tree
            self.edge = clonal_edge
            self.edge_mat = np.full((2 * (n - 1), k), True)
            self.node_height = clonal_node_height
            self.node_mat = np.full((2 * n - 1, k), True)
            self.node_clonal = np.full(2 * n - 1, True)
        else:
            (self.edge, self.edge_mat, self.node_height,
             self.node_mat, self.node_clonal) = _build_arg_graph(
                clonal_edge, clonal_node_height, recomb_edge, n, k
            )

        self.height = float(np.max(self.node_height))
        self.length = float(np.sum(self.edge[:, 2]))

    def __repr__(self) -> str:
        return f"ARG(n={self.n}, rho={self.rho}, L={self.L}, delta={self.delta})"

    def to_dict(self) -> dict:
        """Returns the ARG as a dictionary, similar to the R list output."""
        return {
            "edge": self.edge,
            "edge_mat": self.edge_mat,
            "node_height": self.node_height,
            "node_mat": self.node_mat,
            "node_clonal": self.node_clonal,
            "n": self.n,
            "rho": self.rho,
            "L": self.L,
            "delta": self.delta,
            "height": self.height,
            "length": self.length,
        }


def _clonal_origin_nodes(recomb_edge: np.ndarray, edge_id: int) -> np.ndarray:
    """
    Find the recombination nodes on a given clonal edge, sorted by height.

    Parameters
    ----------
    recomb_edge : np.ndarray
        A recomb_edge matrix (columns b_edge, b_height, a_edge, a_height, x, y).
    edge_id : int
        The (1-indexed) edge index.

    Returns
    -------
    np.ndarray
        Signed recombination codes of the nodes on the edge, from the lowest
        to the highest node. A positive value ``r`` means the "in" (a) node of
        recombination ``r`` (row ``r`` of `recomb_edge`, 1-indexed, where
        column 3 matches `edge_id`); a negative value ``-r`` means the
        "out" (b) node of recombination ``r`` (column 1 matches `edge_id`).
    """
    # Find rows where column 3 (0-indexed: 2) equals edge_id -> type a
    rows_type_a = np.where(recomb_edge[:, 2] == edge_id)[0]
    # Find rows where column 1 (0-indexed: 0) equals edge_id -> type b
    rows_type_b = np.where(recomb_edge[:, 0] == edge_id)[0]

    if rows_type_a.size + rows_type_b.size == 0:
        return np.array([], dtype=float)

    # Column 0: signed code, column 1: height of the node
    found_nodes = np.concatenate([
        np.column_stack([rows_type_a + 1.0, recomb_edge[rows_type_a, 3]]),
        np.column_stack([-(rows_type_b + 1.0), recomb_edge[rows_type_b, 1]]),
    ])

    if found_nodes.shape[0] > 1:
        # Sort by height (second column)
        found_nodes = found_nodes[found_nodes[:, 1].argsort()]

    return found_nodes[:, 0]


def _simulate_recombination_events(clonal_edge: np.ndarray, clonal_node_height: np.ndarray,
                                   n: int, rho_site: float, delta: float, k: int) -> np.ndarray:
    """
    Simulate the recombination events sequentially along the sites.

    Moving from site 1 to site ``k``: new recombinations start at each site,
    and each recombination that covers the previous site extends to the
    current site with probability ``1 - 1/delta`` (geometric segment lengths
    with mean `delta`).

    Parameters
    ----------
    clonal_edge : np.ndarray
        Edge matrix of the clonal tree.
    clonal_node_height : np.ndarray
        Node heights of the clonal tree.
    n : int
        Number of leaves.
    rho_site : float
        The recombination parameter per site.
    delta : float
        The mean of recombinant segment length.
    k : int
        The number of sites.

    Returns
    -------
    np.ndarray, shape (n_recomb, 6)
        The recomb_edge matrix with columns
        ``[b_edge, b_height, a_edge, a_height, x, y]``.
    """
    # Tree length
    tree_length = np.sum(clonal_edge[:, 2])
    # Probability of each clonal edge to receive a recombination
    if n > 1:
        edge_probs = clonal_edge[:, 2] / np.sum(clonal_edge[:, 2])

    # Initialize recombination edges
    nrow_max = 1000
    # Columns: b_edge, b_height, a_edge, a_height, x, y
    recomb_edge = np.full((nrow_max, 6), np.nan)

    # Add recombination sequentially
    n_recomb = 0
    remain_index = np.array([], dtype=int)

    for i in range(1, k + 1):  # i = 1, 2, ..., k
        if i == 1:
            # Recombinations already covering site 1 (stationary start)
            R_new = np.random.poisson(rho_site * delta * tree_length / 2)
            R_old = 0
        else:  # i == 2, ..., k
            survive_index = (np.where(recomb_edge[:n_recomb, 5] == (i - 1))[0]
                             if n_recomb > 0 else np.array([], dtype=int))
            R_new = np.random.poisson(rho_site * tree_length / 2)

            # Recombinations covering site i-1 that extend to site i
            R_old = np.random.binomial(1 * survive_index.size, (1 - 1 / delta))
            remain_index = np.random.choice(survive_index, size=R_old, replace=False)

        if R_new > 0:
            # Expand matrix if needed
            if n_recomb + R_new >= nrow_max:
                recomb_edge = np.vstack([recomb_edge, np.full((nrow_max + n_recomb + R_new, 6), np.nan)])
                nrow_max = 2 * nrow_max + n_recomb + R_new

            # Set x and y columns
            recomb_edge[n_recomb:n_recomb + R_new, 4] = i  # x
            recomb_edge[n_recomb:n_recomb + R_new, 5] = i  # y

            a_rexp = np.random.exponential(1.0, size=R_new)

            # Simulate b_edge (similar to mutation)
            # Sample edges with probability proportional to edge length
            recomb_edge[n_recomb:n_recomb + R_new, 0] = np.random.choice(
                range(1, (2 * n - 1)), R_new, replace=True, p=edge_probs
            )

            for j in range(R_new):
                idx = n_recomb + j
                b_edge_idx = int(recomb_edge[idx, 0]) - 1

                # Simulate b_height uniformly along the chosen edge
                recomb_edge[idx, 1] = (
                    np.random.uniform(0, clonal_edge[b_edge_idx, 2]) +
                    clonal_node_height[int(clonal_edge[b_edge_idx, 1]) - 1]
                )

                # Identify a_height
                # t_above_b: heights of internal nodes minus b_height
                t_above_b = clonal_node_height[n:2 * n - 1] - recomb_edge[idx, 1]

                # Get non-negative values (nodes above b)
                positive_t = t_above_b[t_above_b >= 0]

                # Lengths of the inter-coalescence intervals above b
                i_above_b = np.diff(np.concatenate([[0], np.sort(positive_t)]))

                # Cumulative coalescence "pressure": interval length times the
                # number of lineages present in that interval
                num_intervals = len(i_above_b)
                lineage_counts = np.arange(num_intervals + 1, 1, -1)
                cuml_above_b = np.cumsum(i_above_b * lineage_counts)

                # Determine number of lineages at coalescence time
                num_lineage = (num_intervals + 1) - np.sum(a_rexp[j] > cuml_above_b)

                if num_lineage == (num_intervals + 1):
                    recomb_edge[idx, 3] = a_rexp[j] / num_lineage + recomb_edge[idx, 1]
                else:
                    idx_cuml = num_intervals - num_lineage
                    recomb_edge[idx, 3] = (
                        (a_rexp[j] - cuml_above_b[idx_cuml]) / num_lineage +
                        np.sum(i_above_b[:idx_cuml + 1]) +
                        recomb_edge[idx, 1]
                    )

                # Simulate a_edge
                if num_lineage > 1:
                    a_height = recomb_edge[idx, 3]
                    # Find edges that span the a_height
                    pool_edge = np.where(
                        (clonal_node_height[clonal_edge[:, 0].astype(int) - 1] >= a_height) &
                        (clonal_node_height[clonal_edge[:, 1].astype(int) - 1] <  a_height)
                    )[0] + 1
                    recomb_edge[idx, 2] = np.random.choice(pool_edge)
                else:
                    # Above the clonal root: the virtual root edge 2n-1
                    recomb_edge[idx, 2] = 2 * n - 1

        if R_old > 0 and len(remain_index) > 0:
            # Extend the surviving recombinations to site i
            recomb_edge[remain_index, 5] = i

        n_recomb = n_recomb + R_new

    # Trim recomb_edge to actual size
    return recomb_edge[:n_recomb, :]


def _build_arg_graph(clonal_edge: np.ndarray, clonal_node_height: np.ndarray,
                     recomb_edge: np.ndarray, n: int, k: int):
    """
    Assemble the full ARG from the clonal tree and the recombination events.

    Every recombination adds three nodes: two "out" (b) nodes at ``b_height``
    (the clonal continuation, carrying the sites outside ``x..y``, and the
    recombinant lineage, carrying sites ``x..y``) and one "in" (a) node at
    ``a_height`` where the recombinant lineage joins edge ``a_edge``.
    Nodes are processed from the bottom to the top of the graph and the
    ancestral material of each node is tracked.

    Parameters
    ----------
    clonal_edge : np.ndarray
        Edge matrix of the clonal tree.
    clonal_node_height : np.ndarray
        Node heights of the clonal tree.
    recomb_edge : np.ndarray
        The recomb_edge matrix (at least one row).
    n : int
        Number of leaves.
    k : int
        Number of sites.

    Returns
    -------
    tuple of np.ndarray
        ``(edge, edge_mat, node_height, node_mat, node_clonal)`` as described
        in :class:`ARG`.
    """
    n_recomb = recomb_edge.shape[0]
    node_max = 2 * n - 1 + 3 * n_recomb
    edge_max = 2 * (n - 1) + 4 * n_recomb

    edge_matrix = np.full((edge_max, 3), np.nan)
    # Row of node_mat holding the material of each edge (0-indexed)
    edge_mat_row = np.zeros(edge_max, dtype=np.int64)
    # Every row is fully assigned below, the initial value is irrelevant
    node_mat = np.zeros((node_max, k), dtype=bool)
    # Columns: index, height, recomb, clonal
    node_info = np.full((node_max, 4), np.nan)

    node_info[:, 0] = np.arange(1, node_max + 1)

    # Set clonal node info
    node_info[:2 * n - 1, 1] = clonal_node_height
    node_info[:2 * n - 1, 2] = 0
    node_info[:2 * n - 1, 3] = 1  # True -> 1

    # Set recombination out nodes (b nodes)
    # Interleave: for each recomb event, add two nodes at b_height
    r_idx = np.arange(n_recomb)
    b_rows = 2 * n - 1 + 2 * r_idx
    node_info[b_rows, 1] = recomb_edge[:, 1]           # b_height
    node_info[b_rows, 2] = -(r_idx + 1)                # negative recomb index (1-indexed)
    node_info[b_rows, 3] = 1                           # clonal = True
    node_info[b_rows + 1, 1] = recomb_edge[:, 1]       # same b_height
    node_info[b_rows + 1, 2] = np.nan                  # NA
    node_info[b_rows + 1, 3] = 0                       # clonal = False

    # Set recombination in nodes (a nodes)
    a_rows = 2 * n - 1 + 2 * n_recomb + r_idx
    node_info[a_rows, 1] = recomb_edge[:, 3]           # a_height
    node_info[a_rows, 2] = r_idx + 1                   # positive recomb index (1-indexed)
    node_info[a_rows, 3] = 1                           # clonal = True

    # Sort by height (ties broken by original index)
    sort_order = np.lexsort((node_info[:, 0], node_info[:, 1]))
    node_info = node_info[sort_order, :]

    # Lookup tables (replace the linear searches of the original code):
    # row_of_id[node_id - 1] -> row of that node after sorting
    row_of_id = np.argsort(node_info[:, 0])
    # row_of_code[signed recombination code] -> row of that node after sorting
    codes = node_info[:, 2]
    row_of_code = {float(c): row for row, c in enumerate(codes) if c == c and c != 0}

    # Clonal edges (0-indexed) grouped by their parent node id
    children_edges = {}
    for e, parent in enumerate(clonal_edge[:, 0].astype(int)):
        children_edges.setdefault(parent, []).append(e)

    # Recombination nodes on every edge (edge ids 1..2n-1, list is 0-indexed)
    recomb_node = [_clonal_origin_nodes(recomb_edge, edge_id) for edge_id in range(1, 2 * n)]

    def row_below(edge_pos: int, code: float, top_clonal_id: int) -> int:
        """Row of the node directly below recombination node `code` on an edge."""
        pos = int(np.argmax(recomb_node[edge_pos] == code))
        if pos == 0:
            # Lowest recombination node on the edge: the child is the clonal node
            return int(row_of_id[top_clonal_id - 1])
        return row_of_code[float(recomb_node[edge_pos][pos - 1])]

    # Build ARG edges and track ancestral material
    i = n  # Leaves occupy rows 0..n-1
    node_mat[:n, :] = True
    edge_index = 0

    while i < node_max:
        recomb_val = node_info[i, 2]

        if recomb_val == 0:
            # Clonal tree node: children are the top-most nodes on its two edges
            node_id = int(node_info[i, 0])
            child_rows = np.empty(2, dtype=np.int64)
            for c, e in enumerate(children_edges[node_id]):
                if len(recomb_node[e]) > 0:
                    # Target node is the last (highest) element
                    child_rows[c] = row_of_code[float(recomb_node[e][-1])]
                else:
                    child_rows[c] = row_of_id[int(clonal_edge[e, 1]) - 1]

            # Append edges
            edge_matrix[edge_index:edge_index + 2, 0] = i + 1
            edge_matrix[edge_index:edge_index + 2, 1] = child_rows + 1
            edge_matrix[edge_index:edge_index + 2, 2] = node_info[i, 1] - node_info[child_rows, 1]
            edge_mat_row[edge_index:edge_index + 2] = child_rows

            # Append root node material
            node_mat[i, :] = node_mat[child_rows[0], :] | node_mat[child_rows[1], :]

            edge_index += 2
            i += 1

        elif recomb_val < 0:
            # Recombination edge out node (rows i and i+1 share the same child)
            recomb_idx = int(-recomb_val) - 1
            edge_pos = int(recomb_edge[recomb_idx, 0]) - 1
            child_row = row_below(edge_pos, recomb_val, int(clonal_edge[edge_pos, 1]))

            # Append edges
            edge_matrix[edge_index:edge_index + 2, 0] = [i + 1, i + 2]
            edge_matrix[edge_index:edge_index + 2, 1] = child_row + 1
            edge_matrix[edge_index:edge_index + 2, 2] = node_info[i, 1] - node_info[child_row, 1]
            edge_mat_row[edge_index:edge_index + 2] = [i, i + 1]

            x = int(recomb_edge[recomb_idx, 4])
            y = int(recomb_edge[recomb_idx, 5])

            # Append root node material
            # Sites x..y (1-indexed, inclusive) go to the recombinant lineage
            # (row i+1); all other sites stay on the clonal lineage (row i).
            in_segment = np.zeros(k, dtype=bool)
            in_segment[x - 1:y] = True
            node_mat[i + 1, :] = node_mat[child_row, :] & in_segment
            node_mat[i, :] = node_mat[child_row, :] & ~in_segment

            edge_index += 2
            i += 2

        elif recomb_val > 0:
            # Recombination edge in node
            recomb_idx = int(recomb_val) - 1
            edge_pos = int(recomb_edge[recomb_idx, 2]) - 1
            if edge_pos == 2 * n - 2:
                # Virtual edge above the clonal root: lowest node is the root
                top_clonal_id = 2 * n - 1
            else:
                top_clonal_id = int(clonal_edge[edge_pos, 1])
            child_rows = np.array([
                row_below(edge_pos, recomb_val, top_clonal_id),
                # The recombinant (non-clonal) b node sits right after the clonal one
                row_of_code[-float(recomb_val)] + 1,
            ], dtype=np.int64)

            # Append edges
            edge_matrix[edge_index:edge_index + 2, 0] = i + 1
            edge_matrix[edge_index:edge_index + 2, 1] = child_rows + 1
            edge_matrix[edge_index:edge_index + 2, 2] = node_info[i, 1] - node_info[child_rows, 1]
            edge_mat_row[edge_index:edge_index + 2] = child_rows

            # Append root node material
            node_mat[i, :] = node_mat[child_rows[0], :] | node_mat[child_rows[1], :]

            edge_index += 2
            i += 1
        else:
            # NaN case (recombinant b node, already handled with its pair) - skip
            i += 1

    edge_mat = node_mat[edge_mat_row, :]
    node_height = node_info[:, 1]
    node_clonal = node_info[:, 3].astype(bool)
    return edge_matrix, edge_mat, node_height, node_mat, node_clonal
