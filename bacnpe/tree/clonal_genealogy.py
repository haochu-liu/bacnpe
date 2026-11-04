"""Clonal genealogy (Kingman coalescent tree) simulation."""
from __future__ import annotations

import numpy as np

from .._validation import _check_integer
from .base import _Tree


class ClonalTree(_Tree):
    """
    Clonal Genealogy simulation.

    Simulate coalescent backwards in time to construct a clonal tree.

    Node numbering (1-indexed, as in the original R implementation):
    leaves are ``1..n`` and internal nodes are ``n+1..2n-1`` in order of
    coalescence, so node ``2n-1`` is the root.

    Attributes
    ----------
    n : int
        The number of leaf lineages.
    edge : np.ndarray, shape (2*(n-1), 3), dtype float64
        One row per edge: ``[parent node, child node, edge length]``.
        Edges are stored in order of coalescence (two rows per event).
    node_height : np.ndarray, shape (2*n-1,), dtype float64
        Node heights to recent time (leaves are at height 0).
    height : float
        Height of the tree (time to the most recent common ancestor).
    length : float
        Total branch length of the tree.
    """

    def __init__(self, n: int):
        """
        Initialize a clonal tree by simulating coalescent backwards in time.

        Parameters
        ----------
        n : int
            The number of leaf lineages (a positive integer; Python ``int``
            or NumPy integer).

        Raises
        ------
        TypeError
            If `n` is not an integer (floats and booleans are rejected).
        ValueError
            If `n` is smaller than 1.
        """
        n = _check_integer(n, "n", minimum=1)
        super().__init__(n)
        self.__class__.__name__ = "ClonalTree"
        self._simulate()

    def _simulate(self) -> None:
        """Run the coalescent simulation and fill in the tree attributes."""
        n = self.n
        k = n
        t_sum = 0.0

        # Initialize variables for clonal tree
        # Cols: node1 (parent), node2 (child), length
        self.edge = np.zeros((2 * (n - 1), 3))
        self.node_height = np.zeros(2 * n - 1)

        # Initialize variables and vector
        edge_index = 0
        node_index = n + 1
        # NOTE: the order of `pool` matters for reproducibility, because
        # np.random.choice samples by position.
        pool = list(range(1, n + 1))

        # Clonal tree by coalescent only
        while k > 1:
            # Sample a new event time, rate is k*(k-1)/2.
            rate = k * (k - 1) / 2.0
            event_time = np.random.exponential(scale=1.0 / rate)
            t_sum += event_time

            # Coalescent event: sample 2 distinct nodes from the pool
            leaf_nodes = np.random.choice(pool, size=2, replace=False)

            # Append edges
            # We add two edges: Parent -> Child 1 and Parent -> Child 2
            # 1. Parent index (node1)
            self.edge[edge_index:edge_index + 2, 0] = node_index
            # 2. Child indices (node2)
            self.edge[edge_index:edge_index + 2, 1] = leaf_nodes
            # 3. Edge lengths (current time - height of child)
            child_heights = self.node_height[leaf_nodes - 1]
            self.edge[edge_index:edge_index + 2, 2] = t_sum - child_heights

            # Append root node height
            self.node_height[node_index - 1] = t_sum

            # Updates for iteration: remove the two children, add the parent
            pool = [x for x in pool if x not in leaf_nodes]
            pool.append(node_index)

            edge_index += 2
            node_index += 1
            k -= 1

        self.height = float(np.max(self.node_height))
        self.length = float(np.sum(self.edge[:, 2]))

    def to_dict(self) -> dict:
        """Returns the tree as a dictionary."""
        return {
            "edge": self.edge,
            "node_height": self.node_height,
            "height": self.height,
            "length": self.length,
            "n": self.n,
        }
