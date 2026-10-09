"""Clonal tree data structures and simulation."""

from __future__ import annotations

import numpy as np
from typing import Optional


class _Tree:
    """Internal base class for a phylogenetic tree."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.edge: Optional[np.ndarray] = None
        self.node_height: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n})"


class ClonalTree(_Tree):
    """A random Kingman coalescent genealogy with ``n`` sampled leaves."""

    def __init__(self, n: int) -> None:
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("`n` must be a positive integer.")
        super().__init__(int(n))
        self._simulate()

    def _simulate(self) -> None:
        n = self.n
        self.edge = np.zeros((2 * (n - 1), 3), dtype=float)
        self.node_height = np.zeros(2 * n - 1, dtype=float)

        active_nodes = list(range(1, n + 1))
        elapsed = 0.0
        edge_index = 0
        parent = n + 1

        while len(active_nodes) > 1:
            k = len(active_nodes)
            elapsed += np.random.exponential(scale=2.0 / (k * (k - 1)))
            children = np.random.choice(active_nodes, size=2, replace=False)

            self.edge[edge_index : edge_index + 2, 0] = parent
            self.edge[edge_index : edge_index + 2, 1] = children
            self.edge[edge_index : edge_index + 2, 2] = (
                elapsed - self.node_height[children.astype(int) - 1]
            )
            self.node_height[parent - 1] = elapsed

            chosen = set(children.tolist())
            active_nodes = [node for node in active_nodes if node not in chosen]
            active_nodes.append(parent)
            edge_index += 2
            parent += 1

        self.height = float(np.max(self.node_height))
        self.length = float(np.sum(self.edge[:, 2]))

    def to_dict(self) -> dict[str, object]:
        """Return the simulated tree fields as a dictionary."""
        return {
            "edge": self.edge,
            "node_height": self.node_height,
            "height": self.height,
            "length": self.length,
            "n": self.n,
        }
