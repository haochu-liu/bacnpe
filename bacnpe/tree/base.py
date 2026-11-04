"""Base class shared by the clonal tree and the ancestral recombination graph."""
from __future__ import annotations

from typing import Optional

import numpy as np


class _Tree:
    """Base class for representing a phylogenetic tree structure."""

    def __init__(self, n: int):
        """
        Initialize a tree.

        Parameters
        ----------
        n : int
            The number of leaf lineages.
        """
        self.n: int = n
        self.edge: Optional[np.ndarray] = None
        self.node_height: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.n})"


_Tree.__name__ = "Tree"
