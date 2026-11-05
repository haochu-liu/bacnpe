"""Clonal genealogy: simulated by the coalescent, or loaded from an estimate."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .._validation import _check_integer
from .base import _Tree
from .newick import _read_source, format_newick, parse_newick
from .upgma import upgma_clonal_tree

#: File names used by :meth:`ClonalTree.to_csv` and :meth:`ClonalTree.from_csv`
#: (the names written by the old R scripts).
EDGE_FILE = "clonal_edge.csv"
NODE_HEIGHT_FILE = "clonal_node_height.csv"
TIP_NAMES_FILE = "tip_names.csv"


class ClonalTree(_Tree):
    """
    Clonal genealogy.

    ``ClonalTree(n)`` simulates a tree with `n` leaves under the coalescent.
    A tree estimated from data is loaded with one of the class methods
    instead:

    * :meth:`from_newick` - an estimated phylogeny, e.g. from ClonalFrameML,
      converted to a clonal tree by UPGMA;
    * :meth:`from_csv` - a tree saved by :meth:`to_csv` (or by the old R
      scripts);
    * :meth:`from_arrays` - edge and node-height arrays.

    Node numbering (1-indexed, as in the original R implementation):
    leaves are ``1..n`` and internal nodes are ``n+1..2n-1`` in order of
    increasing height, so node ``2n-1`` is the root.

    Attributes
    ----------
    n : int
        The number of leaf lineages.
    edge : np.ndarray, shape (2*(n-1), 3), dtype float64
        One row per edge: ``[parent node, child node, edge length]``,
        sorted by parent (two rows per internal node).
    node_height : np.ndarray, shape (2*n-1,), dtype float64
        Node heights to recent time (leaves are at height 0).
    height : float
        Height of the tree (time to the most recent common ancestor).
    length : float
        Total branch length of the tree.
    tip_labels : list of str or None
        Names of leaves ``1..n`` (None for simulated trees).
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
        self.tip_labels: Optional[list] = None
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

    # ------------------------------------------------------------------
    # Loading an estimated tree
    # ------------------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        edge: np.ndarray,
        node_height: np.ndarray,
        tip_labels: Optional[Sequence[str]] = None,
    ) -> "ClonalTree":
        """
        Build a clonal tree from its edge and node-height arrays.

        Parameters
        ----------
        edge : array-like, shape (2*(n-1), 3)
            ``[parent node, child node, edge length]`` rows, nodes numbered
            as described in the class docstring. Rows are sorted by parent
            if they are not already.
        node_height : array-like, shape (2*n-1,)
            Node heights; leaves ``1..n`` at 0, internal nodes ``n+1..2n-1``
            non-decreasing.
        tip_labels : sequence of str, optional
            Names of leaves ``1..n``.

        Raises
        ------
        ValueError
            If the arrays do not describe such a binary tree, or the edge
            lengths differ from the height differences.
        """
        edge = np.array(edge, dtype=np.float64)
        node_height = np.array(node_height, dtype=np.float64)
        if edge.size == 0:
            edge = edge.reshape(0, 3)
        if node_height.ndim != 1 or node_height.size % 2 == 0:
            raise ValueError("`node_height` must be a 1-D array of length 2n-1.")
        n = (node_height.size + 1) // 2
        if edge.ndim != 2 or edge.shape != (2 * (n - 1), 3):
            raise ValueError(
                f"`edge` must have shape (2(n-1), 3) = ({2 * (n - 1)}, 3) for "
                f"{node_height.size} node heights, got {edge.shape}."
            )
        if not (np.all(np.isfinite(edge)) and np.all(np.isfinite(node_height))):
            raise ValueError("`edge` and `node_height` must be finite.")
        ids = edge[:, :2]
        if np.any(ids != np.round(ids)):
            raise ValueError("Node ids in the first two columns of `edge` must be integers.")
        parents, children = ids[:, 0].astype(np.int64), ids[:, 1].astype(np.int64)
        if sorted(children.tolist()) != list(range(1, 2 * n - 1)):
            raise ValueError(f"Every node except the root {2 * n - 1} must be the child of exactly one edge.")
        if sorted(parents.tolist()) != [p for p in range(n + 1, 2 * n) for _ in range(2)]:
            raise ValueError(f"Every internal node {n + 1}..{2 * n - 1} must have exactly two children.")
        if np.any(node_height[:n] != 0):
            raise ValueError("Leaves (nodes 1..n) must have height 0.")
        if n > 1 and (node_height[n] < 0 or np.any(np.diff(node_height[n:]) < 0)):
            raise ValueError("Internal nodes n+1..2n-1 must be numbered by non-decreasing height.")
        expected = node_height[parents - 1] - node_height[children - 1]
        tol = 1e-9 * max(1.0, float(node_height.max()))
        if not np.allclose(edge[:, 2], expected, rtol=1e-6, atol=tol):
            raise ValueError("Edge lengths must equal parent height minus child height.")

        tree = cls.__new__(cls)
        _Tree.__init__(tree, n)
        tree.edge = edge[np.argsort(parents, kind="stable")]
        tree.node_height = node_height
        tree.height = float(np.max(node_height))
        tree.length = float(np.sum(edge[:, 2]))
        tree.tip_labels = cls._check_tip_labels(tip_labels, n)
        return tree

    @classmethod
    def from_newick(
        cls,
        source: Union[str, os.PathLike],
        root_height: Union[str, float, None] = "coalescent",
        r_compatible: bool = False,
    ) -> "ClonalTree":
        """
        Load an estimated phylogeny (e.g. from ClonalFrameML) as a clonal tree.

        The tree is made ultrametric by UPGMA on its patristic distances and
        rescaled to coalescent units, as the R scripts
        ``likelihood-free/R/simARG/*_clonal_tree.R`` do with ``ape`` and
        ``phangorn``. Leaves keep their order in the Newick file and their
        names are stored in `tip_labels`.

        Parameters
        ----------
        source : str or path-like
            Path to a Newick file (e.g. ClonalFrameML's
            ``*.labelled_tree.newick``), or a Newick string.
        root_height : "coalescent", float or None, default "coalescent"
            Root height after rescaling. ``"coalescent"`` uses the expected
            root height of the coalescent, ``2 * (1 - 1/n)``; a positive
            number uses that height; None keeps the UPGMA heights (half the
            patristic distances, in the units of the input branch lengths).
        r_compatible : bool, default False
            Reproduce the R scripts exactly. They rescale so that the common
            ancestor of the first and last tip, not necessarily the root, has
            height `root_height`, which made the klebsiella and staph roots
            taller than ``2 * (1 - 1/n)``.

        Returns
        -------
        ClonalTree
        """
        if not (root_height is None or root_height == "coalescent"):
            if isinstance(root_height, (str, bool)) or not isinstance(root_height, (int, float, np.number)):
                raise TypeError('`root_height` must be "coalescent", a positive number or None.')
            if not (np.isfinite(root_height) and root_height > 0):
                raise ValueError("`root_height` must be positive and finite.")
        parsed = parse_newick(_read_source(source))
        labels = [parsed.label[i] for i in parsed.tip_indices]
        if any(lab is None or lab == "" for lab in labels):
            raise ValueError("Every tip of the Newick tree needs a name.")
        edge, node_height = upgma_clonal_tree(parsed, root_height, r_compatible=r_compatible)
        return cls.from_arrays(edge, node_height, tip_labels=labels)

    @classmethod
    def from_csv(cls, directory: Union[str, os.PathLike]) -> "ClonalTree":
        """
        Load a tree saved by :meth:`to_csv` (or by the old R scripts).

        Reads ``clonal_edge.csv`` and ``clonal_node_height.csv`` from
        `directory`, and ``tip_names.csv`` if it exists.
        """
        directory = Path(directory)
        edge = np.loadtxt(directory / EDGE_FILE, delimiter=",", dtype=float, ndmin=2)
        node_height = np.loadtxt(directory / NODE_HEIGHT_FILE, delimiter=",", dtype=float, ndmin=1)
        tip_file = directory / TIP_NAMES_FILE
        tip_labels = None
        if tip_file.exists():
            with open(tip_file, newline="") as fh:
                tip_labels = [row[0] for row in csv.reader(fh) if row]
        return cls.from_arrays(edge, node_height, tip_labels=tip_labels)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def to_csv(self, directory: Union[str, os.PathLike], overwrite: bool = False) -> Path:
        """
        Save the tree as ``clonal_edge.csv``, ``clonal_node_height.csv`` and
        (if the tree has tip labels) ``tip_names.csv`` in `directory`, the
        layout of the old R scripts. Lengths are written at full precision.

        Raises
        ------
        FileExistsError
            If one of the files exists and `overwrite` is False.
        """
        directory = Path(directory)
        files = [directory / EDGE_FILE, directory / NODE_HEIGHT_FILE]
        if self.tip_labels is not None:
            files.append(directory / TIP_NAMES_FILE)
        existing = [str(f) for f in files if f.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"{existing} already exist (pass overwrite=True to replace them).")
        directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(files[0], self.edge, delimiter=",", fmt=["%d", "%d", "%.17g"])
        np.savetxt(files[1], self.node_height, fmt="%.17g")
        if self.tip_labels is not None:
            with open(files[2], "w", newline="") as fh:
                csv.writer(fh, quoting=csv.QUOTE_ALL).writerows([lab] for lab in self.tip_labels)
        return directory

    def to_newick(self) -> str:
        """The tree as a Newick string (leaf numbers if there are no tip labels)."""
        return format_newick(self.edge, self.n, self.tip_labels)

    def to_dict(self) -> dict:
        """Returns the tree as a dictionary."""
        return {
            "edge": self.edge,
            "node_height": self.node_height,
            "height": self.height,
            "length": self.length,
            "n": self.n,
            "tip_labels": self.tip_labels,
        }

    @staticmethod
    def _check_tip_labels(tip_labels, n: int) -> Optional[list]:
        if tip_labels is None:
            return None
        if isinstance(tip_labels, str):
            raise TypeError("`tip_labels` must be a sequence of strings, not a string.")
        labels = list(tip_labels)
        if len(labels) != n:
            raise ValueError(f"`tip_labels` has {len(labels)} names for {n} leaves.")
        if not all(isinstance(lab, str) for lab in labels):
            raise TypeError("`tip_labels` must contain strings.")
        if len(set(labels)) != n:
            raise ValueError("`tip_labels` must be unique.")
        return labels
