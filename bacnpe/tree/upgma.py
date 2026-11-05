"""
Converting an estimated phylogeny into a clonal tree (UPGMA).

Python port of ``likelihood-free/R/simARG/*_clonal_tree.R``::

    t  <- read.tree("cfml.nwk")
    t2 <- phangorn::upgma(ape::cophenetic.phylo(t))
    # rescale so the root height equals the coalescent expectation 2(1 - 1/n),
    # number internal nodes by height and sort the edges by parent

Each step copies the arithmetic of the R functions it replaces
(``ape::dist.nodes``, ``stats::hclust`` with ``method = "average"``,
``ape::as.phylo.hclust``), so ties are broken the same way and the tree
topology and node numbering match the R output exactly.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .newick import _ParsedTree

_INF = 1e300  # the "infinity" used by R's hclust.f


def cophenetic(tree: _ParsedTree) -> np.ndarray:
    """
    Patristic distances between tips, as ``ape::cophenetic.phylo``.

    Uses the recurrence of ``ape::dist.nodes``: nodes are visited in
    preorder, and the distance from a new node to every node visited before
    is the distance from its parent plus its branch length.
    """
    m = len(tree.parent)
    if m > 1 and np.any(np.isnan(tree.length[1:])):
        raise ValueError("Every branch of the Newick tree needs a length.")
    if m > 1 and np.any(tree.length[1:] < 0):
        raise ValueError("The Newick tree has negative branch lengths.")
    D = np.zeros((m, m))
    for d in range(1, m):
        a = tree.parent[d]
        D[d, :d] = D[a, :d] + tree.length[d]
        D[:d, d] = D[d, :d]
    tips = tree.tip_indices
    return D[np.ix_(tips, tips)]


def hclust_average(diss: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average-linkage (UPGMA) clustering, a port of R's ``hclust.f``.

    Parameters
    ----------
    diss : np.ndarray, shape (n, n)
        Symmetric dissimilarity matrix.

    Returns
    -------
    merge : np.ndarray, shape (n-1, 2), int
        R's ``hclust()$merge``: negative entries are singletons ``-j``,
        positive entries are earlier merges (1-indexed).
    height : np.ndarray, shape (n-1,)
        R's ``hclust()$height``.
    """
    D = np.array(diss, dtype=np.float64)
    n = D.shape[0]
    flag = np.ones(n, dtype=bool)
    membr = np.ones(n)
    nn = np.zeros(n, dtype=np.int64)
    disnn = np.full(n, _INF)
    # Nearest neighbour of each i among j > i (first minimum wins, as in R)
    for i in range(n - 1):
        j = i + 1 + int(np.argmin(D[i, i + 1:]))
        nn[i], disnn[i] = j, D[i, j]

    ia = np.zeros(n - 1, dtype=np.int64)
    ib = np.zeros(n - 1, dtype=np.int64)
    crit = np.zeros(n - 1)
    idx = np.arange(n)
    jj = 0
    for step in range(n - 1):
        # Least dissimilarity in the list of nearest neighbours
        cand = np.where(flag[:n - 1], disnn[:n - 1], _INF)
        im = int(np.argmin(cand))
        dmin, jm = cand[im], int(nn[im])
        i2, j2 = min(im, jm), max(im, jm)
        ia[step], ib[step], crit[step] = i2, j2, dmin
        flag[j2] = False

        # Update dissimilarities from the new cluster (group average)
        ks = idx[flag & (idx != i2)]
        new = (membr[i2] * D[i2, ks] + membr[j2] * D[j2, ks]) / (membr[i2] + membr[j2])
        D[i2, ks] = new
        D[ks, i2] = new
        right = ks > i2
        dmin = _INF
        if np.any(right):
            r = int(np.argmin(new[right]))
            if new[right][r] < _INF:
                dmin, jj = new[right][r], int(ks[right][r])
        left = ~right
        better = new[left] < disnn[ks[left]]
        disnn[ks[left][better]] = new[left][better]
        nn[ks[left][better]] = i2
        membr[i2] += membr[j2]
        disnn[i2], nn[i2] = dmin, jj

        # Redetermine the nearest neighbours that pointed to i2 or j2
        for i in np.flatnonzero(flag[:n - 1] & ((nn[:n - 1] == i2) | (nn[:n - 1] == j2))):
            later = idx[i + 1:][flag[i + 1:]]
            if later.size:
                r = int(np.argmin(D[i, later]))
                if D[i, later[r]] < _INF:
                    nn[i], disnn[i] = later[r], D[i, later[r]]
                    continue
            disnn[i] = _INF

    # HCASS2: clusters are represented by their lowest member; convert to the
    # signed merge format of R
    ia1, ib1 = ia + 1, ib + 1
    iia, iib = ia1.copy(), ib1.copy()
    for i in range(n - 2):
        k = min(ia1[i], ib1[i])
        later_a = np.flatnonzero(ia1[i + 1:] == k) + i + 1
        later_b = np.flatnonzero(ib1[i + 1:] == k) + i + 1
        iia[later_a] = -(i + 1)
        iib[later_b] = -(i + 1)
    iia, iib = -iia, -iib
    merge = np.column_stack([iia, iib])
    for i in range(n - 1):
        a, b = merge[i]
        if a > 0 and b < 0:
            merge[i] = b, a
        elif a > 0 and b > 0:
            merge[i] = min(a, b), max(a, b)
    return merge, crit


def upgma_clonal_tree(tree: _ParsedTree, root_height="coalescent",
                      r_compatible: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the clonal-tree arrays of an estimated phylogeny by UPGMA.

    Parameters
    ----------
    tree : _ParsedTree
        The parsed Newick tree.
    root_height : "coalescent", float or None
        Height of the root after rescaling: ``"coalescent"`` uses the
        expected root height of the coalescent, ``2 * (1 - 1/n)``; a number
        uses that height; None keeps the UPGMA heights (half the patristic
        distances, in the units of the input branch lengths).
    r_compatible : bool, default False
        Reproduce the R scripts exactly: they rescale so that the common
        ancestor of tips 1 and n (not necessarily the root) has height
        `root_height`.

    Returns
    -------
    edge : np.ndarray, shape (2(n-1), 3)
        ``[parent, child, length]`` rows sorted by parent. Leaves keep their
        Newick order ``1..n``; internal nodes ``n+1..2n-1`` are numbered by
        increasing height, so ``2n-1`` is the root.
    node_height : np.ndarray, shape (2n-1,)
    """
    D = cophenetic(tree)
    n = D.shape[0]
    if n < 2:
        raise ValueError("The Newick tree needs at least 2 tips.")
    merge, height = hclust_average(D)
    N = n - 1

    # Every length is rescaled as R does: edge.length / e * e2, where e is the
    # UPGMA root height and e2 the requested root height
    e = height[N - 1] / 2
    if r_compatible:
        # The R scripts use e <- dist.nodes(t2)[Ntip(t2), 1] / 2, i.e. the
        # height of the common ancestor of tips 1 and n (the root only when
        # these tips are on opposite sides of it)
        members = []
        for i in range(N):
            m = set()
            for y in merge[i]:
                m |= {-y} if y < 0 else members[y - 1]
            members.append(m)
            if 1 in m and n in m:
                e = height[i] / 2
                break
    if root_height is not None:
        target = 2 * (1 - 1 / n) if isinstance(root_height, str) else float(root_height)
        if e <= 0:
            raise ValueError("All tips are at distance 0; the tree cannot be rescaled.")

    def rescale(x):
        return x if root_height is None else x / e * target

    # Node numbers of ape::as.phylo.hclust (used to break height ties, as R's
    # stable sort does), then renumber the internal nodes by height
    ape_number = np.zeros(N, dtype=np.int64)
    ape_number[N - 1] = n + 1
    cur = n + 2
    for i in range(N - 1, -1, -1):
        for y in merge[i]:
            if y > 0:
                ape_number[y - 1] = cur
                cur += 1
    order = np.lexsort((ape_number, height))           # by height, then ape number
    final_id = np.empty(N, dtype=np.int64)
    final_id[order] = n + 1 + np.arange(N)

    node_height = np.zeros(2 * n - 1)
    node_height[final_id - 1] = rescale(height / 2)
    edge = np.zeros((2 * N, 3))
    for row, i in enumerate(order):
        for l, y in enumerate(merge[i]):
            if y < 0:
                child, length = -y, height[i] / 2
            else:
                child, length = final_id[y - 1], (height[i] - height[y - 1]) / 2
            edge[2 * row + l] = final_id[i], child, rescale(length)
    return edge, node_height
