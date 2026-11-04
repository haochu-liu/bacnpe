"""Public simulator: ClonalOrigin ARG + mutations + summary statistics."""
from __future__ import annotations

import numpy as np

from .._validation import _check_integer, _check_real
from ..summary_stats.segment_stats import segment_summary_stats
from ..tree.clonal_genealogy import ClonalTree
from .arg import ARG
from .mutation import add_mutation


def ClonalOrigin_seq_sim(tree: ClonalTree, rho_site: float, theta_site: float,
                         L: int, delta: float) -> np.ndarray:
    """
    Simulate approximated ARG using ClonalOrigin with sequential method.

    Simulate a sequence of `L` sites in the ClonalOrigin approximation from a
    given clonal tree, add mutations, and compute summary statistics of the
    resulting sequences. LD statistics are reported separately for pairs of
    segregating sites that are "near" (distance < L/2) and "far";
    Kelly's Zns and Hudson's Rm computation are integrated.

    Randomness comes from NumPy's global random state, so call
    ``np.random.seed(...)`` beforehand for reproducible results.

    Parameters
    ----------
    tree : ClonalTree
        The clonal genealogy (at least 3 leaves).
    rho_site : float
        The recombination parameter per site (>= 0).
    theta_site : float
        The mutation rate per site (>= 0).
    L : int
        The number of sites (>= 1).
    delta : float
        The mean of recombinant segment length (>= 1).

    Returns
    -------
    np.ndarray
        A 30-dimensional float64 vector as the summary statistics of
        simulations. Entry names are listed in
        ``bacnpe.summary_stats.STAT_NAMES``:

        ======  ===================================================
        0-7     sums of |D|, D', r^2 and four-gamete failures
                (near, far alternating)
        8-15    the same quantities averaged over pairs
        16      homoplasy index on the clonal tree
        17      proportion of segregating sites
        18      Watterson's theta
        19, 20  Tajima's pi and Wakeley's pi^2
        21      Tajima's D (NaN if there is no segregating site)
        22, 23  Wall's B and Q
        24      Hudson's Rm
        25      Kelly's ZnS
        26-28   a, b, c of the fit r^2 = a * exp(-b * d) + c
        29      L
        ======  ===================================================

    Raises
    ------
    TypeError
        If `tree` is not a ClonalTree, `L` is not an integer, or a rate is
        not a real number.
    ValueError
        If a parameter is out of range (see above), or not finite.
    """
    # Input validation
    if not isinstance(tree, ClonalTree):
        raise TypeError("`tree` must be of class 'ClonalTree'.")
    if tree.n < 3:
        raise ValueError("`tree` must have at least 3 leaves.")
    rho_site = _check_real(rho_site, "rho_site", minimum=0.0)
    theta_site = _check_real(theta_site, "theta_site", minimum=0.0)
    L = _check_integer(L, "L", minimum=1)
    delta = _check_real(delta, "delta", minimum=1.0)

    tree_width = tree.n

    # Simulate ARG and add mutations
    ARG_sim = ARG(tree, rho_site, L, delta)
    node_site = add_mutation(ARG_sim, theta_site)
    # Polarise the leaf sequences relative to the first sequence
    mat = node_site[:tree_width, :] != node_site[0]

    # Compute summary statistics
    s_vec = segment_summary_stats(tree, mat)

    return s_vec
