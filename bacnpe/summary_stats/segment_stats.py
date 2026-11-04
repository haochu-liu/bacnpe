"""The 30-dimensional summary statistic vector used for NPE."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .._validation import _check_binary_matrix
from ..tree.clonal_genealogy import ClonalTree
from .diversity import _tajima_d, _tajima_pi, _watterson_theta
from .homoplasy import _homoplasy_index
from .linkage import _hudson_rm_from_intervals, _pairwise_ld_g4, _segregating_sites
from .regression import _exp_regression
from .wall import _wall_bq

#: Names of the 30 entries returned by :func:`segment_summary_stats`
#: ("near" = pairs of segregating sites closer than L/2, "far" = the others).
STAT_NAMES: List[str] = [
    "D_near_sum", "D_far_sum",                  # 0, 1   sum of |D|
    "D_prime_near_sum", "D_prime_far_sum",      # 2, 3   sum of D'
    "r2_near_sum", "r2_far_sum",                # 4, 5   sum of r^2
    "g4_near_sum", "g4_far_sum",                # 6, 7   number of pairs failing the 4-gamete test
    "D_near_mean", "D_far_mean",                # 8, 9
    "D_prime_near_mean", "D_prime_far_mean",    # 10, 11
    "r2_near_mean", "r2_far_mean",              # 12, 13
    "g4_near_mean", "g4_far_mean",              # 14, 15
    "homoplasy_index",                          # 16
    "prop_segregating_sites",                   # 17
    "watterson_theta",                          # 18
    "tajima_pi",                                # 19
    "wakeley_pi2",                              # 20
    "tajima_D",                                 # 21
    "wall_B", "wall_Q",                         # 22, 23
    "hudson_Rm",                                # 24
    "kelly_ZnS",                                # 25
    "r2_decay_a", "r2_decay_b", "r2_decay_c",   # 26-28  r^2 = a*exp(-b*d) + c
    "L",                                        # 29
]


def _sequential_sum(values: np.ndarray) -> float:
    """
    Sum the values strictly from left to right.

    ``np.sum`` uses pairwise summation; ``np.cumsum`` adds sequentially, which
    reproduces a Python ``total += value`` loop bit for bit.

    Parameters
    ----------
    values : np.ndarray
        1-D array of floats.

    Returns
    -------
    float
        The left-to-right sum (0.0 for an empty array).
    """
    if values.size == 0:
        return 0.0
    return float(np.cumsum(values)[-1])


def segment_summary_stats(tree: ClonalTree, seg_mat: np.ndarray) -> np.ndarray:
    """
    Compute summary statistics for a given clonal tree and genetic matrix.

    Parameters
    ----------
    tree : ClonalTree
        The clonal genealogy.
    seg_mat : np.ndarray
        The genetic matrix of shape (tree.n, L): 0/1 or boolean, rows are
        sequences (leaves of the tree) and columns are sites.

    Returns
    -------
    np.ndarray
        A 30-dimensional float64 vector as the summary statistics of
        simulations (see ``STAT_NAMES`` for the meaning of each entry).
    """
    if not isinstance(tree, ClonalTree):
        raise TypeError("`tree` must be of class 'ClonalTree'.")
    seg_mat = _check_binary_matrix(seg_mat, "seg_mat")
    tree_width = tree.n
    if tree_width != seg_mat.shape[0]:
        raise ValueError("Inconsistent number of sequences between tree and genetic matrix!")
    if tree_width < 3:
        raise ValueError("At least 3 sequences are needed to compute the summary statistics.")
    L = seg_mat.shape[1]
    if L < 1:
        raise ValueError("`seg_mat` must have at least one site.")

    s_vec = np.full(30, np.nan, dtype=float)

    # Identify segregating sites
    idx_seg = _segregating_sites(seg_mat)
    count_S = idx_seg.size

    # Summary statistics LD r and G4 test, over all pairs of segregating sites
    r_squares = np.array([], dtype=float)
    distances = np.array([], dtype=np.int64)
    incompatible_intervals = []

    if count_S >= 2:
        # All pairs (i < j) in the same order as a double loop over i and j
        pos_i, pos_j = np.triu_indices(count_S, k=1)
        site_i, site_j = idx_seg[pos_i], idx_seg[pos_j]
        distances = site_j - site_i

        # Compute LD statistics and G4 test
        pair_stats = _pairwise_ld_g4(seg_mat, site_i, site_j)
        r_squares = pair_stats["r_square"]
        g4 = pair_stats["g4"]
        incompatible_intervals = list(zip(site_i[g4].tolist(), site_j[g4].tolist()))

        near = distances < L / 2
        for offset, mask in ((0, near), (1, ~near)):
            n_pairs = int(np.count_nonzero(mask))
            D_sum = _sequential_sum(np.abs(pair_stats["D"][mask]))
            D_prime_sum = _sequential_sum(pair_stats["D_prime"][mask])
            r2_sum = _sequential_sum(r_squares[mask])
            g4_sum = int(np.count_nonzero(g4[mask]))

            s_vec[0 + offset] = D_sum
            s_vec[2 + offset] = D_prime_sum
            s_vec[4 + offset] = r2_sum
            s_vec[6 + offset] = g4_sum

            s_vec[8 + offset] = D_sum / n_pairs if n_pairs > 0 else 0
            s_vec[10 + offset] = D_prime_sum / n_pairs if n_pairs > 0 else 0
            s_vec[12 + offset] = r2_sum / n_pairs if n_pairs > 0 else 0
            s_vec[14 + offset] = g4_sum / n_pairs if n_pairs > 0 else 0

        # Kelly's Zns estimator
        s_vec[25] = np.mean(r_squares)
    else:
        s_vec[:16] = 0
        s_vec[25] = 0  # Kelly's Zns estimator

    # Summary statistic homoplasy index
    s_vec[16] = _homoplasy_index(tree, seg_mat)

    # Summary statistic proportion of segregating sites
    s_vec[17] = count_S / L

    # Watterson's theta estimator
    s_vec[18] = _watterson_theta(tree_width, count_S)

    # Tajima's pi estimators
    tajima_dict = _tajima_pi(seg_mat, Wakeley=True)
    s_vec[19] = tajima_dict["pi"]
    s_vec[20] = tajima_dict["pi2"]

    # Tajima's D statistic (NaN when there is no segregating site)
    s_vec[21] = _tajima_d(tree_width, s_vec[19], s_vec[18], count_S)

    # Wall's B and Q statistics
    wall_dict = _wall_bq(seg_mat[:, idx_seg])
    s_vec[22] = wall_dict["B"]
    s_vec[23] = wall_dict["Q"]

    # Hudson's Rm estimator
    s_vec[24] = _hudson_rm_from_intervals(incompatible_intervals)

    # Regression coefficient of r^2 on distance
    if len(r_squares) >= 4:  # Ensure enough data points for regression
        df = pd.DataFrame({"x": distances, "y": r_squares})
        mean_df = df.groupby("x")["y"].mean().reset_index()
        try:
            coeff = _exp_regression(np.array(mean_df["x"]), np.array(mean_df["y"]))
            s_vec[26:29] = coeff
        except RuntimeError:
            s_vec[26:29] = 0
    else:
        s_vec[26:29] = 0

    # Add the length of sequence as a summary statistic
    s_vec[29] = L

    return s_vec
