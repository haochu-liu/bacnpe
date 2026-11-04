"""
Pairwise linkage statistics: LD (D, D', r^2), the four-gamete test,
Kelly's Z_nS and Hudson's R_m.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

_EPS = np.finfo(float).eps

# Pairs are processed in blocks to bound memory use for long sequences.
_PAIR_BLOCK = 1 << 16


def _segregating_sites(mat: np.ndarray) -> np.ndarray:
    """
    Indices of the segregating sites of an incidence matrix.

    Parameters
    ----------
    mat : np.ndarray
        Boolean incidence matrix (rows are sequences, columns are sites).

    Returns
    -------
    np.ndarray
        Column indices where both states are present.
    """
    has_true = mat.any(axis=0)
    has_false = ~mat.all(axis=0)
    return np.where(has_true & has_false)[0]


def _pairwise_ld_g4(mat: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray) -> Dict[str, np.ndarray]:
    """
    LD statistics and four-gamete test for many pairs of columns.

    Parameters
    ----------
    mat : np.ndarray
        Boolean incidence matrix (rows are sequences, columns are sites).
    idx_i, idx_j : np.ndarray
        Column indices of the first and second site of every pair.

    Returns
    -------
    dict
        Arrays (one value per pair) with the following keys:
        - 'D': The linkage disequilibrium coefficient.
        - 'D_prime': The normalized linkage disequilibrium coefficient.
        - 'r_square': The square of Pearson's correlation coefficient for LD.
        - 'g4': True if all four gametes (00, 01, 10, 11) are present.
    """
    n = mat.shape[0]
    counts = np.count_nonzero(mat, axis=0)  # number of 1s per column
    n_pairs = idx_i.size

    # Number of sequences carrying 1 at both sites (count of gamete 11)
    n11 = np.empty(n_pairs, dtype=np.int64)
    for start in range(0, n_pairs, _PAIR_BLOCK):
        stop = start + _PAIR_BLOCK
        n11[start:stop] = np.count_nonzero(mat[:, idx_i[start:stop]] & mat[:, idx_j[start:stop]], axis=0)

    # Allele frequencies
    p_A = counts[idx_i] / n
    p_a = 1 - p_A
    p_B = counts[idx_j] / n
    p_b = 1 - p_B
    p_AB = n11 / n

    D_AB = p_AB - (p_A * p_B)
    r_square = D_AB ** 2 / (p_A * p_a * p_B * p_b + _EPS)

    # D' = D / D_max (D > 0) or D / D_min (D <= 0)
    D_prime = np.where(
        D_AB > 0,
        D_AB / (np.minimum(p_A * p_b, p_a * p_B) + _EPS),
        D_AB / (-np.minimum(p_A * p_B, p_a * p_b) + _EPS),
    )

    # Four-gamete test: counts of the patterns 10, 01 and 00
    n10 = counts[idx_i] - n11
    n01 = counts[idx_j] - n11
    n00 = n - n11 - n10 - n01
    g4 = (n00 > 0) & (n01 > 0) & (n10 > 0) & (n11 > 0)

    return {"D": D_AB, "D_prime": D_prime, "r_square": r_square, "g4": g4}


def _ld(mat: np.ndarray) -> Dict[str, float]:
    """
    The square of correlation coefficient for LD.

    Compute the square of Pearson's correlation coefficient for linkage disequilibrium.

    Parameters
    ----------
    mat : np.ndarray
        An incidence matrix with two columns.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'D': The linkage disequilibrium coefficient.
        - 'D_prime': The normalized linkage disequilibrium coefficient.
        - 'r_square': The square of Pearson's correlation coefficient for LD.
    """
    mat = np.asarray(mat, dtype=bool)
    if mat.ndim != 2 or mat.shape[1] != 2:
        raise ValueError("`mat` must have two columns.")
    res = _pairwise_ld_g4(mat, np.array([0]), np.array([1]))
    return {key: float(res[key][0]) for key in ("D", "D_prime", "r_square")}


def _g4_test(mat: np.ndarray) -> bool:
    """
    Four-gamete test.

    Check whether the two sites show all four patterns 00, 01, 10, 11.

    Parameters
    ----------
    mat : np.ndarray
        An incidence matrix with two columns.

    Returns
    -------
    bool
        True if all four patterns (00, 01, 10, 11) are present, False otherwise.
    """
    mat = np.asarray(mat, dtype=bool)
    if mat.ndim != 2 or mat.shape[1] != 2:
        raise ValueError("`mat` must have two columns.")
    if mat.shape[0] < 3:
        raise ValueError("`mat` must have at least 3 rows.")
    return bool(_pairwise_ld_g4(mat, np.array([0]), np.array([1]))["g4"][0])


def _hudson_rm_from_intervals(intervals: Iterable[Tuple[int, int]]) -> int:
    """
    Hudson & Kaplan's R_m from a list of incompatible site intervals.

    Greedy algorithm: sort the intervals by their right end and count the
    maximum number of non-overlapping intervals.

    Parameters
    ----------
    intervals : iterable of (int, int)
        Pairs ``(i, j)`` of site positions (``i < j``) failing the
        four-gamete test.

    Returns
    -------
    int
        Hudson's Rm statistic.
    """
    intervals: List[Tuple[int, int]] = sorted(intervals, key=lambda x: x[1])

    rm_count = 0
    last_cut = -1

    for start, end in intervals:
        if start > last_cut:
            last_cut = end - 1
            rm_count += 1

    return rm_count


def _hudson_rm(mat: np.ndarray) -> int:
    """
    Hudson's Rm statistic.

    Compute Hudson's Rm statistic for a given incidence matrix.

    Parameters
    ----------
    mat : np.ndarray
        An incidence matrix where each row represents a sequence.

    Returns
    -------
    int
        Hudson's Rm statistic.
    """
    mat = np.asarray(mat, dtype=bool)
    # Only segregating sites can fail the four-gamete test
    idx_seg = _segregating_sites(mat)
    if idx_seg.size < 2:
        return 0
    pos_i, pos_j = np.triu_indices(idx_seg.size, k=1)
    g4 = _pairwise_ld_g4(mat, idx_seg[pos_i], idx_seg[pos_j])["g4"]
    return _hudson_rm_from_intervals(zip(idx_seg[pos_i][g4].tolist(), idx_seg[pos_j][g4].tolist()))


def _kelly_z(mat: np.ndarray) -> float:
    """
    Kelly's Z estimator.

    Compute Kelly's Z_nS (mean r^2 over all pairs of segregating sites)
    for a given incidence matrix.

    Parameters
    ----------
    mat : np.ndarray
        An incidence matrix where each row represents a sequence.

    Returns
    -------
    float
        Kelly's Z estimator (0.0 when there are fewer than 2 segregating sites).
    """
    mat = np.asarray(mat, dtype=bool)
    idx_seg = _segregating_sites(mat)
    if idx_seg.size < 2:
        return 0.0
    pos_i, pos_j = np.triu_indices(idx_seg.size, k=1)
    r_square = _pairwise_ld_g4(mat, idx_seg[pos_i], idx_seg[pos_j])["r_square"]
    return float(np.mean(r_square))
