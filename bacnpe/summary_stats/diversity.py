"""
Nucleotide diversity statistics: Watterson's theta, Tajima's pi
(with Wakeley's variance pi^2) and Tajima's D.

This module merges the original ``Watterson_theta.py``, ``Tajima_pi.py``
and ``Tajima_D.py``.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _harmonic_numbers(n: int):
    """
    Harmonic sums used by Watterson's theta and Tajima's D.

    Parameters
    ----------
    n : int
        The number of sequences.

    Returns
    -------
    tuple of float
        ``a_n = sum_{i=1}^{n-1} 1/i`` and ``b_n = sum_{i=1}^{n-1} 1/i^2``.
    """
    an = np.sum(1 / np.arange(1, n))
    bn = np.sum(1 / np.arange(1, n) ** 2)
    return an, bn


def _watterson_theta(n: int, S: int) -> float:
    """
    Watterson's theta estimator.

    Compute Watterson's theta estimator ``S / a_n``.

    Parameters
    ----------
    n : int
        The number of sequences (rows of the incidence matrix).
    S : int
        The number of segregating sites.

    Returns
    -------
    float
        Watterson's theta estimator.
    """
    a_n, _ = _harmonic_numbers(n)
    return S / a_n


def _pairwise_differences(mat: np.ndarray) -> np.ndarray:
    """
    Number of differing sites for every pair of sequences.

    Parameters
    ----------
    mat : np.ndarray
        Boolean incidence matrix (rows are sequences).

    Returns
    -------
    np.ndarray, dtype int64
        Differences for the pairs (i, j), i < j, in row-major order
        (i.e. the order (0,1), (0,2), ..., (1,2), ...).
    """
    n = mat.shape[0]
    x = mat.astype(np.float64)
    counts = x.sum(axis=1)
    # Hamming distance of binary vectors: |a| + |b| - 2 <a, b>
    # (exact: all values are integers far below 2**53)
    dist = np.rint(counts[:, None] + counts[None, :] - 2.0 * (x @ x.T)).astype(np.int64)
    i_idx, j_idx = np.triu_indices(n, k=1)
    return dist[i_idx, j_idx]


def _tajima_pi(mat: np.ndarray, Wakeley: bool = False) -> Dict[str, float]:
    """
    Tajima's pi estimator.

    Compute Tajima's pi estimator (mean number of pairwise differences)
    for a given incidence matrix. And optionally, compute Wakeley's pi squared
    estimator (variance of the pairwise differences) if `Wakeley` is set to True.

    Parameters
    ----------
    mat : np.ndarray
        An incidence matrix where each row represents a sequence.
    Wakeley : bool, optional
        Whether to compute Wakeley's pi squared estimator, by default False

    Returns
    -------
    dict
        Tajima's pi estimator ('pi') and optionally Wakeley's pi squared
        estimator ('pi2').
    """
    n = mat.shape[0]
    pi_list = _pairwise_differences(mat)

    pi = 2 * np.sum(pi_list) / (n * (n - 1))
    result = {"pi": pi}

    if Wakeley:
        pi2 = 2 * np.sum((pi_list - pi) ** 2) / (n * (n - 1))
        result["pi2"] = pi2

    return result


def _tajima_d(n: int, pi_hat: float, theta_hat: float, S: int) -> float:
    """
    Tajima's D estimator.

    Compute Tajima's D statistic from pi and Watterson's theta.

    Parameters
    ----------
    n : int
        The number of sequences (rows of the incidence matrix).
    pi_hat : float
        Tajima's pi estimator.
    theta_hat : float
        Watterson's theta estimator.
    S : int
        The number of segregating sites.

    Returns
    -------
    float
        Tajima's D estimator (NaN when S == 0, since the variance is 0).
    """
    an, bn = _harmonic_numbers(n)

    e1 = (n + 1) / (3 * an * (n - 1)) - 1 / an**2
    e2 = (2 * (n**2 + n + 3) / (9 * n * (n - 1)) - (n + 2) / (n * an) + bn / an**2) / (an**2 + bn)

    with np.errstate(divide="ignore", invalid="ignore"):
        D = (pi_hat - theta_hat) / np.sqrt(e1 * S + e2 * S * (S - 1))
    return D
