"""Wall's B and Q statistics."""
from __future__ import annotations

from typing import Dict

import numpy as np


def _wall_bq(seg_mat: np.ndarray) -> Dict[str, float]:
    """
    Wall's BQ estimator.

    Compute Wall's B and Q estimators for a given incidence matrix.
    B is the proportion of adjacent segregating-site pairs that are congruent
    (identical bipartition of the sequences); Q additionally counts the number
    of distinct congruent bipartitions.

    Parameters
    ----------
    seg_mat : np.ndarray
        An incidence matrix where each row represents a sequence.
        And each column represents a segregating site.

    Returns
    -------
    dict
        Wall's B and Q estimators in a dictionary.
    """
    seg_mat = np.asarray(seg_mat, dtype=bool)
    S = seg_mat.shape[1]

    if S < 2:
        return {"B": 0.0, "Q": 0.0}

    # Polarise every column relative to the first sequence, so that a column
    # and its complement represent the same bipartition
    polarized = seg_mat ^ seg_mat[0]

    # Adjacent sites with the same bipartition are congruent
    congruent = np.all(polarized[:, :-1] == polarized[:, 1:], axis=0)
    B = int(np.count_nonzero(congruent))
    # Number of distinct bipartitions among the congruent pairs
    A = np.unique(polarized[:, :-1][:, congruent], axis=1).shape[1] if B > 0 else 0

    B_norm = B / (S - 1)
    Q = (B + A) / S
    return {"B": B_norm, "Q": Q}
