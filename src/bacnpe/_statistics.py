"""Internal population-genetic summary statistics."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from ._clonal_tree import ClonalTree


def _g4_test(mat: np.ndarray) -> bool:
    if mat.ndim != 2 or mat.shape[1] != 2:
        raise ValueError("`mat` must be a two-dimensional matrix with two columns.")
    if mat.shape[0] == 0:
        raise ValueError("`mat` must have at least one row.")
    col1, col2 = mat[:, 0].astype(bool), mat[:, 1].astype(bool)
    return bool(
        np.any(~col1 & ~col2)
        and np.any(~col1 & col2)
        and np.any(col1 & ~col2)
        and np.any(col1 & col2)
    )


def _ld(mat: np.ndarray) -> dict[str, float]:
    if mat.ndim != 2 or mat.shape[1] != 2:
        raise ValueError("`mat` must be a two-dimensional matrix with two columns.")
    if mat.shape[0] == 0:
        raise ValueError("`mat` must have at least one row.")
    mat = mat.astype(bool)
    p_a1 = float(np.mean(mat[:, 0]))
    p_b1 = float(np.mean(mat[:, 1]))
    p_a0, p_b0 = 1.0 - p_a1, 1.0 - p_b1
    p_ab = float(np.mean(mat[:, 0] & mat[:, 1]))
    d = p_ab - p_a1 * p_b1
    eps = np.finfo(float).eps
    r_square = d**2 / (p_a1 * p_a0 * p_b1 * p_b0 + eps)
    if d > 0:
        d_prime = d / (min(p_a1 * p_b0, p_a0 * p_b1) + eps)
    else:
        d_prime = d / (-min(p_a1 * p_b1, p_a0 * p_b0) + eps)
    return {"D": d, "D_prime": d_prime, "r_square": r_square}


def _watterson_theta(mat: np.ndarray, segregating_sites: int) -> float:
    n = mat.shape[0]
    if n < 2:
        raise ValueError("At least two sequences are required.")
    return float(segregating_sites / np.sum(1.0 / np.arange(1, n)))


def _tajima_pi(mat: np.ndarray, wakeley: bool = False) -> dict[str, float]:
    n = mat.shape[0]
    if n < 2:
        raise ValueError("At least two sequences are required.")
    differences = np.array(
        [np.count_nonzero(mat[i] != mat[j]) for i in range(n - 1) for j in range(i + 1, n)],
        dtype=float,
    )
    pi = float(2.0 * np.sum(differences) / (n * (n - 1)))
    result = {"pi": pi}
    if wakeley:
        result["pi2"] = float(2.0 * np.sum((differences - pi) ** 2) / (n * (n - 1)))
    return result


def _tajima_d(mat: np.ndarray, pi_hat: float, theta_hat: float, segregating_sites: int) -> float:
    n = mat.shape[0]
    if n < 2 or segregating_sites == 0:
        return 0.0
    an = np.sum(1.0 / np.arange(1, n))
    bn = np.sum(1.0 / np.arange(1, n) ** 2)
    e1 = (n + 1) / (3 * an * (n - 1)) - 1 / an**2
    e2 = (
        2 * (n**2 + n + 3) / (9 * n * (n - 1))
        - (n + 2) / (n * an)
        + bn / an**2
    ) / (an**2 + bn)
    variance = e1 * segregating_sites + e2 * segregating_sites * (segregating_sites - 1)
    if variance <= 0:
        return 0.0
    return float((pi_hat - theta_hat) / np.sqrt(variance))


def _wall_bq(seg_mat: np.ndarray) -> dict[str, float]:
    seg_mat = seg_mat.astype(bool)
    sites = seg_mat.shape[1]
    if sites < 2:
        return {"B": 0.0, "Q": 0.0}
    count = 0
    patterns: set[tuple[bool, ...]] = set()
    for i in range(sites - 1):
        col1 = ~seg_mat[:, i] if seg_mat[0, i] else seg_mat[:, i]
        col2 = ~seg_mat[:, i + 1] if seg_mat[0, i + 1] else seg_mat[:, i + 1]
        if np.array_equal(col1, col2):
            count += 1
            patterns.add(tuple(col1.tolist()))
    return {"B": count / (sites - 1), "Q": (count + len(patterns)) / sites}


def _homoplasy_index(tree: ClonalTree, node_site: np.ndarray) -> float:
    n_leaf = tree.n
    minimum_changes = 0
    observed_changes = 0
    for site in range(node_site.shape[1]):
        leaf_states = node_site[:n_leaf, site].astype(bool)
        minimum_changes += int(np.any(leaf_states) and not np.all(leaf_states))
        states = {node: {int(state)} for node, state in zip(range(1, n_leaf + 1), leaf_states)}
        for parent in range(n_leaf + 1, 2 * n_leaf):
            child_rows = np.where(tree.edge[:, 0] == parent)[0]
            children = tree.edge[child_rows, 1].astype(int)
            intersection = states[children[0]] & states[children[1]]
            if intersection:
                states[parent] = intersection
            else:
                states[parent] = states[children[0]] | states[children[1]]
                observed_changes += 1
    if observed_changes == 0:
        return 0.0
    return float(1.0 - minimum_changes / observed_changes)


def _shifted_exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-b * x) + c


def _exp_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x_scaled = x - np.min(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        params, _ = curve_fit(
            _shifted_exponential,
            x_scaled,
            y,
            p0=[1.0, 1.0, float(np.min(y))],
            maxfev=10_000,
        )
    return tuple(float(value) for value in params)


def _segment_summary_stats(tree: ClonalTree, seg_mat: np.ndarray) -> np.ndarray:
    seg_mat = np.asarray(seg_mat, dtype=bool)
    if seg_mat.ndim != 2:
        raise ValueError("`seg_mat` must be a two-dimensional matrix.")
    if tree.n != seg_mat.shape[0]:
        raise ValueError("The tree and genetic matrix must have the same number of sequences.")
    length = seg_mat.shape[1]
    if length < 1:
        raise ValueError("The genetic matrix must contain at least one site.")

    result = np.zeros(46, dtype=float)
    segregating = np.flatnonzero(seg_mat.any(axis=0) & ~seg_mat.all(axis=0))
    counts = {"near": 0, "far": 0, "20_50": 0, "50_80": 0}
    sums = {
        key: {"D": 0.0, "D_prime": 0.0, "r_square": 0.0, "g4": 0.0}
        for key in counts
    }
    r_squares: list[float] = []
    distances: list[int] = []
    incompatible: list[tuple[int, int]] = []

    for left_pos in range(len(segregating) - 1):
        for right_pos in range(left_pos + 1, len(segregating)):
            left, right = int(segregating[left_pos]), int(segregating[right_pos])
            distance = right - left
            ld = _ld(seg_mat[:, [left, right]])
            g4 = _g4_test(seg_mat[:, [left, right]])
            r_squares.append(ld["r_square"])
            distances.append(distance)
            if g4:
                incompatible.append((left, right))

            groups = ["near" if distance < length / 2 else "far"]
            if 20 <= distance < 50:
                groups.append("20_50")
            if 50 <= distance <= 80:
                groups.append("50_80")
            for group in groups:
                counts[group] += 1
                sums[group]["D"] += abs(ld["D"])
                sums[group]["D_prime"] += ld["D_prime"]
                sums[group]["r_square"] += ld["r_square"]
                sums[group]["g4"] += float(g4)

    ordered_groups = ["near", "far", "20_50", "50_80"]
    # Totals occupy 0:8 and 16:24; means occupy 8:16 and 24:32.
    for group_index, group in enumerate(ordered_groups):
        block = 0 if group_index < 2 else 16
        offset = group_index if group_index < 2 else group_index - 2
        values = sums[group]
        for metric_index, metric in enumerate(("D", "D_prime", "r_square", "g4")):
            total_index = block + 2 * metric_index + offset
            mean_index = total_index + 8
            result[total_index] = values[metric]
            result[mean_index] = values[metric] / counts[group] if counts[group] else 0.0

    result[32] = _homoplasy_index(tree, seg_mat)
    count_s = len(segregating)
    result[33] = count_s / length
    result[34] = _watterson_theta(seg_mat, count_s)
    pi = _tajima_pi(seg_mat, wakeley=True)
    result[35], result[36] = pi["pi"], pi["pi2"]
    result[37] = _tajima_d(seg_mat, result[35], result[34], count_s)
    wall = _wall_bq(seg_mat[:, segregating])
    result[38], result[39] = wall["B"], wall["Q"]

    incompatible.sort(key=lambda interval: interval[1])
    last_cut = -1
    for start, end in incompatible:
        if start > last_cut:
            result[40] += 1
            last_cut = end - 1

    result[41] = float(np.mean(r_squares)) if r_squares else 0.0
    if len(r_squares) >= 4 and len(set(distances)) >= 3:
        unique_distances = np.unique(distances)
        means = np.array(
            [np.mean([r for r, d in zip(r_squares, distances) if d == distance])
             for distance in unique_distances]
        )
        try:
            result[42:45] = _exp_regression(unique_distances.astype(float), means)
        except (RuntimeError, ValueError, TypeError, FloatingPointError):
            result[42:45] = 0.0
    result[45] = length
    return result
