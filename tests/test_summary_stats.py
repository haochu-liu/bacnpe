"""
Tests for the internal summary statistic helpers.

Small hand-made matrices with known answers, plus checks that the
vectorised helpers agree exactly with straightforward per-pair loops
(the original implementation).
"""
import numpy as np
import pytest

from bacnpe import ClonalTree, segment_summary_stats
from bacnpe.summary_stats import (
    _exp_regression,
    _g4_test,
    _homoplasy_index,
    _hudson_rm,
    _kelly_z,
    _ld,
    _pairwise_ld_g4,
    _tajima_d,
    _tajima_pi,
    _wall_bq,
    _watterson_theta,
)

EPS = np.finfo(float).eps


# ------------------------------------------------------------ naive references

def ref_ld(mat):
    """Per-pair LD exactly as in the original LD.py."""
    n = mat.shape[0]
    p_A = np.sum(mat[:, 0]) / n
    p_a = 1 - p_A
    p_B = np.sum(mat[:, 1]) / n
    p_b = 1 - p_B
    p_AB = np.sum(mat[:, 0] & mat[:, 1]) / n
    D_AB = p_AB - (p_A * p_B)
    r_square = D_AB ** 2 / (p_A * p_a * p_B * p_b + EPS)
    if D_AB > 0:
        D_prime = D_AB / (min(p_A * p_b, p_a * p_B) + EPS)
    else:
        D_prime = D_AB / (-min(p_A * p_B, p_a * p_b) + EPS)
    return {"D": D_AB, "D_prime": D_prime, "r_square": r_square}


def ref_pi(mat):
    """Tajima's pi and Wakeley's pi^2 exactly as in the original Tajima_pi.py."""
    n = mat.shape[0]
    pi_list = [np.sum(mat[i] != mat[j]) for i in range(n - 1) for j in range(i + 1, n)]
    pi = 2 * np.sum(pi_list) / (n * (n - 1))
    pi2 = 2 * np.sum([(x - pi) ** 2 for x in pi_list]) / (n * (n - 1))
    return pi, pi2


def random_matrix(n, L, p=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, L)) < p


# ------------------------------------------------------------------------- LD

def test_ld_perfect_linkage():
    mat = np.array([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=bool)
    res = _ld(mat)
    assert res["D"] == pytest.approx(0.25)
    assert res["r_square"] == pytest.approx(1.0)
    assert res["D_prime"] == pytest.approx(1.0)
    assert all(isinstance(v, float) for v in res.values())


def test_ld_independent_sites():
    mat = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=bool)
    res = _ld(mat)
    assert res["D"] == pytest.approx(0.0)
    assert res["r_square"] == pytest.approx(0.0)


def test_ld_wrong_shape():
    with pytest.raises(ValueError):
        _ld(np.zeros((5, 3), dtype=bool))


@pytest.mark.parametrize("seed", range(5))
def test_pairwise_ld_matches_per_pair_loop(seed):
    mat = random_matrix(12, 15, seed=seed)
    i, j = np.triu_indices(15, k=1)
    res = _pairwise_ld_g4(mat, i, j)
    for k in range(i.size):
        ref = ref_ld(mat[:, [i[k], j[k]]])
        for key in ("D", "D_prime", "r_square"):
            assert res[key][k] == ref[key]  # bit-for-bit identical


# ------------------------------------------------------- four-gamete / Rm / Z

def test_g4_test():
    assert _g4_test(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    assert not _g4_test(np.array([[0, 0], [0, 1], [1, 1], [1, 1]]))
    with pytest.raises(ValueError):
        _g4_test(np.array([[0, 0], [1, 1]]))  # fewer than 3 rows
    with pytest.raises(ValueError):
        _g4_test(np.zeros((4, 3)))


def test_hudson_rm_known_example():
    # Sites 0-1 incompatible, 1-2 compatible, 2-3 incompatible -> Rm = 2
    mat = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 0],
    ], dtype=bool)
    assert _hudson_rm(mat) == 2


def test_hudson_rm_and_kelly_z_without_segregating_sites():
    mat = np.zeros((5, 10), dtype=bool)
    assert _hudson_rm(mat) == 0
    assert _kelly_z(mat) == 0.0


def test_kelly_z_matches_mean_r2():
    mat = random_matrix(10, 8, seed=3)
    seg = np.where(mat.any(0) & ~mat.all(0))[0]
    r2 = [ref_ld(mat[:, [a, b]])["r_square"] for x, a in enumerate(seg) for b in seg[x + 1:]]
    assert _kelly_z(mat) == pytest.approx(np.mean(r2))


# ------------------------------------------------------------------ diversity

def test_watterson_theta():
    # n = 4: a_n = 1 + 1/2 + 1/3
    assert _watterson_theta(4, 11) == pytest.approx(11 / (1 + 1 / 2 + 1 / 3))
    assert _watterson_theta(4, 0) == 0.0


def test_tajima_pi_known_value():
    mat = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=bool)
    # pairwise differences: (0,1)=1, (0,2)=2, (1,2)=1 -> mean 4/3
    res = _tajima_pi(mat, Wakeley=True)
    assert res["pi"] == pytest.approx(4 / 3)
    assert res["pi2"] == pytest.approx(((1 - 4 / 3) ** 2 * 2 + (2 - 4 / 3) ** 2) / 3)
    assert set(_tajima_pi(mat)) == {"pi"}


@pytest.mark.parametrize("seed", range(5))
def test_tajima_pi_matches_loop(seed):
    mat = random_matrix(9, 40, seed=seed)
    res = _tajima_pi(mat, Wakeley=True)
    pi, pi2 = ref_pi(mat)
    assert res["pi"] == pi and res["pi2"] == pi2


def test_tajima_d():
    # Neutral expectation pi == theta -> D == 0; no segregating site -> NaN
    assert _tajima_d(10, 3.0, 3.0, 8) == 0.0
    assert np.isnan(_tajima_d(10, 0.0, 0.0, 0))
    assert _tajima_d(10, 1.0, 3.0, 8) < 0 < _tajima_d(10, 5.0, 3.0, 8)


# ------------------------------------------------------------------ Wall's B/Q

def test_wall_bq():
    # Columns 0,1 share a bipartition (1 is the complement of 0),
    # 1,2 differ, 2,3 share another bipartition
    mat = np.array([
        [0, 1, 0, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 1, 1],
    ], dtype=bool)
    res = _wall_bq(mat)
    assert res["B"] == pytest.approx(2 / 3)
    assert res["Q"] == pytest.approx((2 + 2) / 4)
    assert _wall_bq(mat[:, :1]) == {"B": 0.0, "Q": 0.0}


# ------------------------------------------------------------------ homoplasy

def _caterpillar_tree():
    """Tree ((1,2),3),4) with fixed edges."""
    tree = ClonalTree(4)
    tree.edge = np.array([
        [5, 1, 1.0], [5, 2, 1.0],
        [6, 5, 1.0], [6, 3, 2.0],
        [7, 6, 1.0], [7, 4, 3.0],
    ])
    tree.node_height = np.array([0, 0, 0, 0, 1, 2, 3], dtype=float)
    return tree


def test_homoplasy_index_compatible_and_homoplastic_sites():
    tree = _caterpillar_tree()
    compatible = np.array([[1], [1], [0], [0]], dtype=bool)  # one change
    assert _homoplasy_index(tree, compatible) == 0.0
    homoplastic = np.array([[1], [0], [1], [0]], dtype=bool)  # two changes
    assert _homoplasy_index(tree, homoplastic) == pytest.approx(0.5)
    assert _homoplasy_index(tree, np.zeros((4, 3), dtype=bool)) == 0.0


# -------------------------------------------------------------- the 30 stats

def test_segment_summary_stats_output():
    tree = ClonalTree(8)
    mat = random_matrix(8, 50, p=0.2, seed=1)
    s = segment_summary_stats(tree, mat)
    assert s.shape == (30,) and s.dtype == np.float64
    assert s[29] == 50
    # 0/1 integer input gives the same result as boolean input
    np.testing.assert_array_equal(s, segment_summary_stats(tree, mat.astype(int)))


def test_segment_summary_stats_bad_inputs():
    tree = ClonalTree(8)
    with pytest.raises(ValueError):
        segment_summary_stats(tree, np.zeros((7, 10), dtype=bool))  # wrong #rows
    with pytest.raises(ValueError):
        segment_summary_stats(tree, np.full((8, 10), 2))  # not binary
    with pytest.raises(ValueError):
        segment_summary_stats(tree, np.zeros(8, dtype=bool))  # not 2-D
    with pytest.raises(TypeError):
        segment_summary_stats(tree, np.full((8, 10), "a"))
    with pytest.raises(TypeError):
        segment_summary_stats("tree", np.zeros((8, 10), dtype=bool))


# ----------------------------------------------------------------- regression

def test_exp_regression_recovers_parameters():
    x = np.arange(1, 60, dtype=float)
    y = 0.8 * np.exp(-0.1 * (x - 1)) + 0.05
    a, b, c = _exp_regression(x, y)
    assert (a, b, c) == pytest.approx((0.8, 0.1, 0.05), rel=1e-4)


def test_segment_summary_stats_is_public():
    import bacnpe
    import bacnpe.summary_stats
    assert "segment_summary_stats" in bacnpe.__all__
    assert "segment_summary_stats" in bacnpe.summary_stats.__all__
    assert bacnpe.segment_summary_stats is bacnpe.summary_stats.segment_summary_stats
