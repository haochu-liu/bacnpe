"""Tests for the public ClonalOrigin_seq_sim function."""
import numpy as np
import pytest

from bacnpe import ClonalOrigin_seq_sim, ClonalTree
from bacnpe.summary_stats import STAT_NAMES

# Entries that are counts or proportions and must be >= 0
NON_NEGATIVE = [0, 1, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 29]


def test_stat_names_length():
    assert len(STAT_NAMES) == 30
    assert len(set(STAT_NAMES)) == 30


@pytest.mark.parametrize(
    "n, rho, theta, L, delta",
    [
        (3, 0.0, 0.1, 10, 1.0),
        (10, 0.05, 0.05, 100, 20.0),
        (20, 0.1, 0.02, 200, 5.0),
        (8, 0.01, 0.0, 50, 10.0),
        (5, 0.02, 0.05, 1, 3.0),
    ],
)
def test_output_type_shape_and_ranges(n, rho, theta, L, delta):
    tree = ClonalTree(n)
    s = ClonalOrigin_seq_sim(tree, rho, theta, L, delta)
    assert isinstance(s, np.ndarray)
    assert s.shape == (30,)
    assert s.dtype == np.float64
    assert s[29] == L
    # Everything except Tajima's D (NaN when no segregating site) is finite
    assert np.all(np.isfinite(np.delete(s, 21)))
    assert np.all(s[NON_NEGATIVE] >= 0)
    # Proportions
    assert 0 <= s[16] <= 1  # homoplasy index
    assert 0 <= s[17] <= 1  # proportion of segregating sites
    assert 0 <= s[22] <= 1  # Wall's B
    # Integer-valued counts
    for idx in (6, 7, 24):
        assert float(s[idx]).is_integer()


def test_no_mutation_gives_no_polymorphism():
    tree = ClonalTree(10)
    s = ClonalOrigin_seq_sim(tree, 0.05, 0.0, 100, 10.0)
    # No segregating sites: all LD / diversity statistics are zero
    assert np.all(s[:21] == 0)
    assert np.isnan(s[21])  # Tajima's D is 0/0
    assert np.all(s[22:29] == 0)


def test_no_recombination_has_no_homoplasy():
    # Without recombination every site evolves on the clonal tree, so
    # parsimony on the clonal tree needs exactly one change per site
    # (unless a site is hit twice by mutation, which is rare at this theta).
    tree = ClonalTree(10)
    s = ClonalOrigin_seq_sim(tree, 0.0, 0.001, 200, 10.0)
    assert s[16] == 0.0


def test_reproducible_with_seed():
    tree = ClonalTree(10)
    np.random.seed(99)
    a = ClonalOrigin_seq_sim(tree, 0.05, 0.05, 100, 20.0)
    np.random.seed(99)
    b = ClonalOrigin_seq_sim(tree, 0.05, 0.05, 100, 20.0)
    np.testing.assert_array_equal(a, b)
    c = ClonalOrigin_seq_sim(tree, 0.05, 0.05, 100, 20.0)
    assert not np.array_equal(a, c, equal_nan=True)


def test_numpy_scalars_accepted():
    tree = ClonalTree(np.int64(6))
    s = ClonalOrigin_seq_sim(tree, np.float32(0.05), np.float64(0.05), np.int32(50), np.int64(10))
    assert s.shape == (30,) and s[29] == 50


def test_integer_rates_accepted():
    tree = ClonalTree(6)
    assert ClonalOrigin_seq_sim(tree, 0, 0, 20, 5).shape == (30,)


# ---------------------------------------------------------------- bad inputs

@pytest.mark.parametrize("bad_tree", [None, 10, "tree", np.zeros((18, 3))])
def test_tree_must_be_clonal_tree(bad_tree):
    with pytest.raises(TypeError):
        ClonalOrigin_seq_sim(bad_tree, 0.05, 0.05, 100, 20.0)


@pytest.mark.parametrize("n", [1, 2])
def test_tree_too_small(n):
    with pytest.raises(ValueError):
        ClonalOrigin_seq_sim(ClonalTree(n), 0.05, 0.05, 100, 20.0)


@pytest.mark.parametrize("bad_L", [100.0, "100", None, True, [100]])
def test_L_wrong_type(tree10, bad_L):
    with pytest.raises(TypeError):
        ClonalOrigin_seq_sim(tree10, 0.05, 0.05, bad_L, 20.0)


@pytest.mark.parametrize("bad_L", [0, -5])
def test_L_bad_value(tree10, bad_L):
    with pytest.raises(ValueError):
        ClonalOrigin_seq_sim(tree10, 0.05, 0.05, bad_L, 20.0)


@pytest.mark.parametrize("arg", ["rho_site", "theta_site", "delta"])
@pytest.mark.parametrize("bad", ["0.1", None, [0.1], np.array([0.1, 0.2]), False])
def test_rates_wrong_type(tree10, arg, bad):
    kwargs = dict(rho_site=0.05, theta_site=0.05, L=100, delta=20.0)
    kwargs[arg] = bad
    with pytest.raises(TypeError):
        ClonalOrigin_seq_sim(tree10, **kwargs)


@pytest.mark.parametrize(
    "arg, bad",
    [
        ("rho_site", -0.1), ("rho_site", np.nan), ("rho_site", np.inf),
        ("theta_site", -1.0), ("theta_site", np.nan), ("theta_site", -np.inf),
        ("delta", 0.0), ("delta", -2.0), ("delta", 0.5), ("delta", np.nan), ("delta", np.inf),
    ],
)
def test_rates_bad_value(tree10, arg, bad):
    kwargs = dict(rho_site=0.05, theta_site=0.05, L=100, delta=20.0)
    kwargs[arg] = bad
    with pytest.raises(ValueError):
        ClonalOrigin_seq_sim(tree10, **kwargs)


def test_bad_input_does_not_touch_tree(tree10):
    edge = tree10.edge.copy()
    with pytest.raises(ValueError):
        ClonalOrigin_seq_sim(tree10, -1.0, 0.05, 100, 20.0)
    np.testing.assert_array_equal(edge, tree10.edge)


def test_simulation_does_not_modify_tree(tree10):
    edge, heights = tree10.edge.copy(), tree10.node_height.copy()
    ClonalOrigin_seq_sim(tree10, 0.1, 0.05, 100, 20.0)
    np.testing.assert_array_equal(edge, tree10.edge)
    np.testing.assert_array_equal(heights, tree10.node_height)
