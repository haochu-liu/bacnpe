"""Tests for the internal ARG and mutation simulators."""
import numpy as np
import pytest

from bacnpe import ClonalTree
from bacnpe.simulation import _ARG, _add_mutation, _clonal_origin_nodes


@pytest.mark.parametrize("rho", [0.0, 0.02, 0.2])
@pytest.mark.parametrize("n, L, delta", [(3, 5, 1.0), (10, 100, 20.0), (25, 40, 3.0)])
def test_arg_structure(n, L, delta, rho):
    arg = _ARG(ClonalTree(n), rho, L, delta)
    n_edges, n_nodes = arg.edge.shape[0], arg.node_height.shape[0]

    # Sizes: every recombination adds 3 nodes and 4 edges
    n_recomb = (n_nodes - (2 * n - 1)) // 3
    assert n_nodes == 2 * n - 1 + 3 * n_recomb
    assert n_edges == 2 * (n - 1) + 4 * n_recomb

    # Types and shapes
    assert arg.edge.dtype == np.float64 and arg.edge.shape[1] == 3
    assert arg.edge_mat.dtype == bool and arg.edge_mat.shape == (n_edges, L)
    assert arg.node_mat.dtype == bool and arg.node_mat.shape == (n_nodes, L)
    assert arg.node_clonal.dtype == bool and arg.node_clonal.shape == (n_nodes,)
    assert isinstance(arg.height, float) and isinstance(arg.length, float)
    assert arg.rho == pytest.approx(L * rho) and arg.L == L and arg.delta == delta

    # Graph sanity: heights sorted, non-negative edge lengths, leaves carry all sites
    assert np.all(np.diff(arg.node_height) >= 0)
    assert np.all(arg.edge[:, 2] >= 0)
    assert np.all(arg.node_mat[:n])
    parents = arg.edge[:, 0].astype(int)
    children = arg.edge[:, 1].astype(int)
    assert np.all(arg.node_height[parents - 1] >= arg.node_height[children - 1])
    # Every site is ancestral at the top node of the graph
    assert np.all(arg.node_mat[-1])
    assert arg.length == pytest.approx(arg.edge[:, 2].sum())


def test_arg_without_recombination_is_clonal_tree():
    tree = ClonalTree(8)
    arg = _ARG(tree, 0.0, 30, 5.0)
    np.testing.assert_array_equal(arg.edge, tree.edge)
    np.testing.assert_array_equal(arg.node_height, tree.node_height)
    assert arg.edge_mat.all() and arg.node_mat.all() and arg.node_clonal.all()


def test_arg_custom_k():
    arg = _ARG(ClonalTree(5), 0.05, 100, 10.0, k=7)
    assert arg.node_mat.shape[1] == 7 and arg.L == 100


def test_arg_bad_inputs():
    tree = ClonalTree(5)
    with pytest.raises(TypeError):
        _ARG("not a tree", 0.1, 10, 2.0)
    with pytest.raises(TypeError):
        _ARG(tree, 0.1, 10.0, 2.0)
    with pytest.raises(ValueError):
        _ARG(tree, 0.1, 10, 0.5)
    with pytest.raises(ValueError):
        _ARG(tree, -0.1, 10, 2.0)


@pytest.mark.parametrize("theta", [0.0, 0.01, 0.5])
def test_add_mutation_output(theta):
    n, L = 10, 60
    arg = _ARG(ClonalTree(n), 0.05, L, 10.0)
    node_site = _add_mutation(arg, theta)
    assert isinstance(node_site, np.ndarray)
    assert node_site.dtype == bool
    assert node_site.shape == (arg.node_height.shape[0], L)
    if theta == 0.0:
        # Without mutation every node has the root sequence
        assert np.all(node_site == node_site[0])


def test_clonal_origin_nodes_sorted_by_height():
    # columns: b_edge, b_height, a_edge, a_height, x, y
    recomb_edge = np.array([
        [1, 0.5, 2, 1.5, 1, 1],
        [2, 0.2, 1, 0.9, 1, 1],
        [3, 0.1, 1, 0.3, 1, 1],
    ], dtype=float)
    # Edge 1 holds: out node of recomb 1 (0.5), in nodes of recomb 2 (0.9) and 3 (0.3)
    np.testing.assert_array_equal(_clonal_origin_nodes(recomb_edge, 1), [3, -1, 2])
    np.testing.assert_array_equal(_clonal_origin_nodes(recomb_edge, 2), [-2, 1])
    assert _clonal_origin_nodes(recomb_edge, 5).size == 0
