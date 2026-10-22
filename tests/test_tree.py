"""Tests for the public ClonalTree class."""
import numpy as np
import pytest

from bacnpe import ClonalTree


@pytest.mark.parametrize("n", [2, 3, 10, 50])
def test_shapes_and_types(n):
    tree = ClonalTree(n)
    assert tree.n == n and isinstance(tree.n, int)
    assert isinstance(tree.edge, np.ndarray) and tree.edge.dtype == np.float64
    assert tree.edge.shape == (2 * (n - 1), 3)
    assert isinstance(tree.node_height, np.ndarray) and tree.node_height.shape == (2 * n - 1,)
    assert isinstance(tree.height, float) and isinstance(tree.length, float)


@pytest.mark.parametrize("n", [2, 5, 30])
def test_tree_is_valid_genealogy(n):
    tree = ClonalTree(n)
    parents = tree.edge[:, 0].astype(int)
    children = tree.edge[:, 1].astype(int)

    # Leaves at height 0, internal node heights strictly increasing
    assert np.all(tree.node_height[:n] == 0)
    assert np.all(np.diff(tree.node_height[n - 1:]) > 0)
    # Every internal node has exactly two children, every non-root exactly one parent
    assert sorted(np.unique(parents, return_counts=True)[1].tolist()) == [2] * (n - 1)
    assert sorted(children.tolist()) == list(range(1, 2 * n - 1))
    # Edge lengths are positive and equal to the height differences
    assert np.all(tree.edge[:, 2] > 0)
    np.testing.assert_allclose(
        tree.edge[:, 2], tree.node_height[parents - 1] - tree.node_height[children - 1]
    )
    assert tree.height == pytest.approx(tree.node_height.max())
    assert tree.length == pytest.approx(tree.edge[:, 2].sum())


def test_single_leaf_tree():
    tree = ClonalTree(1)
    assert tree.edge.shape == (0, 3)
    assert tree.node_height.shape == (1,)
    assert tree.height == 0.0 and tree.length == 0.0


def test_numpy_integer_accepted():
    assert ClonalTree(np.int64(6)).n == 6
    assert isinstance(ClonalTree(np.int32(6)).n, int)


def test_reproducible_with_seed():
    np.random.seed(1)
    a = ClonalTree(12)
    np.random.seed(1)
    b = ClonalTree(12)
    np.testing.assert_array_equal(a.edge, b.edge)


def test_to_dict_and_repr():
    tree = ClonalTree(4)
    d = tree.to_dict()
    assert set(d) == {"edge", "node_height", "height", "length", "n"}
    assert d["n"] == 4
    assert repr(tree) == "ClonalTree(n=4)"


@pytest.mark.parametrize("bad", [2.0, 2.5, "3", None, [3], True, np.float64(3.0)])
def test_wrong_type_raises_type_error(bad):
    with pytest.raises(TypeError):
        ClonalTree(bad)


@pytest.mark.parametrize("bad", [0, -1, np.int64(-5)])
def test_bad_value_raises_value_error(bad):
    with pytest.raises(ValueError):
        ClonalTree(bad)
