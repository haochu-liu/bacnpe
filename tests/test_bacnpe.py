import numpy as np
import pytest

import bacnpe
from bacnpe import ClonalOrigin_seq_sim, ClonalTree


def test_public_api_contains_only_requested_callables():
    assert bacnpe.__all__ == ["ClonalTree", "ClonalOrigin_seq_sim"]


def test_clonal_tree_result_types_and_sizes():
    np.random.seed(10)
    tree = ClonalTree(8)

    assert tree.n == 8
    assert isinstance(tree.edge, np.ndarray)
    assert tree.edge.shape == (14, 3)
    assert tree.node_height.shape == (15,)
    assert np.issubdtype(tree.edge.dtype, np.floating)
    assert np.all(tree.edge[:, 2] >= 0)
    assert tree.height == pytest.approx(tree.node_height.max())
    assert tree.length == pytest.approx(tree.edge[:, 2].sum())


@pytest.mark.parametrize("bad_n", [0, -1, 1.5, "8", True, None])
def test_clonal_tree_rejects_invalid_n(bad_n):
    with pytest.raises(ValueError):
        ClonalTree(bad_n)


def test_sequence_simulator_result_type_and_size_no_events():
    np.random.seed(20)
    tree = ClonalTree(6)
    result = ClonalOrigin_seq_sim(tree, rho_site=0, theta_site=0, L=12, delta=4)

    assert isinstance(result, np.ndarray)
    assert result.shape == (46,)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))
    assert result[45] == 12


def test_sequence_simulator_result_type_and_size_with_random_events():
    np.random.seed(30)
    tree = ClonalTree(8)
    result = ClonalOrigin_seq_sim(
        tree, rho_site=0.03, theta_site=0.08, L=30, delta=8
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (46,)
    assert np.all(np.isfinite(result))
    assert 0 <= result[33] <= 1
    assert result[45] == 30


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("tree", object(), TypeError),
        ("rho_site", -0.1, ValueError),
        ("rho_site", "0.1", TypeError),
        ("theta_site", -0.1, ValueError),
        ("theta_site", np.inf, ValueError),
        ("L", 0, ValueError),
        ("L", 2.5, TypeError),
        ("L", True, TypeError),
        ("delta", 0.5, ValueError),
        ("delta", "10", TypeError),
    ],
)
def test_sequence_simulator_rejects_invalid_inputs(argument, value, error):
    np.random.seed(40)
    values = {
        "tree": ClonalTree(5),
        "rho_site": 0.01,
        "theta_site": 0.02,
        "L": 20,
        "delta": 5,
    }
    values[argument] = value
    with pytest.raises(error):
        ClonalOrigin_seq_sim(**values)


def test_sequence_simulator_rejects_single_leaf_tree():
    with pytest.raises(ValueError):
        ClonalOrigin_seq_sim(ClonalTree(1), 0.01, 0.02, 20, 5)
