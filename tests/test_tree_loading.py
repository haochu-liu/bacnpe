"""Tests for loading estimated trees into ClonalTree (Newick, CSV, arrays)."""
import csv
from pathlib import Path

import numpy as np
import pytest

from bacnpe import ClonalOrigin_seq_sim, ClonalTree
from bacnpe.tree.newick import parse_newick
from bacnpe.tree.upgma import cophenetic, hclust_average

# Input trees and the output of the R scripts likelihood-free/R/simARG/*_clonal_tree.R
# (ape + phangorn): clonal_edge.csv, clonal_node_height.csv, tip_names.csv
DATA = Path(__file__).parent / "data"
# klebsiella, estimated by ClonalFrameML (100 tips, not ultrametric)
REAL_TREE = DATA / "real_tree"
CFML = REAL_TREE / "cfml.nwk"
# clonal frame simulated by SimBac (30 tips, ultrametric)
SIM_TREE = DATA / "sim_tree"
SIMBAC = SIM_TREE / "clonal_frame.nwk"
DATASETS = {"real_tree": (REAL_TREE, CFML, 100), "sim_tree": (SIM_TREE, SIMBAC, 30)}


def _r_output(name):
    directory = DATASETS[name][0]
    edge = np.loadtxt(directory / "clonal_edge.csv", delimiter=",")
    node_height = np.loadtxt(directory / "clonal_node_height.csv", delimiter=",")
    with open(directory / "tip_names.csv", newline="") as fh:
        tips = [row[0] for row in csv.reader(fh)]
    return edge, node_height, tips


def _assert_valid(tree):
    n = tree.n
    parents, children = tree.edge[:, 0].astype(int), tree.edge[:, 1].astype(int)
    assert tree.edge.shape == (2 * (n - 1), 3) and tree.node_height.shape == (2 * n - 1,)
    assert np.all(tree.node_height[:n] == 0)
    assert np.all(np.diff(tree.node_height[n:]) >= 0)
    assert np.all(np.diff(parents) >= 0)                       # sorted by parent
    assert sorted(children.tolist()) == list(range(1, 2 * n - 1))
    np.testing.assert_allclose(
        tree.edge[:, 2], tree.node_height[parents - 1] - tree.node_height[children - 1],
        atol=1e-12,
    )
    assert tree.height == pytest.approx(tree.node_height.max())
    assert tree.length == pytest.approx(tree.edge[:, 2].sum())


def _clades(tree):
    """{frozenset of tip labels below an internal node: its height}."""
    labels = tree.tip_labels or [str(i) for i in range(1, tree.n + 1)]
    below = {i: frozenset([labels[i - 1]]) for i in range(1, tree.n + 1)}
    for p, c, _ in tree.edge:                     # parents come after their children
        below[int(p)] = below.get(int(p), frozenset()) | below[int(c)]
    return {below[i]: tree.node_height[i - 1] for i in range(tree.n + 1, 2 * tree.n)}


def _mrca(edge, a, b):
    """Common ancestor of nodes `a` and `b` in an edge matrix."""
    parent = {int(c): int(p) for p, c, _ in edge}

    def ancestors(x):
        out = [x]
        while out[-1] in parent:
            out.append(parent[out[-1]])
        return out

    of_b = set(ancestors(b))
    return next(x for x in ancestors(a) if x in of_b)


# ----------------------------------------------------------------------
# Estimated tree -> clonal tree, against the R scripts
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", DATASETS)
def test_from_newick_r_compatible_matches_r_script(name):
    _, newick, n = DATASETS[name]
    edge, node_height, tips = _r_output(name)
    tree = ClonalTree.from_newick(newick, r_compatible=True)
    _assert_valid(tree)
    assert tree.n == n and tree.tip_labels == tips
    np.testing.assert_array_equal(tree.edge[:, :2], edge[:, :2])     # same topology and numbering
    np.testing.assert_allclose(tree.edge[:, 2], edge[:, 2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(tree.node_height, node_height, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name", DATASETS)
def test_r_scripts_scale_the_common_ancestor_of_first_and_last_tip(name):
    """
    In the R output itself, 2(1 - 1/n) is the height of the common ancestor of
    tips 1 and n of the UPGMA tree t2 (e <- dist.nodes(t2)[Ntip(t2), 1] / 2).
    """
    _, _, n = DATASETS[name]
    edge, node_height, _ = _r_output(name)
    mrca = _mrca(edge, 1, n)
    assert node_height[mrca - 1] == pytest.approx(2 * (1 - 1 / n), rel=1e-13)
    # sim_tree: that node is the root; real_tree: it is not
    assert (mrca == 2 * n - 1) == (name == "sim_tree")


def test_from_newick_default_matches_r_when_that_node_is_the_root():
    edge, node_height, tips = _r_output("sim_tree")
    tree = ClonalTree.from_newick(SIMBAC)
    assert tree.tip_labels == tips
    np.testing.assert_array_equal(tree.edge[:, :2], edge[:, :2])
    np.testing.assert_allclose(tree.edge[:, 2], edge[:, 2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(tree.node_height, node_height, rtol=1e-12, atol=1e-14)
    assert tree.height == pytest.approx(2 * (1 - 1 / 30), rel=1e-14)


def test_from_newick_default_scales_root_to_coalescent_expectation():
    edge, node_height, tips = _r_output("real_tree")
    tree = ClonalTree.from_newick(CFML)
    _assert_valid(tree)
    assert tree.height == pytest.approx(2 * (1 - 1 / 100), rel=1e-14)
    assert tree.tip_labels == tips
    np.testing.assert_array_equal(tree.edge[:, :2], edge[:, :2])
    # Same tree as R up to one scale factor (R scaled a non-root node to 2(1 - 1/n))
    ratio = node_height[100:] / tree.node_height[100:]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    assert ratio[0] > 1.1


def test_upgma_keeps_an_ultrametric_tree():
    """SimBac's tree is already ultrametric: UPGMA keeps its root height."""
    parsed = parse_newick(SIMBAC.read_text())
    root_distance = np.zeros(len(parsed.parent))
    for d in range(1, len(parsed.parent)):
        root_distance[d] = root_distance[parsed.parent[d]] + parsed.length[d]
    raw = ClonalTree.from_newick(SIMBAC, root_height=None)
    assert raw.height == pytest.approx(root_distance[parsed.tip_indices].max(), rel=1e-6)


def test_from_newick_root_height_options():
    default = ClonalTree.from_newick(CFML)
    raw = ClonalTree.from_newick(CFML, root_height=None)
    tall = ClonalTree.from_newick(CFML, root_height=5)
    np.testing.assert_array_equal(raw.edge[:, :2], default.edge[:, :2])
    # Unscaled UPGMA root height = half the average distance between the two root clades
    assert 0 < raw.height < 0.01                                 # substitutions per site
    np.testing.assert_allclose(raw.node_height * (default.height / raw.height),
                               default.node_height, rtol=1e-12, atol=1e-15)
    assert tall.height == pytest.approx(5.0)


def test_from_newick_accepts_path_str_and_newick_string():
    a = ClonalTree.from_newick(CFML)
    b = ClonalTree.from_newick(str(CFML))
    c = ClonalTree.from_newick(CFML.read_text())
    for t in (b, c):
        np.testing.assert_array_equal(t.edge, a.edge)
        assert t.tip_labels == a.tip_labels


@pytest.mark.parametrize("name", DATASETS)
def test_loaded_tree_simulates_like_the_old_attribute_override(name):
    """The old example.py overwrote the attributes of a simulated tree."""
    directory, _, n = DATASETS[name]
    edge, node_height, _ = _r_output(name)
    old = ClonalTree(n)
    old.edge, old.node_height = edge, node_height
    old.height, old.length = np.max(node_height), np.sum(edge[:, 2])
    new = ClonalTree.from_csv(directory)

    np.random.seed(7)
    expected = ClonalOrigin_seq_sim(old, 0.05, 0.05, 300, 50.0)
    np.random.seed(7)
    got = ClonalOrigin_seq_sim(new, 0.05, 0.05, 300, 50.0)
    np.testing.assert_array_equal(got, expected)


# ----------------------------------------------------------------------
# UPGMA details on small trees
# ----------------------------------------------------------------------
def test_small_ultrametric_tree():
    tree = ClonalTree.from_newick("((A:1,B:1):2,(C:2,D:2):1);", root_height=None)
    assert tree.tip_labels == ["A", "B", "C", "D"]
    np.testing.assert_allclose(tree.node_height, [0, 0, 0, 0, 1, 2, 3])
    np.testing.assert_allclose(tree.edge, [[5, 1, 1], [5, 2, 1], [6, 3, 2], [6, 4, 2],
                                           [7, 5, 2], [7, 6, 1]])
    scaled = ClonalTree.from_newick("((A:1,B:1):2,(C:2,D:2):1);")
    np.testing.assert_allclose(scaled.node_height, np.array([0, 0, 0, 0, 1, 2, 3]) * 1.5 / 3)


def test_non_ultrametric_tree_and_first_minimum_tie_break():
    # d(A,B) = d(A,C) = 4, d(B,C) = 6: like R's hclust, the first pair (A,B) is merged
    tree = ClonalTree.from_newick("((A:1,B:3):1,C:2);", root_height=None)
    np.testing.assert_allclose(tree.node_height, [0, 0, 0, 2, 2.5])
    np.testing.assert_allclose(tree.edge, [[4, 1, 2], [4, 2, 2], [5, 3, 2.5], [5, 4, 0.5]])


def test_cophenetic_and_hclust():
    parsed = parse_newick("(A:1,(B:2,C:3):0.5,D:4);")         # multifurcating root
    D = cophenetic(parsed)
    np.testing.assert_allclose(D, [[0, 3.5, 4.5, 5], [3.5, 0, 5, 6.5],
                                   [4.5, 5, 0, 7.5], [5, 6.5, 7.5, 0]])
    merge, height = hclust_average(D)
    # R: hclust(as.dist(D), "average")
    np.testing.assert_array_equal(merge, [[-1, -2], [-3, 1], [-4, 2]])
    np.testing.assert_allclose(height, [3.5, 4.75, 19 / 3])


def test_newick_to_csv_round_trip(tmp_path):
    tree = ClonalTree.from_newick(CFML)
    out = tree.to_csv(tmp_path / "kleb")
    assert sorted(p.name for p in out.iterdir()) == ["clonal_edge.csv", "clonal_node_height.csv",
                                                     "tip_names.csv"]
    back = ClonalTree.from_csv(out)
    np.testing.assert_array_equal(back.edge, tree.edge)                # full precision
    np.testing.assert_array_equal(back.node_height, tree.node_height)
    assert back.tip_labels == tree.tip_labels
    with pytest.raises(FileExistsError):
        tree.to_csv(out)
    tree.to_csv(out, overwrite=True)


def test_csv_without_tip_names(tmp_path):
    np.random.seed(3)
    sim = ClonalTree(6)
    sim.to_csv(tmp_path)
    assert not (tmp_path / "tip_names.csv").exists()
    back = ClonalTree.from_csv(tmp_path)
    np.testing.assert_array_equal(back.edge, sim.edge)
    assert back.tip_labels is None


def test_to_newick_round_trip():
    tree = ClonalTree.from_newick(CFML)
    back = ClonalTree.from_newick(tree.to_newick(), root_height=None)
    assert sorted(back.tip_labels) == sorted(tree.tip_labels)
    a, b = _clades(tree), _clades(back)
    assert a.keys() == b.keys()
    np.testing.assert_allclose([b[k] for k in a], [a[k] for k in a], rtol=1e-12, atol=1e-15)


def test_simulated_tree_to_newick_uses_leaf_numbers():
    np.random.seed(1)
    sim = ClonalTree(5)
    back = ClonalTree.from_newick(sim.to_newick(), root_height=None)
    a, b = _clades(sim), _clades(back)
    assert a.keys() == b.keys()
    np.testing.assert_allclose([b[k] for k in a], [a[k] for k in a], rtol=1e-12)


def test_simulation_runs_on_loaded_tree():
    tree = ClonalTree.from_newick(CFML)
    stats = ClonalOrigin_seq_sim(tree, 0.02, 0.05, 200, 30.0)
    assert stats.shape == (30,)


# ----------------------------------------------------------------------
# Newick parsing
# ----------------------------------------------------------------------
def test_parser_handles_quotes_comments_whitespace_and_notation():
    text = "[comment]\n( 'tip one':4.5e-1 , (B:2.5E-1,'it''s':0.25)NODE_7:0.15 ) root ;\n"
    tree = ClonalTree.from_newick(text, root_height=None)
    assert tree.tip_labels == ["tip one", "B", "it's"]
    np.testing.assert_allclose(tree.node_height, [0, 0, 0, 0.25, 0.425])
    assert "'tip one'" in tree.to_newick() and "'it''s'" in tree.to_newick()


@pytest.mark.parametrize("text, match", [
    ("((A:1,B:1),C:1);", "length"),
    ("((A:1,B:1):1,C:1;", "parenthes"),
    ("(A:1,B:1));", r"\)"),
    ("(A:1,B:1);(A:1,B:1);", "more than one tree"),
    ("((A:1,:1):1,C:1);", "name"),
    ("(A:1,B:x);", "branch length"),
    ("A;", "at least 2 tips"),
    ("(A:1,A:1);", "unique"),
    ("(A:-1,B:1);", "negative"),
    ("(A:0,B:0);", "distance 0"),
])
def test_bad_newick_raises(text, match):
    with pytest.raises(ValueError, match=match):
        ClonalTree.from_newick(text)


@pytest.mark.parametrize("bad, err", [("tall", TypeError), (True, TypeError), (0, ValueError),
                                      (-1.0, ValueError), (np.inf, ValueError)])
def test_bad_root_height(bad, err):
    with pytest.raises(err):
        ClonalTree.from_newick("(A:1,B:1);", root_height=bad)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ClonalTree.from_newick("no_such_tree.nwk")


# ----------------------------------------------------------------------
# from_arrays
# ----------------------------------------------------------------------
EDGE = np.array([[4, 1, 1.0], [4, 2, 1.0], [5, 3, 3.0], [5, 4, 2.0]])
HEIGHT = np.array([0, 0, 0, 1.0, 3.0])


def test_from_arrays_copies_and_sorts_edges():
    shuffled = EDGE[[2, 0, 3, 1]]
    tree = ClonalTree.from_arrays(shuffled, HEIGHT, tip_labels=("a", "b", "c"))
    _assert_valid(tree)
    np.testing.assert_array_equal(tree.edge[:, 0], [4, 4, 5, 5])
    assert tree.n == 3 and tree.height == 3.0 and tree.length == 7.0
    assert tree.tip_labels == ["a", "b", "c"] and repr(tree) == "ClonalTree(n=3)"
    shuffled[0, 2] = 99.0
    assert tree.edge[:, 2].max() == 3.0


def test_from_arrays_single_leaf():
    tree = ClonalTree.from_arrays(np.empty((0, 3)), [0.0])
    assert tree.n == 1 and tree.edge.shape == (0, 3) and tree.height == 0.0


@pytest.mark.parametrize("edge, height, match", [
    (EDGE[:3], HEIGHT, "shape"),
    (EDGE, HEIGHT[:4], "length 2n-1"),
    (np.where(np.arange(12).reshape(4, 3) == 1, 1.5, EDGE), HEIGHT, "integers"),
    (EDGE * [1, 0, 1] + [0, 1, 0], HEIGHT, "child"),
    (np.array([[4, 1, 1.0], [4, 2, 1.0], [4, 3, 1.0], [5, 4, 2.0]]), HEIGHT, "child|two children"),
    (EDGE, np.array([0, 0.5, 0, 1.0, 3.0]), "height 0"),
    (np.array([[5, 1, 2.0], [5, 2, 2.0], [6, 3, 1.0], [6, 4, 1.0], [7, 5, 1.0], [7, 6, 2.0]]),
     np.array([0, 0, 0, 0, 2.0, 1.0, 3.0]), "non-decreasing"),
    (EDGE * [1, 1, 2], HEIGHT, "Edge lengths"),
    (np.where(np.isin(np.arange(12).reshape(4, 3), [2]), np.nan, EDGE), HEIGHT, "finite"),
])
def test_from_arrays_rejects_invalid_trees(edge, height, match):
    with pytest.raises(ValueError, match=match):
        ClonalTree.from_arrays(edge, height)


@pytest.mark.parametrize("labels, err", [("abc", TypeError), (["a", "b"], ValueError),
                                         (["a", "b", 3], TypeError), (["a", "a", "b"], ValueError)])
def test_from_arrays_bad_tip_labels(labels, err):
    with pytest.raises(err):
        ClonalTree.from_arrays(EDGE, HEIGHT, tip_labels=labels)
