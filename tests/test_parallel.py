"""Tests for the parallel simulation driver."""
import numpy as np
import pytest

from bacnpe import ClonalOrigin_seq_sim, ClonalTree, parallel_sim
from bacnpe.parallel import PARAM_NAMES, PARAMS_FILE, STATS_FILE
from bacnpe.summary_stats import STAT_NAMES

PRIORS = dict(rho_range=(0.0, 0.2), theta_range=(0.0, 0.1), L_range=(20, 60))


def _run(tree, path, n_sims=7, **kwargs):
    kwargs = {**dict(delta=10.0, chunk_size=3, max_workers=1, seed=1, progress=False), **kwargs}
    return parallel_sim(tree, n_sims=n_sims, save_dir=path, **PRIORS, **kwargs)


def _load(paths):
    return tuple(np.loadtxt(p, delimiter=",", ndmin=2) for p in paths)


def test_output_files(tree10, tmp_path):
    out = tmp_path / "new" / "dir"           # created if missing
    paths = _run(tree10, out)
    assert paths == (out / PARAMS_FILE, out / STATS_FILE)
    params, stats = _load(paths)
    assert params.shape == (7, len(PARAM_NAMES)) and stats.shape == (7, len(STAT_NAMES))

    rho, theta, L = params.T
    assert np.all((0.0 <= rho) & (rho <= 0.2))
    assert np.all((0.0 <= theta) & (theta <= 0.1))
    assert np.all((20 <= L) & (L <= 60)) and np.all(L == np.round(L))
    # The last summary statistic is L, so rows of the two files line up
    np.testing.assert_array_equal(stats[:, 29], L)

    header = paths[1].read_text().splitlines()[:2]
    assert header == ["# " + ",".join(STAT_NAMES), "# seed=1"]


def test_matches_direct_simulation(tree10, tmp_path):
    """Row i equals ClonalOrigin_seq_sim seeded with the i-th simulation seed."""
    params, stats = _load(_run(tree10, tmp_path, n_sims=4))
    seeds = np.random.SeedSequence(1).spawn(2)[1].generate_state(4, dtype=np.uint32)
    for (rho, theta, L), s, sim_seed in zip(params, stats, seeds):
        np.random.seed(int(sim_seed))
        expected = ClonalOrigin_seq_sim(tree10, rho, theta, int(L), 10.0)
        np.testing.assert_allclose(s, expected, rtol=1e-15, equal_nan=True)


def test_reproducible_and_independent_of_workers_and_chunks(tree10, tmp_path):
    serial = _load(_run(tree10, tmp_path / "a", chunk_size=3, max_workers=1))
    parallel = _load(_run(tree10, tmp_path / "b", chunk_size=2, max_workers=2))
    for x, y in zip(serial, parallel):
        np.testing.assert_array_equal(x, y)
    other_seed = _load(_run(tree10, tmp_path / "c", seed=2))
    assert not np.array_equal(serial[0], other_seed[0])


def test_global_random_state_untouched(tree10, tmp_path):
    before = np.random.get_state()[1].copy()
    _run(tree10, tmp_path)
    np.testing.assert_array_equal(np.random.get_state()[1], before)


def test_point_priors_and_single_chunk(tree10, tmp_path):
    # Positional arguments in the documented order; low == high fixes the value
    paths = parallel_sim(tree10, (0.05, 0.05), (0, 0), (30, 30), 5.0, 3, tmp_path,
                             chunk_size=100, max_workers=1, seed=0,
                             progress=False)
    params, stats = _load(paths)
    np.testing.assert_array_equal(params, [[0.05, 0.0, 30]] * 3)
    assert stats.shape == (3, 30)


def test_overwrite(tree10, tmp_path):
    _run(tree10, tmp_path)
    with pytest.raises(FileExistsError):
        _run(tree10, tmp_path)
    _run(tree10, tmp_path, n_sims=2, overwrite=True)
    params, stats = _load((tmp_path / PARAMS_FILE, tmp_path / STATS_FILE))
    assert params.shape[0] == 2 and stats.shape[0] == 2   # replaced, not appended


@pytest.mark.parametrize("kwargs, error", [
    (dict(rho_range=(0.2, 0.1)), ValueError),           # low > high
    (dict(rho_range=(-0.1, 0.1)), ValueError),          # negative rate
    (dict(theta_range=(0, float("inf"))), ValueError),
    (dict(theta_range=0.1), TypeError),                 # not a pair
    (dict(theta_range=(0.1,)), ValueError),
    (dict(L_range=(0, 10)), ValueError),                # L < 1
    (dict(L_range=(10.0, 20)), TypeError),              # L must be integer
    (dict(n_sims=0), ValueError),
    (dict(n_sims=5.0), TypeError),
    (dict(chunk_size=0), ValueError),
    (dict(max_workers=0), ValueError),
    (dict(delta=0.5), ValueError),
    (dict(seed=-1), ValueError),
    (dict(save_dir=3), TypeError),
])
def test_bad_inputs(tree10, tmp_path, kwargs, error):
    args = dict(clonal_tree=tree10, n_sims=3, save_dir=tmp_path, delta=10.0,
                max_workers=1, progress=False, **PRIORS)
    args.update(kwargs)
    with pytest.raises(error):
        parallel_sim(**args)
    # Nothing is written when the inputs are rejected
    assert not (tmp_path / PARAMS_FILE).exists()


def test_bad_tree(tmp_path):
    with pytest.raises(TypeError):
        _run("tree", tmp_path)
    with pytest.raises(ValueError):
        _run(ClonalTree(2), tmp_path)


@pytest.mark.parametrize("progress", [True, False])
def test_progress_bar(tree10, tmp_path, capsys, progress):
    _run(tree10, tmp_path, n_sims=3, progress=progress)
    err = capsys.readouterr().err
    assert ("Running simulations" in err) == progress


def test_signature_and_module():
    import inspect
    import types
    import bacnpe
    params = inspect.signature(parallel_sim).parameters
    positional = [name for name, p in params.items()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional == ["clonal_tree", "rho_range", "theta_range", "L_range",
                          "delta", "n_sims", "save_dir"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
               for name, p in params.items() if name not in positional)
    # The module (bacnpe.parallel) is not shadowed by the function
    assert isinstance(bacnpe.parallel, types.ModuleType)
    assert bacnpe.parallel.parallel_sim is bacnpe.parallel_sim
