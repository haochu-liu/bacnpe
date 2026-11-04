"""
Run many ClonalOrigin simulations in parallel to build an NPE training set.

Parameters are drawn from uniform priors and each draw is simulated with
:func:`bacnpe.ClonalOrigin_seq_sim` in a pool of worker processes. Results
are written to disk chunk by chunk, so a long run keeps its finished chunks
if it is interrupted.

Every simulation gets its own seed derived from ``seed``. The results
therefore depend only on ``seed``, not on ``max_workers``, ``chunk_size`` or
how the operating system schedules the workers.

Note (Windows / macOS): worker processes are started with "spawn", so a
script that calls :func:`parallel_sim` with ``max_workers > 1`` must
guard the call with ``if __name__ == "__main__":``.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from ._validation import _check_integer, _check_real
from .simulation.seq_sim import ClonalOrigin_seq_sim
from .summary_stats.segment_stats import STAT_NAMES
from .tree.clonal_genealogy import ClonalTree

#: Column names of the parameter file written by :func:`parallel_sim`.
PARAM_NAMES = ["rho_site", "theta_site", "L"]

PARAMS_FILE = "params.csv"
STATS_FILE = "stats.csv"


def _check_range(value, name: str, minimum: float, integer: bool = False) -> Tuple:
    """Validate a prior range ``(low, high)`` with ``minimum <= low <= high``."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"`{name}` must be a pair (low, high), got {type(value).__name__}.")
    value = tuple(value)
    if len(value) != 2:
        raise ValueError(f"`{name}` must be a pair (low, high), got {len(value)} value(s).")
    check = _check_integer if integer else _check_real
    low = check(value[0], f"{name}[0]", minimum=minimum)
    high = check(value[1], f"{name}[1]", minimum=minimum)
    if low > high:
        raise ValueError(f"`{name}` must satisfy low <= high, got ({low}, {high}).")
    return low, high


def _simulate_one(task: Tuple[float, float, int, int], tree: ClonalTree,
                  delta: float) -> np.ndarray:
    """
    Run one simulation with its own seed (executed in a worker process).

    NumPy's global random state is restored afterwards, so running in the
    main process (``max_workers=1``) does not change the caller's state.
    """
    rho_site, theta_site, L, sim_seed = task
    state = np.random.get_state()
    try:
        np.random.seed(sim_seed)
        return ClonalOrigin_seq_sim(tree, rho_site, theta_site, L, delta)
    finally:
        np.random.set_state(state)


def parallel_sim(
    clonal_tree: ClonalTree,
    rho_range: Tuple[float, float],
    theta_range: Tuple[float, float],
    L_range: Tuple[int, int],
    delta: float,
    n_sims: int,
    save_dir: Union[str, os.PathLike],
    *,
    chunk_size: int = 100,
    max_workers: Optional[int] = None,
    seed: Optional[int] = None,
    overwrite: bool = False,
    progress: bool = True,
) -> Tuple[Path, Path]:
    """
    Simulate summary statistics for parameters drawn from uniform priors.

    For each of `n_sims` simulations, ``rho_site``, ``theta_site`` and ``L``
    are drawn independently from their priors. Then
    ``ClonalOrigin_seq_sim(clonal_tree, rho_site, theta_site, L, delta)`` is
    run in parallel worker processes.

    Two CSV files are written to `save_dir`. Row ``i`` of each file belongs
    to simulation ``i``:

    * ``params.csv`` has ``n_sims`` rows and columns ``rho_site, theta_site, L``.
      It is written before the simulations start.
    * ``stats.csv`` has ``n_sims`` rows and 30 columns, named in
      ``bacnpe.summary_stats.STAT_NAMES``. It is filled in `chunk_size` rows
      at a time.

    Each file starts with ``#`` comment lines: the column names and the
    seed. ``np.loadtxt(path, delimiter=",")`` skips them, and so does
    ``pandas.read_csv(path, comment="#", header=None)``.

    Parameters
    ----------
    clonal_tree : ClonalTree
        The clonal genealogy (at least 3 leaves), shared by all simulations.
    rho_range : (float, float)
        ``(low, high)`` of the uniform prior on the recombination rate per
        site (``0 <= low <= high``).
    theta_range : (float, float)
        ``(low, high)`` of the uniform prior on the mutation rate per site
        (``0 <= low <= high``).
    L_range : (int, int)
        ``(low, high)`` of the uniform prior on the number of sites. Both ends
        are included (``1 <= low <= high``).
    delta : float
        Mean recombinant segment length (>= 1), the same for all simulations.
    n_sims : int
        Number of simulations (>= 1).
    save_dir : str or path-like
        Directory for ``params.csv`` and ``stats.csv``. It is created if it
        does not exist.
    chunk_size : int, default 100
        Number of simulations run and written to ``stats.csv`` at a time. A
        smaller chunk saves results more often but leaves workers idle more
        often at chunk boundaries.
    max_workers : int, optional
        Number of worker processes. Defaults to the number of CPUs.
        ``max_workers=1`` runs everything in the current process (useful for
        debugging) and gives exactly the same results.
    seed : int, optional
        Seed for the prior draws and the simulations. ``None`` draws a fresh
        seed. The seed used is written in the header of both files.
    overwrite : bool, default False
        Whether existing ``params.csv`` / ``stats.csv`` in `save_dir` may be
        replaced. If False, a `FileExistsError` is raised instead.
    progress : bool, default True
        Whether to show a ``tqdm`` progress bar.

    Returns
    -------
    (Path, Path)
        Paths of ``params.csv`` and ``stats.csv``.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If an argument is out of range.
    FileExistsError
        If an output file already exists and `overwrite` is False.

    Examples
    --------
    >>> from bacnpe import ClonalTree, parallel_sim
    >>> if __name__ == "__main__":                       # doctest: +SKIP
    ...     tree = ClonalTree(27)
    ...     params_path, stats_path = parallel_sim(
    ...         tree, rho_range=(0, 0.2), theta_range=(0, 0.1),
    ...         L_range=(400, 10000), delta=542.0, n_sims=10000,
    ...         save_dir="sims", chunk_size=100, max_workers=12, seed=1)
    """
    # ---- input validation (everything is checked before any work starts)
    if not isinstance(clonal_tree, ClonalTree):
        raise TypeError("`clonal_tree` must be of class 'ClonalTree'.")
    if clonal_tree.n < 3:
        raise ValueError("`clonal_tree` must have at least 3 leaves.")
    rho_low, rho_high = _check_range(rho_range, "rho_range", minimum=0.0)
    theta_low, theta_high = _check_range(theta_range, "theta_range", minimum=0.0)
    L_low, L_high = _check_range(L_range, "L_range", minimum=1, integer=True)
    delta = _check_real(delta, "delta", minimum=1.0)
    n_sims = _check_integer(n_sims, "n_sims", minimum=1)
    chunk_size = _check_integer(chunk_size, "chunk_size", minimum=1)
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = _check_integer(max_workers, "max_workers", minimum=1)
    if seed is not None:
        seed = _check_integer(seed, "seed", minimum=0)
    if not isinstance(save_dir, (str, os.PathLike)):
        raise TypeError(f"`save_dir` must be a path, got {type(save_dir).__name__}.")

    save_dir = Path(save_dir)
    params_path = save_dir / PARAMS_FILE
    stats_path = save_dir / STATS_FILE
    if not overwrite:
        for path in (params_path, stats_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it).")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- draw parameters from the priors, and one seed per simulation
    root = np.random.SeedSequence(seed)
    prior_seq, sim_seq = root.spawn(2)
    rng = np.random.default_rng(prior_seq)
    rho = rng.uniform(rho_low, rho_high, size=n_sims)
    theta = rng.uniform(theta_low, theta_high, size=n_sims)
    L = rng.integers(L_low, L_high, size=n_sims, endpoint=True)
    sim_seeds = sim_seq.generate_state(n_sims, dtype=np.uint32)

    seed_line = f"seed={root.entropy}"
    np.savetxt(params_path, np.column_stack((rho, theta, L)), delimiter=",",
               header=",".join(PARAM_NAMES) + "\n" + seed_line)

    tasks = [(float(r), float(t), int(l), int(s))
             for r, t, l, s in zip(rho, theta, L, sim_seeds)]
    run_one = partial(_simulate_one, tree=clonal_tree, delta=delta)

    # ---- run the simulations chunk by chunk, appending results to stats.csv
    pbar = tqdm(total=n_sims, desc="Running simulations", disable=not progress)
    executor = ProcessPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    try:
        with open(stats_path, "w") as f:
            np.savetxt(f, np.empty((0, len(STAT_NAMES))), delimiter=",",
                       header=",".join(STAT_NAMES) + "\n" + seed_line)
            for start in range(0, n_sims, chunk_size):
                chunk = tasks[start:start + chunk_size]
                results = executor.map(run_one, chunk) if executor else map(run_one, chunk)
                rows = []
                for row in results:
                    rows.append(row)
                    pbar.update(1)
                np.savetxt(f, np.vstack(rows), delimiter=",")
                f.flush()
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
        pbar.close()

    return params_path, stats_path
