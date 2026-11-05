"""Sampling the trained posterior for many observed genes."""
from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from .._validation import _check_integer


def _get_posterior(posterior: Any):
    """Accept a posterior, an NPEResult or the dictionary returned by load_npe."""
    if isinstance(posterior, dict):
        if "posterior" not in posterior:
            raise TypeError("A dictionary `posterior` must be the output of load_npe.")
        posterior = posterior["posterior"]
    elif hasattr(posterior, "posterior") and not hasattr(posterior, "sample"):
        posterior = posterior.posterior                       # NPEResult
    if not callable(getattr(posterior, "sample", None)):
        raise TypeError(
            "`posterior` must be an sbi posterior, an NPEResult or the output of load_npe, "
            f"got {type(posterior).__name__}."
        )
    return posterior


def sample_posteriors(
    posterior: Any,
    x_obs,
    num_samples: int = 5000,
    *,
    reject_outside_prior: bool = True,
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Draw posterior samples for each observed gene.

    This is the last part of Algorithm 2 of the paper: for each observed
    gene ``g``, draw `num_samples` parameter vectors from
    ``q(omega | e(z_obs,g))``. The result can be passed to
    :func:`bacnpe.detection.detect_hotspots`, e.g.
    ``detect_hotspots(starts, samples[:, :, 0], ...)`` for the recombination
    rate.

    Parameters
    ----------
    posterior : sbi posterior, NPEResult or dict
        The trained posterior: ``train_npe(...).posterior``, the
        :class:`NPEResult` itself, or the dictionary returned by
        :func:`bacnpe.npe.load_npe`.
    x_obs : array-like, shape (n_genes, n_stats) or (n_stats,)
        Observed summary vectors, one row per gene, with the same columns as
        the training data (``segment_summary_stats`` output, length last).
        A NumPy array, torch tensor or pandas DataFrame.
    num_samples : int, default 5000
        Number of posterior draws per gene (M in the paper).
    reject_outside_prior : bool, default True
        Discard draws outside the prior support, as in Algorithm 2 (and
        resample until `num_samples` draws are kept). This needs a posterior
        trained with a prior (e.g. from :func:`build_prior`). The
        B. cereus and S. aureus notebooks used False, which can give
        non-positive rates.
    seed : int, optional
        Seed for torch's random generator, for reproducible draws. The
        caller's random state is restored afterwards. If None, the current
        global state is used.
    show_progress : bool, default True
        Show a progress bar over genes.

    Returns
    -------
    np.ndarray, shape (n_genes, num_samples, n_params), dtype float64
        ``samples[g, m]`` is draw ``m`` for gene ``g``; with the bacnpe
        prior the parameters are ``(rho_site, theta_site, L)``. Rows of
        `x_obs` with NaN or infinite entries cannot be conditioned on and
        are filled with NaN (with a warning).

    Notes
    -----
    Genes without segregating sites (``prop_segregating_sites == 0``)
    carry almost no information about the rates; the notebooks removed them
    before sampling.
    """
    posterior = _get_posterior(posterior)
    num_samples = _check_integer(num_samples, "num_samples", minimum=1)
    if seed is not None:
        seed = _check_integer(seed, "seed", minimum=0)
    if not isinstance(reject_outside_prior, (bool, np.bool_)):
        raise TypeError("`reject_outside_prior` must be a bool.")

    if hasattr(x_obs, "to_numpy"):                         # pandas DataFrame / Series
        x_obs = x_obs.to_numpy()
    if isinstance(x_obs, torch.Tensor):
        x_obs = x_obs.detach().cpu().numpy()
    try:
        x = np.asarray(x_obs, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError("`x_obs` must be a numeric array.") from None
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"`x_obs` must have shape (n_genes, n_stats), got {np.shape(x_obs)}.")

    valid = np.all(np.isfinite(x), axis=1)
    if not np.any(valid):
        raise ValueError("Every row of `x_obs` contains NaN or inf.")
    if not np.all(valid):
        bad = np.flatnonzero(~valid)
        warnings.warn(
            f"{bad.size} of {len(x)} observations contain NaN or inf and get NaN samples "
            f"(rows {bad[:10].tolist()}{' ...' if bad.size > 10 else ''}).",
            stacklevel=2,
        )

    device = getattr(posterior, "_device", "cpu")
    x_torch = torch.as_tensor(x, dtype=torch.float32, device=device)

    out = None
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        for g in tqdm(range(len(x)), desc="Sampling posteriors", disable=not show_progress):
            if not valid[g]:
                continue
            draws = posterior.sample(
                (num_samples,), x=x_torch[g],
                show_progress_bars=False, reject_outside_prior=bool(reject_outside_prior),
            )
            draws = draws.detach().cpu().numpy().astype(np.float64)
            if out is None:
                out = np.full((len(x), num_samples, draws.shape[-1]), np.nan)
            out[g] = draws
    return out
