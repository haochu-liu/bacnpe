"""
Neural posterior estimation (NPE) of (rho_site, theta_site, L) with sbi.

Uses PyTorch and sbi. ``import bacnpe`` does not import this subpackage, so
torch is only loaded when NPE is used.

Public API
----------
build_prior
    Optional prior on (rho_site, theta_site, L), using the ranges from the
    simulations; pass it to ``train_npe`` to restrict posterior draws to them.
train_npe
    Train NPE on all simulations and return an :class:`NPEResult`.
build_density_estimator
    The density estimator used by ``train_npe`` (for custom sbi workflows).
save_npe, load_npe
    Save / load a trained posterior with its settings.
sample_posteriors
    Posterior samples for every observed gene, shape (n_genes, M, 3).
DiscreteUniform, LeaveLengthOut_NN
    The prior on L and an optional embedding network for ``train_npe``.

Example
-------
>>> import numpy as np
>>> from bacnpe.npe import train_npe, save_npe
>>> theta = np.loadtxt("sims/params.csv", delimiter=",")   # doctest: +SKIP
>>> x = np.loadtxt("sims/stats.csv", delimiter=",")        # doctest: +SKIP
>>> result = train_npe(theta, x, seed=100)                 # doctest: +SKIP
>>> save_npe(result, "trained_npe_posterior.pt")           # doctest: +SKIP
"""
from .checkpoint import load_npe, save_npe
from .distributions import DiscreteUniform
from .embedding import LeaveLengthOut_NN
from .prior import build_prior
from .sample import sample_posteriors
from .train import NPEResult, build_density_estimator, train_npe

__all__ = [
    "build_prior",
    "train_npe",
    "NPEResult",
    "build_density_estimator",
    "save_npe",
    "load_npe",
    "sample_posteriors",
    "DiscreteUniform",
    "LeaveLengthOut_NN",
]
