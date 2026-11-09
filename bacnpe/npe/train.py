"""Training NPE on simulated (theta, x) pairs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sbi.inference import NPE_C
from sbi.neural_nets import posterior_nn
from torch.distributions import Distribution

from .._validation import _check_integer, _check_real


@dataclass
class NPEResult:
    """
    Output of :func:`train_npe`.

    Attributes
    ----------
    posterior : sbi posterior
        The trained posterior. Sample it with
        ``posterior.sample((n,), x=x_obs)``.
    inference : sbi.inference.NPE_C
        The trainer. Use it for loss curves, e.g.
        ``sbi.analysis.plot_summary(result.inference, tags=["training_loss", "validation_loss"])``.
    settings : dict
        Training settings and data counts. :func:`save_npe` stores them in the
        checkpoint.
    """

    posterior: Any
    inference: NPE_C
    settings: Dict[str, Any] = field(default_factory=dict)


def _as_float_tensor(value, name: str, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        arr = np.asarray(value)
        if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
            raise TypeError(f"`{name}` must be numeric, got dtype {arr.dtype}.")
        tensor = torch.as_tensor(arr)
    if tensor.ndim != 2:
        raise ValueError(f"`{name}` must be a 2-D matrix, got {tensor.ndim} dimension(s).")
    return tensor.to(device=device, dtype=torch.float32)


def _drop_invalid(theta: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Drop simulations whose theta or x contain NaN or infinite values."""
    valid = torch.isfinite(x).all(dim=1) & torch.isfinite(theta).all(dim=1)
    return theta[valid], x[valid], int((~valid).sum())


def build_density_estimator(
    embedding_net: Optional[nn.Module] = None,
    model: str = "nsf",
    hidden_features: int = 30,
    num_transforms: int = 8,
    num_bins: int = 5,
):
    """
    Build the sbi density-estimator builder used by :func:`train_npe`.

    The defaults are the flow settings from the original notebooks: a neural
    spline flow (``nsf``) with z-scored ``theta`` and ``x``.

    Parameters
    ----------
    embedding_net : nn.Module, optional
        Embedding network for ``x``. ``None`` (default) uses no embedding, so
        the flow is conditioned on the full (z-scored) ``x``. The notebooks used
        ``LeaveLengthOut_NN(input_dim=30, num_hiddens=128, num_hidden_layers=1,
        num_outputs=2)``.
    model : str, default "nsf"
        sbi density estimator: ``"nsf"``, ``"maf"``, ``"mdn"``, ``"made"``, ...
    hidden_features, num_transforms, num_bins : int
        Passed to :func:`sbi.neural_nets.posterior_nn`.

    Returns
    -------
    callable
        A builder to pass as ``density_estimator`` to ``sbi.inference.NPE_C``.
    """
    if embedding_net is None:
        embedding_net = nn.Identity()
    elif not isinstance(embedding_net, nn.Module):
        raise TypeError("`embedding_net` must be a torch nn.Module or None.")
    return posterior_nn(
        model=model,
        embedding_net=embedding_net,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=num_bins,
        z_score_theta="independent",
        z_score_x="independent",
    )


def train_npe(
    theta,
    x,
    prior: Optional[Distribution] = None,
    *,
    seed: int = 100,
    embedding_net: Optional[nn.Module] = None,
    model: str = "nsf",
    hidden_features: int = 30,
    num_transforms: int = 8,
    num_bins: int = 5,
    training_batch_size: int = 100,
    learning_rate: float = 1e-4,
    max_num_epochs: int = 500,
    stop_after_epochs: int = 20,
    clip_max_norm: Optional[float] = None,
    device: str = "cpu",
    show_progress_bars: bool = True,
) -> NPEResult:
    """
    Train NPE (sbi's ``NPE_C``) on all simulations.

    Simulations with NaN or infinite values (e.g. Tajima's D when there is no
    segregating site) are dropped first. The flow and training defaults match
    the original notebooks. By default there is no prior and no embedding
    network.

    Training and validation losses per epoch are in
    ``result.inference.summary``. To plot them, run
    ``sbi.analysis.plot_summary(result.inference, tags=["training_loss", "validation_loss"])``.

    Parameters
    ----------
    theta : array_like or torch.Tensor, shape (n_sims, 3)
        Simulated parameters ``(rho_site, theta_site, L)``, e.g. the rows of
        ``params.csv`` from :func:`bacnpe.parallel_sim`.
    x : array_like or torch.Tensor, shape (n_sims, 30)
        Summary statistics, e.g. the rows of ``stats.csv``.
    prior : torch Distribution or None, default None
        ``None`` lets sbi use an empirical prior built from `theta`, which is
        what the original notebooks ended up doing. It has no support limits,
        so posterior draws are not restricted to the simulated ranges and
        rates can be non-positive. To restrict them, pass a prior from
        :func:`bacnpe.npe.build_prior` with the ranges used for the
        simulations; the posterior then rejects samples outside its support.
    seed : int, default 100
        Seeds torch and NumPy before the network is built and trained, so the
        network weights are reproducible too. The caller's random states are
        restored afterwards.
    embedding_net : nn.Module, optional
        Embedding network for ``x``. ``None`` (default) trains NPE directly on
        the full ``x``. For the notebook setting pass
        ``LeaveLengthOut_NN(input_dim=30, num_hiddens=128, num_hidden_layers=1, num_outputs=2)``.
        Its weights are initialised when it is created, so create it after
        ``torch.manual_seed`` if they should be reproducible.
    model, hidden_features, num_transforms, num_bins
        Flow settings, see :func:`build_density_estimator`.
    training_batch_size, learning_rate, max_num_epochs, stop_after_epochs, clip_max_norm
        Passed to ``NPE_C.train``.
    device : str, default "cpu"
        Torch device for training.
    show_progress_bars : bool, default True
        Show sbi's training progress.

    Returns
    -------
    NPEResult
        The posterior, the trainer and the settings.

    Raises
    ------
    TypeError, ValueError
        If `theta` / `x` are not numeric 2-D matrices with the same number of
        rows, if `theta` does not have 3 columns, or if a setting is out of range.
    """
    theta = _as_float_tensor(theta, "theta", device)
    x = _as_float_tensor(x, "x", device)
    if theta.shape[0] != x.shape[0]:
        raise ValueError(f"`theta` and `x` must have the same number of rows, "
                         f"got {theta.shape[0]} and {x.shape[0]}.")
    if theta.shape[1] != 3:
        raise ValueError(f"`theta` must have 3 columns (rho_site, theta_site, L), got {theta.shape[1]}.")
    if embedding_net is not None and not isinstance(embedding_net, nn.Module):
        raise TypeError("`embedding_net` must be a torch nn.Module or None.")
    if prior is not None and not isinstance(prior, Distribution):
        raise TypeError("`prior` must be a torch Distribution (see build_prior) or None.")
    seed = _check_integer(seed, "seed", minimum=0)
    training_batch_size = _check_integer(training_batch_size, "training_batch_size", minimum=1)
    learning_rate = _check_real(learning_rate, "learning_rate", minimum=0.0)
    max_num_epochs = _check_integer(max_num_epochs, "max_num_epochs", minimum=1)
    stop_after_epochs = _check_integer(stop_after_epochs, "stop_after_epochs", minimum=1)
    if clip_max_norm is not None:
        clip_max_norm = _check_real(clip_max_norm, "clip_max_norm", minimum=0.0)

    n_total = theta.shape[0]
    theta, x, n_dropped = _drop_invalid(theta, x)
    if theta.shape[0] == 0:
        raise ValueError("No valid simulations left after dropping NaN/inf rows.")

    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            density_estimator = build_density_estimator(
                embedding_net, model=model, hidden_features=hidden_features,
                num_transforms=num_transforms, num_bins=num_bins)
            inference = NPE_C(prior=prior, density_estimator=density_estimator,
                              device=device, show_progress_bars=show_progress_bars)
            estimator = inference.append_simulations(theta, x).train(
                training_batch_size=training_batch_size,
                learning_rate=learning_rate,
                max_num_epochs=max_num_epochs,
                stop_after_epochs=stop_after_epochs,
                clip_max_norm=clip_max_norm,
            )
            posterior = inference.build_posterior(estimator)
        finally:
            np.random.set_state(np_state)

    summary = inference.summary
    settings = dict(
        seed=seed,
        model=model,
        embedding_net=repr(embedding_net) if embedding_net is not None else None,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=num_bins,
        training_batch_size=training_batch_size,
        learning_rate=learning_rate,
        max_num_epochs=max_num_epochs,
        stop_after_epochs=stop_after_epochs,
        clip_max_norm=clip_max_norm,
        prior=repr(prior) if prior is not None else None,
        n_simulations=n_total,
        n_dropped=n_dropped,
        epochs_trained=int(summary["epochs_trained"][-1]),
        best_validation_loss=float(summary["best_validation_loss"][-1]),
    )
    return NPEResult(posterior=posterior, inference=inference, settings=settings)
