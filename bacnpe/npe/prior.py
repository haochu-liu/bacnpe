"""The NPE prior on (rho_site, theta_site, L)."""
from __future__ import annotations

from typing import Tuple

import torch
from sbi.utils import MultipleIndependent
from torch.distributions import Uniform

from .._validation import _check_range
from .distributions import DiscreteUniform


def build_prior(
    rho_range: Tuple[float, float],
    theta_range: Tuple[float, float],
    L_range: Tuple[int, int],
    device: str = "cpu",
) -> MultipleIndependent:
    """
    Build the prior on ``(rho_site, theta_site, L)`` for NPE.

    ``rho_site`` and ``theta_site`` are continuous uniform and ``L`` is
    uniform on the integers ``L_range[0]..L_range[1]`` (both ends included).
    The arguments mean the same as in :func:`bacnpe.parallel_sim`, so pass the
    ranges used for the simulations.

    Parameters
    ----------
    rho_range : (float, float)
        ``(low, high)`` of the recombination rate per site, ``0 <= low < high``.
    theta_range : (float, float)
        ``(low, high)`` of the mutation rate per site, ``0 <= low < high``.
    L_range : (int, int)
        ``(low, high)`` of the number of sites, ``1 <= low <= high``.
    device : str, default "cpu"
        Torch device of the prior.

    Returns
    -------
    sbi.utils.MultipleIndependent
        A 3-dimensional prior. Pass it to :func:`bacnpe.npe.train_npe`.

    Raises
    ------
    TypeError, ValueError
        If a range is not a valid ``(low, high)`` pair. A continuous range
        needs ``low < high``.
    """
    rho_low, rho_high = _check_range(rho_range, "rho_range", minimum=0.0)
    theta_low, theta_high = _check_range(theta_range, "theta_range", minimum=0.0)
    L_low, L_high = _check_range(L_range, "L_range", minimum=1, integer=True)
    for name, (low, high) in (("rho_range", (rho_low, rho_high)),
                              ("theta_range", (theta_low, theta_high))):
        if low == high:
            raise ValueError(f"`{name}` must satisfy low < high for a continuous prior.")

    dists = [
        Uniform(low=torch.tensor([rho_low]), high=torch.tensor([rho_high])),
        Uniform(low=torch.tensor([theta_low]), high=torch.tensor([theta_high])),
        DiscreteUniform(low=torch.tensor([float(L_low)]), high=torch.tensor([float(L_high)])),
    ]
    return MultipleIndependent(dists=dists, validate_args=False, device=device)
