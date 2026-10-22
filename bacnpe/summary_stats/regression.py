"""Exponential decay fit of r^2 against the distance between sites."""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


def _shifted_exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Shifted exponential model ``a * exp(-b * x) + c``.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    a : float
        Amplitude.
    b : float
        Growth/decay rate.
    c : float
        Vertical shift (asymptote).

    Returns
    -------
    np.ndarray
        Model values at `x`.
    """
    return a * np.exp(-b * x) + c


def _exp_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit an exponential regression model to the given data.

    The model is defined as:
        y = a * exp(-b * (x - min(x))) + c

    Parameters
    ----------
    x : numpy array
        Independent variable data.
    y : numpy array
        Dependent variable data.

    Returns
    -------
    tuple
        Optimal parameters (a, b, c) for the fitted model.

    Raises
    ------
    RuntimeError
        If the least-squares optimisation does not converge.
    """
    initial_guess = [1.0, 1.0, np.min(y)]
    x_scaled = x - np.min(x)

    # Silence the optimiser's warnings (e.g. covariance could not be
    # estimated, overflow in exp during the search) - they do not affect
    # the fitted parameters.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        popt, _ = curve_fit(
            _shifted_exponential,
            x_scaled,
            y,
            p0=initial_guess
        )
    return tuple(popt)
