"""
Input validation helpers shared by the public entry points.

These helpers convert user input to plain Python ``int`` / ``float`` values so
that the simulation code always works with the same data types, and raise
informative errors for unpermitted inputs:

* ``TypeError``  - the input has the wrong type (e.g. a string, a float where
  an integer is required, or a ``bool``).
* ``ValueError`` - the input has the right type but an unpermitted value
  (e.g. negative, NaN, infinite or below a minimum).
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Iterable
from typing import Optional, Tuple

import numpy as np


def _check_integer(value, name: str, minimum: Optional[int] = None) -> int:
    """
    Validate an integer input and return it as a Python ``int``.

    Parameters
    ----------
    value : Any
        The value to check. Python ``int`` and NumPy integer scalars are
        accepted; ``bool`` and floats (even ``3.0``) are rejected.
    name : str
        Name of the argument, used in error messages.
    minimum : int, optional
        Smallest permitted value (inclusive).

    Returns
    -------
    int
        The validated value.
    """
    # bool is a subclass of int in Python, but True/False is never a sensible
    # sample size or sequence length, so reject it explicitly.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"`{name}` must be a single integer, got {type(value).__name__}.")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}, got {value}.")
    return value


def _check_real(value, name: str, minimum: Optional[float] = None) -> float:
    """
    Validate a real-valued input and return it as a Python ``float``.

    Parameters
    ----------
    value : Any
        The value to check. Python/NumPy integers and floats are accepted;
        ``bool``, strings, arrays, NaN and infinite values are rejected.
    name : str
        Name of the argument, used in error messages.
    minimum : float, optional
        Smallest permitted value (inclusive).

    Returns
    -------
    float
        The validated value.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"`{name}` must be a single real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"`{name}` must be finite, got {value}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}, got {value}.")
    return value


def _check_binary_matrix(mat, name: str = "mat") -> np.ndarray:
    """
    Validate a 2-D incidence matrix and return it as a boolean array.

    Parameters
    ----------
    mat : array_like
        A 2-D matrix containing only 0/1 or True/False values
        (rows are sequences, columns are sites).
    name : str
        Name of the argument, used in error messages.

    Returns
    -------
    np.ndarray
        The matrix as a ``bool`` array of the same shape.
    """
    arr = np.asarray(mat)
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be a 2-D matrix, got {arr.ndim} dimension(s).")
    if arr.dtype == bool:
        return arr
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise TypeError(f"`{name}` must contain booleans or 0/1 numbers, got dtype {arr.dtype}.")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"`{name}` must contain only 0/1 (or True/False) values.")
    return arr.astype(bool)


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
