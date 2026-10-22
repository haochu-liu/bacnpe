"""Shared fixtures for the bacnpe test suite."""
import numpy as np
import pytest

from bacnpe import ClonalTree


@pytest.fixture(autouse=True)
def _seed():
    """Seed NumPy's global random state before every test (reproducible tests)."""
    np.random.seed(12345)


@pytest.fixture
def tree10():
    """A clonal tree with 10 leaves."""
    return ClonalTree(10)
