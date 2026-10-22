"""
bacnpe - simulation of bacterial sequences under the ClonalOrigin model, with
summary statistics for neural posterior estimation (NPE) of local
recombination and mutation rates.

Public API
----------
ClonalTree
    Simulate a clonal genealogy (Kingman coalescent).
ClonalOrigin_seq_sim
    Simulate recombination and mutation on a clonal tree and return a
    30-dimensional vector of summary statistics.

Example
-------
>>> import numpy as np
>>> from bacnpe import ClonalTree, ClonalOrigin_seq_sim
>>> np.random.seed(1)
>>> tree = ClonalTree(20)
>>> stats = ClonalOrigin_seq_sim(tree, rho_site=0.02, theta_site=0.05, L=200, delta=30.0)
>>> stats.shape
(30,)
"""
from .simulation.seq_sim import ClonalOrigin_seq_sim
from .tree.clonal_genealogy import ClonalTree

__version__ = "0.1.0"

__all__ = ["ClonalTree", "ClonalOrigin_seq_sim", "__version__"]
