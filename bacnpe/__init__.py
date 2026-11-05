"""
bacnpe - simulation of bacterial sequences under the ClonalOrigin model, and then train
a neural posterior estimation (NPE) model with simulation to detect local recombination and mutation hotspots.

Public API
----------
ClonalTree
    Simulate a clonal genealogy (Kingman coalescent).
ARG
    Simulate recombination on a clonal genealogy, giving an approximated
    ancestral recombination graph (ClonalOrigin model).
add_mutation
    Add mutations to an ARG and return the allelic state of every node.
ClonalOrigin_seq_sim
    Simulate recombination and mutation on a approximated ancestral recombination graph (ARG)
    based on a clonal genealogy, and return a 30-dimensional vector of summary statistics.
parallel_sim
    Draw (rho, theta, L) from uniform priors and run ClonalOrigin_seq_sim in
    parallel, saving parameters and summary statistics to CSV files.
segment_summary_stats
    Compute the 30-dimensional vector of summary statistics from a clonal
    genealogy and a 0/1 sequence matrix (rows are the leaves).

Example
-------
>>> import numpy as np
>>> from bacnpe import ARG, ClonalTree, ClonalOrigin_seq_sim, add_mutation, segment_summary_stats
>>> np.random.seed(1)
>>> tree = ClonalTree(20)
>>> stats = ClonalOrigin_seq_sim(tree, rho_site=0.02, theta_site=0.05, L=200, delta=30.0)
>>> stats.shape
(30,)
>>> arg = ARG(tree, rho_site=0.02, L=200, delta=30.0)
>>> node_site = add_mutation(arg, theta_site=0.05)
>>> node_site.shape[1]
200
>>> segment_summary_stats(tree, node_site[:tree.n]).shape
(30,)
"""
from .parallel import parallel_sim
from .simulation.arg import ARG
from .simulation.mutation import add_mutation
from .simulation.seq_sim import ClonalOrigin_seq_sim
from .summary_stats.segment_stats import segment_summary_stats
from .tree.clonal_genealogy import ClonalTree

__version__ = "0.3.0"

__all__ = ["ClonalTree", "ARG", "add_mutation", "ClonalOrigin_seq_sim",
           "parallel_sim", "segment_summary_stats", "__version__"]
