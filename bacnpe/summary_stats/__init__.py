"""
Summary statistics of simulated sequence data.

Everything here is an internal helper (underscore prefix) except
:func:`segment_summary_stats`, which computes the 30-entry summary vector
from a clonal tree and a 0/1 sequence matrix, and ``STAT_NAMES``, which
documents those 30 entries (also returned by :func:`bacnpe.ClonalOrigin_seq_sim`).
"""
from .diversity import _tajima_d, _tajima_pi, _watterson_theta
from .homoplasy import _homoplasy_index
from .linkage import _g4_test, _hudson_rm, _kelly_z, _ld, _pairwise_ld_g4
from .regression import _exp_regression, _shifted_exponential
from .segment_stats import STAT_NAMES, segment_summary_stats
from .wall import _wall_bq

__all__ = ["segment_summary_stats", "STAT_NAMES"]
