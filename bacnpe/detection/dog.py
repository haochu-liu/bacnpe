"""
Difference-of-Gaussians (DoG) hotspot detection for irregularly spaced genes.

Gene-level log rates are treated as a one-dimensional signal on a circular
chromosome. A broad Gaussian smoother (the background) is subtracted from a
narrow Gaussian smoother (the local signal). NPE uncertainty is carried
through by applying the filter to every posterior draw. Distances are the
shortest path around the chromosome, so genes near the coordinate origin
borrow information from genes near the chromosome end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .._validation import _check_integer, _check_real

_EPS = 1e-12

#: Columns of the table returned by :meth:`DoGHotspotResult.to_dataframe`.
HOTSPOT_COLUMNS = [
    "start_position_bp",
    "end_position_bp",
    "n_genes",
    "peak_probability",
    "peak_fold_enrichment",
    "median_probability",
    "wraps_origin",
]


def _weighted_gaussian_matrix(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    bandwidth_bp: float,
    precision: np.ndarray,
    chromosome_length_bp: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a circular, uncertainty-weighted Gaussian smoothing matrix.

    Distances use the shorter of the clockwise and anticlockwise paths:
    ``d(x, y) = min(|x - y|, L - |x - y|)``. Also returns the Kish effective
    number of genes behind each smoothed value.
    """
    linear_distance = np.abs(target_positions[:, None] - source_positions[None, :])
    distance = np.minimum(linear_distance, chromosome_length_bp - linear_distance)
    kernel = np.exp(-0.5 * (distance / bandwidth_bp) ** 2)
    weighted = kernel * precision[None, :]
    support = weighted.sum(axis=1)
    matrix = weighted / np.maximum(support[:, None], _EPS)

    # Kish effective sample size, useful for diagnosing unsupported locations
    effective_n = support ** 2 / np.maximum((weighted ** 2).sum(axis=1), _EPS)
    return matrix, effective_n


@dataclass
class DoGHotspotResult:
    """
    Result of :meth:`DoGHotspotDetector.fit`.

    All per-gene arrays are sorted by genomic position. ``sort_order`` maps
    them back to the input order: element ``i`` of a sorted array belongs to
    input gene ``sort_order[i]``.
    """

    positions_bp: np.ndarray
    posterior_rate_median: np.ndarray
    posterior_rate_lower: np.ndarray
    posterior_rate_upper: np.ndarray
    narrow_rate_median: np.ndarray
    broad_rate_median: np.ndarray
    dog_log_enrichment_median: np.ndarray
    dog_log_enrichment_lower: np.ndarray
    dog_log_enrichment_upper: np.ndarray
    hotspot_probability: np.ndarray
    hotspot_call: np.ndarray
    narrow_effective_n: np.ndarray
    broad_effective_n: np.ndarray
    narrow_bandwidth_bp: float
    broad_bandwidth_bp: float
    chromosome_length_bp: float
    min_fold_enrichment: float
    probability_threshold: float
    sort_order: np.ndarray
    #: Gene end coordinates (sorted like ``positions_bp``), if given to ``fit``
    end_positions_bp: Optional[np.ndarray] = None
    #: Default ``max_gap_bp`` for :meth:`segments`, :meth:`to_dataframe` and :meth:`plot`
    max_gap_bp: Optional[float] = None

    @property
    def local_fold_enrichment_median(self) -> np.ndarray:
        """Median narrow-scale / broad-scale fold enrichment."""
        return np.exp(self.dog_log_enrichment_median)

    def _max_gap(self, max_gap_bp: Optional[float]) -> float:
        if max_gap_bp is None:
            max_gap_bp = self.max_gap_bp
        if max_gap_bp is None:
            max_gap_bp = 2.0 * self.narrow_bandwidth_bp
        max_gap_bp = _check_real(max_gap_bp, "max_gap_bp")
        if max_gap_bp <= 0:
            raise ValueError("max_gap_bp must be positive")
        return max_gap_bp

    def segments(self, max_gap_bp: Optional[float] = None) -> List[dict]:
        """
        Group hotspot-called genes into regions.

        Consecutive called genes are grouped while the distance between
        neighbouring genes is at most `max_gap_bp`. If the last and first genes
        are both called and close enough across the chromosome origin, their
        runs are merged into one segment with ``wraps_origin=True``.

        Parameters
        ----------
        max_gap_bp : float, optional
            Largest gap within a segment. Defaults to the detector's
            ``max_gap_bp``, or twice the narrow bandwidth if that is None.

        Returns
        -------
        list of dict
            One dict per segment with keys ``start_index``, ``end_index``
            (positions in the sorted arrays), ``indices`` (all sorted indices
            in the segment), ``start_position_bp``, ``end_position_bp``
            (positions of the first and last gene), ``n_genes``,
            ``wraps_origin``, ``peak_probability``, ``median_probability`` and
            ``peak_fold_enrichment``.
        """
        max_gap_bp = self._max_gap(max_gap_bp)

        called = self.hotspot_call
        n = len(called)
        if n == 0 or not np.any(called):
            return []

        # Ordinary runs in sorted coordinate order
        runs: List[List[int]] = []
        current: List[int] = []
        for i in range(n):
            if not called[i]:
                if current:
                    runs.append(current)
                    current = []
                continue
            if current:
                gap = self.positions_bp[i] - self.positions_bp[current[-1]]
                if gap > max_gap_bp:
                    runs.append(current)
                    current = []
            current.append(i)
        if current:
            runs.append(current)

        # The last and first genes are neighbours on a circular chromosome
        boundary_gap = self.positions_bp[0] + self.chromosome_length_bp - self.positions_bp[-1]
        if len(runs) >= 2 and called[0] and called[-1] and boundary_gap <= max_gap_bp:
            wrapped = runs[-1] + runs[0]
            runs = [wrapped] + runs[1:-1]

        segments: List[dict] = []
        fold = self.local_fold_enrichment_median
        for indices in runs:
            idx = np.asarray(indices, dtype=int)
            start_index, end_index = int(idx[0]), int(idx[-1])
            segments.append({
                "start_index": start_index,
                "end_index": end_index,
                "indices": idx,
                "start_position_bp": float(self.positions_bp[start_index]),
                "end_position_bp": float(self.positions_bp[end_index]),
                "n_genes": int(len(idx)),
                "wraps_origin": bool(np.any(np.diff(idx) < 0)),
                "peak_probability": float(self.hotspot_probability[idx].max()),
                "median_probability": float(np.median(self.hotspot_probability[idx])),
                "peak_fold_enrichment": float(fold[idx].max()),
            })
        return segments

    def segment_gene_indices(self, max_gap_bp: Optional[float] = None) -> List[np.ndarray]:
        """
        Input-order indices of the genes in each segment.

        Use them to look genes up in the table the positions came from, e.g.
        ``gene_info.iloc[result.segment_gene_indices()[1]]``.
        """
        return [self.sort_order[seg["indices"]] for seg in self.segments(max_gap_bp)]

    def to_dataframe(self, max_gap_bp: Optional[float] = None) -> pd.DataFrame:
        """
        One row per hotspot segment, as stored in the notebooks.

        ``end_position_bp`` is the end coordinate of the segment's last gene
        when gene ends were passed to ``fit``, and that gene's position
        otherwise. For a segment that wraps the origin, ``start_position_bp``
        is larger than ``end_position_bp``.

        Returns
        -------
        pandas.DataFrame
            Columns :data:`HOTSPOT_COLUMNS`, with proper dtypes (empty if
            no hotspot is called).
        """
        rows = []
        for seg in self.segments(max_gap_bp):
            end = (self.end_positions_bp[seg["end_index"]]
                   if self.end_positions_bp is not None else seg["end_position_bp"])
            rows.append({**{k: seg[k] for k in HOTSPOT_COLUMNS}, "end_position_bp": float(end)})
        df = pd.DataFrame(rows, columns=HOTSPOT_COLUMNS)
        return df.astype({"start_position_bp": float, "end_position_bp": float,
                          "n_genes": int, "peak_probability": float,
                          "peak_fold_enrichment": float, "median_probability": float,
                          "wraps_origin": bool})

    def summary(self, max_gap_bp: Optional[float] = None) -> str:
        """The segment report printed in the notebooks, as a string."""
        lines = ["Detected hotspot segments:"]
        for seg in self.segments(max_gap_bp):
            lines.append(
                f"  {seg['start_position_bp'] / 1e6:.2f}-"
                f"{seg['end_position_bp'] / 1e6:.2f} Mb; "
                f"{seg['n_genes']} genes; "
                f"peak P={seg['peak_probability']:.3f}; "
                f"peak enrichment={seg['peak_fold_enrichment']:.2f}x"
            )
        if len(lines) == 1:
            lines.append("  none")
        return "\n".join(lines)

    def plot(self, *args, **kwargs):
        """Plot rates, DoG enrichment and hotspot probability, see :func:`plot_dog_hotspots`."""
        from .plot import plot_dog_hotspots

        return plot_dog_hotspots(self, *args, **kwargs)


class DoGHotspotDetector:
    """
    Detect local rate elevations relative to a broad genomic background.

    Parameters
    ----------
    narrow_bandwidth_bp : float
        Gaussian bandwidth of the local signal. Choose it near the expected
        hotspot radius, not its full width.
    broad_bandwidth_bp : float
        Gaussian bandwidth of the background. Must be larger than the narrow
        bandwidth, commonly by a factor of 4 to 10.
    chromosome_length_bp : float
        Total length of the circular chromosome, used for distances across
        the coordinate origin.
    min_fold_enrichment : float, default 1.5
        A location is enriched in a posterior draw when the narrow smoother
        exceeds the broad smoother by at least this factor (> 1).
    probability_threshold : float, default 0.90
        Minimum fraction of posterior draws that must be enriched to call a
        hotspot (between 0 and 1).
    residual_log_sd : float, default 0.20
        Extra log-rate variability added to each gene's posterior variance
        when building the smoothing weights, so very precise genes do not
        dominate their neighbourhoods.
    min_effective_genes : float, default 1.0
        Locations with fewer effective genes than this in either smoother are
        not called, although their scores are still reported.
    max_posterior_draws : int or None, default 5000
        Randomly subsample to this many draws for speed (None: use all).
    random_state : int or None, default 0
        Seed used only when draws are subsampled.
    max_gap_bp : float, optional
        Default largest gap within a hotspot segment, used by the result's
        ``segments``, ``to_dataframe`` and ``plot``. None means twice the
        narrow bandwidth.
    """

    def __init__(
        self,
        narrow_bandwidth_bp: float,
        broad_bandwidth_bp: float,
        chromosome_length_bp: float,
        min_fold_enrichment: float = 1.5,
        probability_threshold: float = 0.90,
        residual_log_sd: float = 0.20,
        min_effective_genes: float = 1.0,
        max_posterior_draws: Optional[int] = 5000,
        random_state: Optional[int] = 0,
        max_gap_bp: Optional[float] = None,
    ):
        chromosome_length_bp = _check_real(chromosome_length_bp, "chromosome_length_bp")
        narrow_bandwidth_bp = _check_real(narrow_bandwidth_bp, "narrow_bandwidth_bp")
        broad_bandwidth_bp = _check_real(broad_bandwidth_bp, "broad_bandwidth_bp")
        min_fold_enrichment = _check_real(min_fold_enrichment, "min_fold_enrichment")
        probability_threshold = _check_real(probability_threshold, "probability_threshold")
        residual_log_sd = _check_real(residual_log_sd, "residual_log_sd")
        min_effective_genes = _check_real(min_effective_genes, "min_effective_genes")
        if chromosome_length_bp <= 0:
            raise ValueError("chromosome_length_bp must be positive")
        if narrow_bandwidth_bp <= 0:
            raise ValueError("narrow_bandwidth_bp must be positive")
        if broad_bandwidth_bp <= narrow_bandwidth_bp:
            raise ValueError("broad_bandwidth_bp must exceed narrow_bandwidth_bp")
        if min_fold_enrichment <= 1:
            raise ValueError("min_fold_enrichment must exceed 1")
        if not 0 < probability_threshold < 1:
            raise ValueError("probability_threshold must be between 0 and 1")
        if residual_log_sd < 0:
            raise ValueError("residual_log_sd cannot be negative")
        if min_effective_genes <= 0:
            raise ValueError("min_effective_genes must be positive")
        if max_posterior_draws is not None:
            max_posterior_draws = _check_integer(max_posterior_draws, "max_posterior_draws", minimum=2)
        if random_state is not None:
            random_state = _check_integer(random_state, "random_state", minimum=0)
        if max_gap_bp is not None:
            max_gap_bp = _check_real(max_gap_bp, "max_gap_bp")
            if max_gap_bp <= 0:
                raise ValueError("max_gap_bp must be positive")

        self.narrow_bandwidth_bp = narrow_bandwidth_bp
        self.broad_bandwidth_bp = broad_bandwidth_bp
        self.chromosome_length_bp = chromosome_length_bp
        self.min_fold_enrichment = min_fold_enrichment
        self.probability_threshold = probability_threshold
        self.residual_log_sd = residual_log_sd
        self.min_effective_genes = min_effective_genes
        self.max_posterior_draws = max_posterior_draws
        self.random_state = random_state
        self.max_gap_bp = max_gap_bp

    def fit(
        self,
        positions_bp: ArrayLike,
        posterior_draws: ArrayLike,
        rates_are_log: bool = False,
        end_positions_bp: Optional[ArrayLike] = None,
    ) -> DoGHotspotResult:
        """
        Apply the DoG filter to gene-level posterior draws.

        Parameters
        ----------
        positions_bp : array_like, shape (n_genes,)
            One coordinate per gene (unique, within ``[0, chromosome_length_bp]``).
            The notebooks use gene start positions.
        posterior_draws : array_like, shape (n_draws, n_genes)
            NPE posterior draws of one parameter (e.g. ``rho_site``) per gene.
        rates_are_log : bool, default False
            True if the draws are already natural-log rates. Otherwise every
            draw must be positive.
        end_positions_bp : array_like, shape (n_genes,), optional
            Gene end coordinates. Only used for ``end_position_bp`` in
            :meth:`DoGHotspotResult.to_dataframe`.

        Returns
        -------
        DoGHotspotResult
            Arrays sorted by genomic position.
        """
        positions = np.asarray(positions_bp, dtype=float)
        draws = np.asarray(posterior_draws, dtype=float)

        if positions.ndim != 1:
            raise ValueError("positions_bp must be one-dimensional")
        if draws.ndim != 2:
            raise ValueError("posterior_draws must have shape (n_draws, n_genes)")
        if draws.shape[1] != len(positions):
            raise ValueError("posterior_draws columns must match positions_bp")
        if draws.shape[0] < 2:
            raise ValueError("At least two posterior draws are required")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(draws)):
            raise ValueError("positions_bp and posterior_draws must be finite")
        if np.any(positions < 0) or np.any(positions > self.chromosome_length_bp):
            raise ValueError("positions_bp must lie in [0, chromosome_length_bp]")
        if len(np.unique(positions)) != len(positions):
            raise ValueError("Gene positions must be unique")
        if not rates_are_log and np.any(draws <= 0):
            raise ValueError("Rates must be positive when rates_are_log=False")
        ends = None
        if end_positions_bp is not None:
            ends = np.asarray(end_positions_bp, dtype=float)
            if ends.shape != positions.shape:
                raise ValueError("end_positions_bp must have the same length as positions_bp")

        order = np.argsort(positions)
        positions = positions[order]
        draws = draws[:, order]
        if ends is not None:
            ends = ends[order]

        if self.max_posterior_draws is not None and draws.shape[0] > self.max_posterior_draws:
            rng = np.random.default_rng(self.random_state)
            selected = rng.choice(draws.shape[0], size=self.max_posterior_draws, replace=False)
            draws = draws[selected]

        log_draws = draws if rates_are_log else np.log(draws)
        posterior_log_variance = np.var(log_draws, axis=0, ddof=1)
        precision = 1.0 / (posterior_log_variance + self.residual_log_sd ** 2 + _EPS)

        narrow_matrix, narrow_effective_n = _weighted_gaussian_matrix(
            positions, positions, self.narrow_bandwidth_bp, precision, self.chromosome_length_bp)
        broad_matrix, broad_effective_n = _weighted_gaussian_matrix(
            positions, positions, self.broad_bandwidth_bp, precision, self.chromosome_length_bp)

        narrow_log_draws = log_draws @ narrow_matrix.T
        broad_log_draws = log_draws @ broad_matrix.T
        dog_draws = narrow_log_draws - broad_log_draws

        threshold = np.log(self.min_fold_enrichment)
        hotspot_probability = np.mean(dog_draws > threshold, axis=0)
        adequate_support = ((narrow_effective_n >= self.min_effective_genes)
                            & (broad_effective_n >= self.min_effective_genes))
        hotspot_call = (hotspot_probability >= self.probability_threshold) & adequate_support

        rate_draws = np.exp(log_draws)
        rate_q = np.quantile(rate_draws, [0.025, 0.5, 0.975], axis=0)
        dog_q = np.quantile(dog_draws, [0.025, 0.5, 0.975], axis=0)

        return DoGHotspotResult(
            positions_bp=positions,
            posterior_rate_median=rate_q[1],
            posterior_rate_lower=rate_q[0],
            posterior_rate_upper=rate_q[2],
            narrow_rate_median=np.exp(np.median(narrow_log_draws, axis=0)),
            broad_rate_median=np.exp(np.median(broad_log_draws, axis=0)),
            dog_log_enrichment_median=dog_q[1],
            dog_log_enrichment_lower=dog_q[0],
            dog_log_enrichment_upper=dog_q[2],
            hotspot_probability=hotspot_probability,
            hotspot_call=hotspot_call,
            narrow_effective_n=narrow_effective_n,
            broad_effective_n=broad_effective_n,
            narrow_bandwidth_bp=self.narrow_bandwidth_bp,
            broad_bandwidth_bp=self.broad_bandwidth_bp,
            chromosome_length_bp=self.chromosome_length_bp,
            min_fold_enrichment=self.min_fold_enrichment,
            probability_threshold=self.probability_threshold,
            sort_order=order,
            end_positions_bp=ends,
            max_gap_bp=self.max_gap_bp,
        )


def detect_hotspots(
    gene_starts: ArrayLike,
    posterior_samples: ArrayLike,
    chromosome_length_bp: float,
    *,
    gene_ends: Optional[ArrayLike] = None,
    narrow_bandwidth_bp: float = 10_000,
    broad_bandwidth_bp: float = 1_000_000,
    min_fold_enrichment: float = 1.5,
    probability_threshold: float = 0.90,
    residual_log_sd: float = 0.20,
    min_effective_genes: float = 1.0,
    max_posterior_draws: Optional[int] = 5000,
    random_state: Optional[int] = 10,
    max_gap_bp: Optional[float] = 800_000,
    clip_nonpositive: bool = True,
) -> DoGHotspotResult:
    """
    Run DoG hotspot detection on per-gene NPE posterior samples.

    This is the workflow of the hotspots notebooks: non-positive posterior
    samples are replaced by machine epsilon, genes are placed at their start
    positions, and :class:`DoGHotspotDetector` is fitted. The defaults are the
    values used in the notebooks.

    Parameters
    ----------
    gene_starts : array_like, shape (n_genes,)
        Start coordinate of each gene (used as its position).
    posterior_samples : array_like, shape (n_genes, n_samples)
        Posterior samples of one parameter for each gene, e.g.
        ``theta_post[:, :, 0]`` for ``rho_site`` or ``theta_post[:, :, 1]``
        for ``theta_site``. Genes need not be sorted.
    chromosome_length_bp : float
        Length of the circular chromosome.
    gene_ends : array_like, shape (n_genes,), optional
        End coordinate of each gene, used for ``end_position_bp`` in
        :meth:`DoGHotspotResult.to_dataframe`.
    narrow_bandwidth_bp, broad_bandwidth_bp, min_fold_enrichment,
    probability_threshold, residual_log_sd, min_effective_genes,
    max_posterior_draws, random_state, max_gap_bp
        See :class:`DoGHotspotDetector`.
    clip_nonpositive : bool, default True
        Replace samples ``<= 0`` by ``np.finfo(float).eps`` (as in the
        notebooks), so their log is defined. If False, non-positive samples
        raise a `ValueError`.

    Returns
    -------
    DoGHotspotResult
        Use ``.summary()``, ``.plot(...)``, ``.to_dataframe()`` and
        ``.segment_gene_indices()``.
    """
    samples = np.array(posterior_samples, dtype=float)    # copy: the input is never modified
    if samples.ndim != 2:
        raise ValueError("posterior_samples must have shape (n_genes, n_samples)")
    if clip_nonpositive:
        samples[samples <= 0] = np.finfo(np.float64).eps
    detector = DoGHotspotDetector(
        narrow_bandwidth_bp=narrow_bandwidth_bp,
        broad_bandwidth_bp=broad_bandwidth_bp,
        chromosome_length_bp=chromosome_length_bp,
        min_fold_enrichment=min_fold_enrichment,
        probability_threshold=probability_threshold,
        residual_log_sd=residual_log_sd,
        min_effective_genes=min_effective_genes,
        max_posterior_draws=max_posterior_draws,
        random_state=random_state,
        max_gap_bp=max_gap_bp,
    )
    return detector.fit(gene_starts, samples.T, end_positions_bp=gene_ends)
