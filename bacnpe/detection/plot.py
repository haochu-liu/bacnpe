"""Plotting DoG hotspot detection results."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from .dog import DoGHotspotResult


def plot_dog_hotspots(
    result: "DoGHotspotResult",
    rate_label: str = "Estimated rate",
    title_text: str = "Difference-of-Gaussians Hotspot Detection",
    title_fontsize: float = 16.0,
    position_unit_bp: float = 1e6,
    position_unit_name: str = "Mb",
    max_gap_bp: Optional[float] = None,
    hotspot_padding_bp: float = 10_000,
    rate_axis_log: bool = False,
    figsize: Tuple[float, float] = (14.0, 9.0),
    save_path: Optional[Union[str, os.PathLike]] = None,
    format: str = "pdf",
):
    """
    Plot gene rates, DoG fold enrichment and hotspot probability.

    Three panels share the position axis, and called hotspot segments are
    shaded on all of them:

    1. Per-gene posterior medians with 95% intervals (red = hotspot), plus
       the narrow and broad smoothers.
    2. Median local / broad fold enrichment with its 95% interval, and the
       minimum enrichment threshold.
    3. Probability that the enrichment exceeds the threshold, and the
       probability threshold.

    Parameters
    ----------
    result : DoGHotspotResult
        Output of :meth:`DoGHotspotDetector.fit` or :func:`detect_hotspots`.
    rate_label : str
        y-axis label of the top panel, e.g. ``"Recombination rate"``.
    title_text : str
        Figure title.
    title_fontsize : float, default 16
        Font size of the title.
    position_unit_bp : float, default 1e6
        Base pairs per unit on the position axis.
    position_unit_name : str, default "Mb"
        Name of that unit, shown in the axis label.
    max_gap_bp : float, optional
        Largest gap within a shaded segment. Defaults to ``result.max_gap_bp``.
    hotspot_padding_bp : float, default 10000
        Padding added on both sides of each shaded segment.
    rate_axis_log : bool, default False
        Use a log scale for the rate axis.
    figsize : (float, float), default (14, 9)
        Figure size in inches.
    save_path : str or path-like, optional
        If given, the figure is saved there.
    format : str, default "pdf"
        File format used with `save_path`.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray of Axes)
    """
    import matplotlib.pyplot as plt

    if position_unit_bp <= 0:
        raise ValueError("position_unit_bp must be positive")

    x = result.positions_bp / position_unit_bp
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3.0, 2.1, 1.8], "hspace": 0.08},
    )
    ax_rate, ax_dog, ax_prob = axes
    legend_kw = dict(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
                     borderaxespad=0.0, fontsize=9)

    # Shade called regions on all panels, including regions crossing the origin
    chromosome_end = result.chromosome_length_bp / position_unit_bp
    padding = hotspot_padding_bp / position_unit_bp
    for segment in result.segments(max_gap_bp=max_gap_bp):
        i0, i1 = segment["start_index"], segment["end_index"]
        left = max(0.0, x[i0] - padding)
        right = min(chromosome_end, x[i1] + padding)
        spans = [(left, chromosome_end), (0.0, right)] if segment["wraps_origin"] else [(left, right)]
        for span_left, span_right in spans:
            for ax in axes:
                ax.axvspan(span_left, span_right, color="#d62728", alpha=0.10, lw=0)

    # Panel 1: gene rates and the two smoothers
    colors = np.where(result.hotspot_call, "#d62728", "#4c78a8")
    ax_rate.vlines(x, result.posterior_rate_lower, result.posterior_rate_upper,
                   color=colors, alpha=0.25, zorder=1, linewidth=0.5)
    ax_rate.scatter(x, result.posterior_rate_median, c=colors, s=18, edgecolor="white",
                    alpha=0.8, linewidth=0.25, zorder=2, label="Gene posterior median")
    ax_rate.plot(x, result.narrow_rate_median, color="#e68613", linewidth=1.5, zorder=4,
                 label=f"Narrow smoother ({result.narrow_bandwidth_bp / 1e3:g} kb)")
    ax_rate.plot(x, result.broad_rate_median, color="#222222", linewidth=1.5, zorder=3,
                 label=f"Broad smoother ({result.broad_bandwidth_bp / 1e3:g} kb)")
    if rate_axis_log:
        ax_rate.set_yscale("log")
    ax_rate.set_ylabel(rate_label)
    ax_rate.set_title(title_text, fontsize=title_fontsize)
    ax_rate.grid(alpha=0.2)
    ax_rate.legend(**legend_kw)

    # Panel 2: fold enrichment
    ax_dog.fill_between(x, np.exp(result.dog_log_enrichment_lower),
                        np.exp(result.dog_log_enrichment_upper),
                        color="#7f7f7f", alpha=0.22, zorder=1, label="95% interval")
    ax_dog.plot(x, np.exp(result.dog_log_enrichment_median), color="#7b3294",
                linewidth=1.6, zorder=2, label="Median fold enrichment")
    ax_dog.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0, zorder=3,
                   label="No enrichment")
    ax_dog.axhline(result.min_fold_enrichment, color="#d62728", linestyle="--", linewidth=1.2,
                   zorder=4, label=f"Minimum enrichment = {result.min_fold_enrichment:g}x")
    ax_dog.set_ylabel("Local / broad\nfold enrichment")
    ax_dog.grid(alpha=0.2)
    ax_dog.legend(**legend_kw)

    # Panel 3: hotspot probability
    ax_prob.plot(x, result.hotspot_probability, color="#222222", linewidth=1.4, zorder=1)
    ax_prob.scatter(x, result.hotspot_probability, c=colors, s=18, linewidth=0.25, zorder=2)
    ax_prob.axhline(result.probability_threshold, color="#d62728", linestyle="--", linewidth=1.2,
                    zorder=3, label=f"Threshold = {result.probability_threshold:g}")
    ax_prob.set_ylim(-0.03, 1.03)
    ax_prob.set_ylabel("P(enrichment\nexceeds threshold)")
    ax_prob.set_xlabel(f"Chromosome position ({position_unit_name})")
    ax_prob.grid(alpha=0.2)
    ax_prob.legend(**legend_kw)
    ax_prob.set_xlim(0.0, chromosome_end)

    # Reserve the right side of the figure for the legends
    fig.subplots_adjust(left=0.10, right=0.76, bottom=0.08, top=0.94, hspace=0.08)
    fig.align_ylabels(axes)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", format=format)
    return fig, axes
