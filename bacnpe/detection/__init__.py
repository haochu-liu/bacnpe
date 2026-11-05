"""
Hotspot detection from per-gene NPE posteriors.

Public API
----------
detect_hotspots
    The notebook workflow in one call: clip non-positive samples, fit the
    Difference-of-Gaussians detector, return a :class:`DoGHotspotResult`.
DoGHotspotDetector, DoGHotspotResult
    The detector and its result (``segments``, ``summary``, ``to_dataframe``,
    ``segment_gene_indices``, ``plot``).
plot_dog_hotspots
    The three-panel plot (also available as ``result.plot(...)``).
save_hotspot_tables
    Save hotspot tables to an Excel workbook or CSV files.

Example
-------
>>> from bacnpe.detection import detect_hotspots, save_hotspot_tables
>>> recomb = detect_hotspots(gene_starts, theta_post[:, :, 0], 5_227_440,
...                          gene_ends=gene_ends)                         # doctest: +SKIP
>>> print(recomb.summary())                                               # doctest: +SKIP
>>> recomb.plot(rate_label="Recombination rate")                          # doctest: +SKIP
>>> save_hotspot_tables({"Recombination": recomb.to_dataframe()},
...                     "bacillus_hotspots.xlsx")                         # doctest: +SKIP
"""
from .dog import HOTSPOT_COLUMNS, DoGHotspotDetector, DoGHotspotResult, detect_hotspots
from .export import save_hotspot_tables
from .plot import plot_dog_hotspots

__all__ = [
    "detect_hotspots",
    "DoGHotspotDetector",
    "DoGHotspotResult",
    "HOTSPOT_COLUMNS",
    "plot_dog_hotspots",
    "save_hotspot_tables",
]
