"""Saving hotspot tables."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Union

import pandas as pd


def save_hotspot_tables(
    tables: Mapping[str, pd.DataFrame],
    path: Union[str, os.PathLike],
    overwrite: bool = False,
) -> Path:
    """
    Save hotspot tables, e.g. one for recombination and one for mutation.

    Parameters
    ----------
    tables : mapping of str to DataFrame
        Name -> table, e.g.
        ``{"Recombination": recomb.to_dataframe(), "Mutation": mutate.to_dataframe()}``.
    path : str or path-like
        * ending in ``.xlsx``: one Excel workbook with one sheet per table
          (needs ``openpyxl``), as in the notebooks;
        * anything else: a directory with one ``<name>.csv`` per table.
    overwrite : bool, default False
        Whether existing files may be replaced. If False, a `FileExistsError`
        is raised instead.

    Returns
    -------
    Path
        The workbook or directory written.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"`path` must be a path, got {type(path).__name__}.")
    if not tables:
        raise ValueError("`tables` is empty.")
    for name, df in tables.items():
        if not isinstance(name, str) or not name:
            raise TypeError("Table names must be non-empty strings.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Table {name!r} must be a pandas DataFrame, got {type(df).__name__}.")

    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it).")
        bad = [n for n in tables if len(n) > 31 or any(c in n for c in "[]:*?/\\")]
        if bad:
            raise ValueError(f"Invalid Excel sheet names (max 31 characters, none of []:*?/\\): {bad}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path) as writer:
            for name, df in tables.items():
                df.to_excel(writer, sheet_name=name, index=False)
        return path

    targets = {name: path / f"{name}.csv" for name in tables}
    if not overwrite:
        existing = [str(p) for p in targets.values() if p.exists()]
        if existing:
            raise FileExistsError(f"{existing} already exist (pass overwrite=True to replace them).")
    path.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(targets[name], index=False)
    return path
