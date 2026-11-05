"""Saving and loading trained NPE posteriors."""
from __future__ import annotations

import os
import pickle
import types
import warnings
from pathlib import Path
from typing import Any, Dict, Union

import sbi
import torch

from .train import NPEResult

# Classes pickled by the original scripts under their old top-level module
# names (``sys.path.append('../pysimARG')``), mapped to their bacnpe homes.
_LEGACY_CLASSES = {
    ("LeaveLengthOut_NN", "LeaveLengthOut_NN"): ("bacnpe.npe.embedding", "LeaveLengthOut_NN"),
    ("discrete_uniform", "DiscreteUniform"): ("bacnpe.npe.distributions", "DiscreteUniform"),
}


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module, name = _LEGACY_CLASSES.get((module, name), (module, name))
        return super().find_class(module, name)


# torch.load takes a "pickle module": anything with an Unpickler and a load
_legacy_pickle = types.ModuleType("_bacnpe_legacy_pickle")
_legacy_pickle.Unpickler = _LegacyUnpickler
_legacy_pickle.load = lambda f, **kw: _LegacyUnpickler(f, **kw).load()


def save_npe(result: Union[NPEResult, Any], path: Union[str, os.PathLike],
             overwrite: bool = False, **metadata) -> Path:
    """
    Save a trained posterior with its settings and library versions.

    The checkpoint is a dict saved with :func:`torch.save`, holding
    ``"posterior"``, ``"sbi_version"``, ``"torch_version"``,
    ``"bacnpe_version"``, the training settings (when `result` is an
    :class:`NPEResult`) and any extra `metadata`. ``checkpoint["posterior"]``
    works as in the original notebooks.

    Parameters
    ----------
    result : NPEResult or sbi posterior
        The output of :func:`bacnpe.npe.train_npe`, or a posterior on its own.
    path : str or path-like
        File to write, usually ending in ``.pt``. Parent directories are created.
    overwrite : bool, default False
        Whether an existing file may be replaced. If False, a
        `FileExistsError` is raised instead.
    **metadata
        Extra entries to store, e.g. ``data="bacillus"``. They override the
        training settings with the same name.

    Returns
    -------
    Path
        The path written.
    """
    from .. import __version__ as bacnpe_version

    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"`path` must be a path, got {type(path).__name__}.")
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it).")

    if isinstance(result, NPEResult):
        posterior, settings = result.posterior, dict(result.settings)
    else:
        posterior, settings = result, {}
    reserved = {"posterior", "sbi_version", "torch_version", "bacnpe_version"}
    clash = reserved & set(metadata)
    if clash:
        raise ValueError(f"These metadata names are reserved: {sorted(clash)}.")

    checkpoint: Dict[str, Any] = {
        "posterior": posterior,
        "sbi_version": sbi.__version__,
        "torch_version": torch.__version__,
        "bacnpe_version": bacnpe_version,
        **settings,
        **metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def load_npe(path: Union[str, os.PathLike], map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load a checkpoint written by :func:`save_npe` or by the original notebooks.

    Checkpoints from the original notebooks refer to ``LeaveLengthOut_NN`` and
    ``DiscreteUniform`` by their old module names. These are mapped to the
    classes in :mod:`bacnpe.npe`, so old checkpoints load without the
    ``pysimARG`` folder.

    A warning is shown if the checkpoint was saved with a different sbi
    version.

    .. warning::
       Loading unpickles Python objects (``weights_only=False``). Only load
       files you trust.

    Parameters
    ----------
    path : str or path-like
        The checkpoint file.
    map_location : str, default "cpu"
        Passed to :func:`torch.load`.

    Returns
    -------
    dict
        The checkpoint. The posterior is ``checkpoint["posterior"]``.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False,
                            pickle_module=_legacy_pickle)
    if not isinstance(checkpoint, dict) or "posterior" not in checkpoint:
        raise ValueError(f"{path} is not an NPE checkpoint (no 'posterior' entry).")
    saved = checkpoint.get("sbi_version")
    if saved is not None and saved != sbi.__version__:
        warnings.warn(f"Checkpoint was saved with sbi {saved}, but sbi {sbi.__version__} "
                      "is installed; the posterior may not work.", stacklevel=2)
    return checkpoint
