"""Minimal Newick reader and writer (no external dependencies)."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

_STRUCTURE = "(),:;"


@dataclass
class _ParsedTree:
    """
    A rooted tree read from Newick, nodes in preorder (index 0 is the root).

    Children follow their order in the Newick string, so tips appear in the
    same order as in the file, which is how ``ape::read.tree`` numbers them.
    """
    parent: np.ndarray            # (m,) int, -1 for the root
    length: np.ndarray            # (m,) float, NaN where no length was given
    label: List[Optional[str]]    # (m,) labels (None if absent)
    is_tip: np.ndarray            # (m,) bool

    @property
    def tip_indices(self) -> np.ndarray:
        return np.flatnonzero(self.is_tip)


def _read_source(source: Union[str, os.PathLike]) -> str:
    """Return Newick text from a path or from a Newick string."""
    if isinstance(source, os.PathLike):
        return Path(source).read_text()
    if not isinstance(source, str):
        raise TypeError(f"Expected a path or a Newick string, got {type(source).__name__}.")
    stripped = source.strip()
    if stripped.startswith("(") or stripped.endswith(";"):
        return source
    return Path(source).read_text()


def _strip_comments(text: str) -> str:
    """Remove ``[...]`` comments (outside quoted labels)."""
    out, depth, quoted = [], 0, False
    for c in text:
        if c == "'" and depth == 0:
            quoted = not quoted
        elif not quoted and c == "[":
            depth += 1
            continue
        elif not quoted and c == "]" and depth > 0:
            depth -= 1
            continue
        if depth == 0:
            out.append(c)
    return "".join(out)


def parse_newick(text: str) -> _ParsedTree:
    """
    Parse a single Newick tree.

    Supports quoted and unquoted labels, internal node labels (kept but not
    used), branch lengths in any float notation, ``[...]`` comments and
    multifurcations (e.g. the unrooted trifurcating root of ClonalFrameML).
    """
    s = _strip_comments(text)
    parent: List[int] = [-1]
    length: List[float] = [math.nan]
    label: List[Optional[str]] = [None]
    n_children: List[int] = [0]

    def new_node(p: int) -> int:
        parent.append(p)
        length.append(math.nan)
        label.append(None)
        n_children.append(0)
        n_children[p] += 1
        return len(parent) - 1

    cur, i, n, depth, ended = 0, 0, len(s), 0, False
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c == "(":
            depth += 1
            cur = new_node(cur)
            i += 1
        elif c == ",":
            if parent[cur] < 0:
                raise ValueError("Malformed Newick: ',' outside parentheses.")
            cur = new_node(parent[cur])
            i += 1
        elif c == ")":
            depth -= 1
            if depth < 0 or parent[cur] < 0:
                raise ValueError("Malformed Newick: unbalanced ')'.")
            cur = parent[cur]
            i += 1
        elif c == ":":
            j = i + 1
            while j < n and s[j] not in _STRUCTURE and not s[j].isspace():
                j += 1
            token = s[i + 1:j]
            try:
                length[cur] = float(token)
            except ValueError:
                raise ValueError(f"Malformed Newick: invalid branch length {token!r}.") from None
            i = j
        elif c == ";":
            ended = True
            i += 1
            break
        elif c == "'":
            j, chars = i + 1, []
            while True:
                if j >= n:
                    raise ValueError("Malformed Newick: unterminated quoted label.")
                if s[j] == "'":
                    if j + 1 < n and s[j + 1] == "'":     # '' is an escaped quote
                        chars.append("'")
                        j += 2
                        continue
                    break
                chars.append(s[j])
                j += 1
            label[cur] = "".join(chars)
            i = j + 1
        else:
            j = i
            while j < n and s[j] not in _STRUCTURE and not s[j].isspace():
                j += 1
            label[cur] = s[i:j]
            i = j

    if depth != 0:
        raise ValueError("Malformed Newick: unbalanced parentheses.")
    if ended and s[i:].strip():
        raise ValueError("The Newick text contains more than one tree; pass a single tree.")
    if not ended and not s.strip():
        raise ValueError("Empty Newick text.")

    return _ParsedTree(
        parent=np.asarray(parent, dtype=np.int64),
        length=np.asarray(length, dtype=np.float64),
        label=label,
        is_tip=np.asarray(n_children) == 0,
    )


def _format_label(name: str) -> str:
    if name and not any(c in name for c in "()[]',:;") and not any(c.isspace() for c in name):
        return name
    return "'" + name.replace("'", "''") + "'"


def format_newick(edge: np.ndarray, n: int, tip_labels: Optional[Sequence[str]] = None) -> str:
    """
    Write a clonal tree (``[parent, child, length]`` rows, 1-indexed nodes,
    root ``2n-1``) as a Newick string. Lengths use Python's shortest
    round-trip representation, so reading the string back is lossless.
    """
    if n == 1:
        return f"{_format_label(tip_labels[0]) if tip_labels else '1'};"
    children = {}
    for p, c, length in edge:
        children.setdefault(int(p), []).append((int(c), float(length)))

    def write(node: int) -> str:
        if node <= n:
            return _format_label(tip_labels[node - 1]) if tip_labels else str(node)
        return "(" + ",".join(f"{write(c)}:{length!r}" for c, length in children[node]) + ")"

    return write(2 * n - 1) + ";"
