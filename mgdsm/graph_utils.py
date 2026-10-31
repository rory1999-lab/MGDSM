"""Attribute graph construction used by the semantic encoder."""
from __future__ import annotations

import numpy as np


def cosine_adjacency(
    attrs: np.ndarray, add_self_loops: bool = True, top_k: int = 5
) -> np.ndarray:
    """Build a symmetric graph from each row's strongest cosine neighbors.

    Attributes have shape [number of classes, number of attributes].
    A zero top_k gives no cross-class edges. Self loops are optional.
    """
    attrs = np.asarray(attrs, dtype=np.float32)
    if attrs.ndim != 2 or min(attrs.shape) == 0 or not np.isfinite(attrs).all():
        raise ValueError("attrs must be a nonempty finite two-dimensional matrix")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")

    normalized = attrs / (np.linalg.norm(attrs, axis=1, keepdims=True) + 1e-8)
    adjacency = np.clip(normalized @ normalized.T, 0.0, 1.0)
    np.fill_diagonal(adjacency, 0.0)
    n = len(attrs)
    k = min(top_k, n - 1)
    mask = np.zeros_like(adjacency)
    if k:
        for i in range(n):
            neighbors = np.argsort(adjacency[i])[-k:]
            mask[i, neighbors] = 1.0
    adjacency *= np.maximum(mask, mask.T)
    if add_self_loops:
        adjacency += np.eye(n, dtype=np.float32)
    return adjacency.astype(np.float32)


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Return D^(-1/2) A D^(-1/2) for a nonnegative adjacency matrix."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if (
        adjacency.ndim != 2
        or adjacency.shape[0] != adjacency.shape[1]
        or adjacency.shape[0] == 0
        or not np.isfinite(adjacency).all()
        or np.any(adjacency < 0)
    ):
        raise ValueError("adjacency must be a nonempty finite nonnegative square matrix")
    degrees = adjacency.sum(axis=1)
    inverse_sqrt = np.power(np.clip(degrees, 1e-8, None), -0.5)
    scaling = np.diag(inverse_sqrt)
    return (scaling @ adjacency @ scaling).astype(np.float32)
