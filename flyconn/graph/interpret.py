"""Adapter to ``connectome_interpreter`` (optional extra ``flyconn[interpret]``, ADR-0002).

Converts a :class:`ConnectivityMatrix` into the input-proportion matrix and the
index-keyed dicts that connectome_interpreter expects, and wraps its
``compress_paths_signed``. Import this module only when the extra is installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from flyconn.graph.matrix import ConnectivityMatrix


def _require() -> Any:
    try:
        import connectome_interpreter as ci
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        msg = "connectome_interpreter is not installed; pip install 'flyconn[interpret]'"
        raise ImportError(msg) from exc
    return ci


def to_interpreter_inputs(
    m: ConnectivityMatrix, group: str | None = None
) -> tuple[sp.csc_matrix, dict[int, int], dict[int, str]]:
    """Return ``(inprop, idx_to_sign, idx_to_group)`` for connectome_interpreter.

    ``inprop`` is pre-in-rows, each column normalised by the postsynaptic
    neuron's total input (float32 CSC). ``idx_to_sign`` maps matrix index to
    +1/-1 (neurons with unknown sign are treated as +1 by the library, so they
    are reported in the returned dict as +1 but their rows are zeroed in
    ``inprop`` to match flyconn's exclusion rule). ``idx_to_group`` maps index
    to ``meta[group]`` (or the neuron id as a string).
    """
    norm = m.input_normalized()
    unknown = m.signs == 0
    if unknown.any():
        keep = sp.diags((~unknown).astype(float))
        norm = sp.csr_matrix(keep @ norm)
    inprop = sp.csc_matrix(norm, dtype=np.float32)
    idx_to_sign = {i: (-1 if s < 0 else 1) for i, s in enumerate(m.signs)}
    if group is not None:
        values = m.meta[group].to_numpy(dtype=object)
        idx_to_group = {i: str(v) for i, v in enumerate(values)}
    else:
        idx_to_group = {i: str(n) for i, n in enumerate(m.neuron_ids)}
    return inprop, idx_to_sign, idx_to_group


def compress_paths_signed(
    m: ConnectivityMatrix, hops: int, *, output_threshold: float = 0.0, **kwargs: Any
) -> list[sp.csr_matrix]:
    """Net signed influence per step 1..hops via ``connectome_interpreter.compress_paths_signed``.

    The library returns ``(excitatory_steps, inhibitory_steps)`` as non-negative
    magnitudes (an even number of inhibitory hops counts as excitation); the net
    value returned here is ``excitatory - inhibitory`` per step. ``output_threshold``
    defaults to 0 (the library default 1e-4 would drop small entries).
    """
    ci = _require()
    inprop, idx_to_sign, _ = to_interpreter_inputs(m)
    exc, inh = ci.compress_paths_signed(
        inprop, idx_to_sign, target_layer_number=hops, output_threshold=output_threshold, **kwargs
    )
    return [
        sp.csr_matrix(sp.csr_matrix(e) - sp.csr_matrix(i)) for e, i in zip(exc, inh, strict=True)
    ]
