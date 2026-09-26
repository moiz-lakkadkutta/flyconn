"""Synthetic connectome fixtures for tests and examples.

The generator is deterministic for a given seed and emits the harmonized
schema seed described in ``docs/PLAN.md`` §3 so that unit tests never touch
real data or the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NT_CLASSES: tuple[str, ...] = (
    "acetylcholine",
    "glutamate",
    "gaba",
    "dopamine",
    "octopamine",
    "serotonin",
    "histamine",
)
"""Transmitter classes of the seven-class MaleCNS classifier, in a fixed order."""

_SUPER_CLASSES: tuple[str, ...] = (
    "central_brain_intrinsic",
    "optic_lobe_intrinsic",
    "sensory",
    "descending",
    "motor",
)
_SIDES: tuple[str, ...] = ("left", "right")
_DATASET = "synthetic"
_VERSION = "0"


def synthetic_connectome(
    n_neurons: int = 50,
    n_edges: int = 200,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(neurons, edges)`` tables for a small random connectome.

    Neurons carry ``dataset, version, neuron_id, super_class, cell_type, side,
    nt_pred, nt_conf`` and one ``nt_p_<transmitter>`` column per class in
    :data:`NT_CLASSES` (rows sum to one; ``nt_pred`` is the argmax and
    ``nt_conf`` its probability). Edges carry ``dataset, version, pre, post,
    weight`` with no self-loops and no duplicate ``(pre, post)`` pairs.

    Raises:
        ValueError: if ``n_edges`` exceeds the number of possible directed pairs.
    """
    max_edges = n_neurons * (n_neurons - 1)
    if n_edges > max_edges:
        msg = f"n_edges={n_edges} exceeds {max_edges} possible directed pairs"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)

    neuron_ids = np.arange(10_001, 10_001 + n_neurons, dtype=np.int64)
    probs = rng.dirichlet(np.full(len(NT_CLASSES), 0.5), size=n_neurons)
    argmax = probs.argmax(axis=1)
    neurons = pd.DataFrame(
        {
            "dataset": _DATASET,
            "version": _VERSION,
            "neuron_id": neuron_ids,
            "super_class": rng.choice(_SUPER_CLASSES, size=n_neurons),
            "cell_type": [f"T{i:03d}" for i in rng.integers(0, max(1, n_neurons // 3), n_neurons)],
            "side": rng.choice(_SIDES, size=n_neurons),
            "nt_pred": [NT_CLASSES[i] for i in argmax],
            "nt_conf": probs[np.arange(n_neurons), argmax],
        }
    )
    for j, nt in enumerate(NT_CLASSES):
        neurons[f"nt_p_{nt}"] = probs[:, j]

    # Sample distinct ordered pairs (pre != post) without replacement.
    pair_codes = rng.choice(max_edges, size=n_edges, replace=False)
    pre_idx = pair_codes // (n_neurons - 1)
    post_idx = pair_codes % (n_neurons - 1)
    post_idx = post_idx + (post_idx >= pre_idx)  # skip the diagonal
    weights = rng.geometric(p=0.25, size=n_edges).astype(np.int32)
    edges = pd.DataFrame(
        {
            "dataset": _DATASET,
            "version": _VERSION,
            "pre": neuron_ids[pre_idx],
            "post": neuron_ids[post_idx],
            "weight": weights,
        }
    )
    return neurons, edges
