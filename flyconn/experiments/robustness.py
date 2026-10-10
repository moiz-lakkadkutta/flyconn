"""Robustness-audited silencing screens (M13, ADR-0011).

One axis at a time around the calibrated reference: w_syn, sampled NT signs, network
threshold and a cross-dataset replicate. Every variant silences the reference run's
candidates, so effects are comparable across variants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flyconn.experiments.spec import NO_CONTROLS, ExperimentSpec, SimParams, derive
from flyconn.sim.params import CALIBRATIONS, default_params


@dataclass(frozen=True)
class Variant:
    axis: str
    label: str
    spec: ExperimentSpec


def _r6(x: float) -> float:
    return round(float(x), 6)


def w_syn_values(spec: ExperimentSpec) -> tuple[float, ...]:
    """w_syn values of the w_syn axis, sorted, without the reference value."""
    r = spec.robustness
    if r is None or r.w_syn == "off":
        return ()
    ref = spec.params.w_syn_mv or default_params(spec.dataset).w_syn_mv
    if r.w_syn == "auto":
        cal = CALIBRATIONS.get(spec.dataset)
        if cal is None:
            msg = (
                f"robustness w_syn: auto needs a calibration entry for {spec.dataset}; "
                "give an explicit list of mV values or 'off'"
            )
            raise ValueError(msg)
        values = {*cal.ci_mv, *cal.sensitivity_mv}
    elif isinstance(r.w_syn, str):
        msg = f"robustness w_syn must be 'auto', 'off' or a list of mV values, got {r.w_syn!r}"
        raise ValueError(msg)
    else:
        values = set(r.w_syn)
    return tuple(sorted({_r6(v) for v in values} - {_r6(ref)}))


def plan_variants(spec: ExperimentSpec) -> list[Variant]:
    """Reference plus one variant per axis value; variants run without shared controls."""
    r = spec.robustness
    if r is None:
        msg = "spec has no robustness block"
        raise ValueError(msg)
    parent = spec.source_sha256

    def variant(axis: str, label: str, **changes: object) -> Variant:
        s = derive(spec, parent_sha256=parent, robustness=None, controls=NO_CONTROLS, **changes)
        return Variant(axis, label, s)

    out = [Variant("reference", "reference", spec)]
    out += [variant("w_syn", f"wsyn_{w:g}", params=SimParams(w)) for w in w_syn_values(spec)]
    for k in range(r.nt_samples):
        net = replace(spec.network, sign_policy="sampled", sign_seed=spec.seed + 2000 + k)
        out.append(variant("nt", f"nt_s{k}", network=net))
    for mw in r.min_weights:
        if mw != spec.network.min_weight:
            out.append(
                variant("min_weight", f"minw_{mw}", network=replace(spec.network, min_weight=mw))
            )
    if r.replicate:
        label = "rep_" + r.replicate.replace("@", "_").replace(".", "_")
        out.append(variant("replicate", label, dataset=r.replicate, params=SimParams()))
    return out
