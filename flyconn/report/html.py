"""Render an experiment result as a single self-contained HTML page."""

from __future__ import annotations

import base64
import html
import io
import json
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from flyconn.data.store import Store
    from flyconn.experiments.runner import ExperimentResult
    from flyconn.experiments.spec import ExperimentSpec


def _fig_to_img(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="figure" src="data:image/png;base64,{b64}" style="max-width:100%">'


def _readout_figure(readouts: pd.DataFrame) -> str:
    if readouts.empty:
        return "<p>No readout statistics.</p>"
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.35 * len(readouts)))
    labels = [
        f"{r} : {c}" for r, c in zip(readouts["readout"], readouts["comparison"], strict=True)
    ]
    y = range(len(readouts))
    err = [
        readouts["difference"] - readouts["ci_low"],
        readouts["ci_high"] - readouts["difference"],
    ]
    ax.errorbar(readouts["difference"], list(y), xerr=err, fmt="o", color="#1f4e79", capsize=3)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("difference in mean readout rate (Hz) with bootstrap 95% CI")
    ax.invert_yaxis()
    return _fig_to_img(fig)


def _top_figure(top: pd.DataFrame) -> str:
    if top.empty:
        return ""
    fig, ax = plt.subplots(figsize=(7, 0.4 + 0.3 * len(top)))
    ids = [str(n) for n in top["neuron_id"].tolist()]
    if "cell_type" in top.columns:
        types = [str(t) if t is not None and t == t else "" for t in top["cell_type"].tolist()]
    else:
        types = [""] * len(ids)
    names = [f"{n} {t}".strip() for n, t in zip(ids, types, strict=True)]
    ax.barh(names, top["rate_hz_stimulated"], color="#6a9fb5", label="stimulated")
    if "rate_hz_perturbed" in top:
        ax.barh(names, top["rate_hz_perturbed"], color="#c0504d", alpha=0.6, label="perturbed")
    ax.set_xlabel("mean rate (Hz)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    return _fig_to_img(fig)


def _table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    if df.empty:
        return "<p>(empty)</p>"
    d = df.copy()

    def fmt(x: float) -> str:
        return floatfmt.format(x)

    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(fmt)
    return d.to_html(index=False, escape=True, classes="tbl")


def render_report(result: ExperimentResult, spec: ExperimentSpec, store: Store) -> str:
    """Self-contained HTML: summary, effects vs controls, top neurons, provenance, citations."""
    prov = result.provenance
    if "significant" in result.readouts.columns:
        n_sig = int(result.readouts["significant"].sum())
    else:
        n_sig = 0
    calibration = prov.get("network", {}).get("calibration", "")
    cal_html = (
        f'<p class="warn"><b>Calibration:</b> {html.escape(str(calibration))}</p>'
        if calibration
        else ""
    )
    cites = "".join(
        f"<li>{html.escape(c['text'])} doi:{html.escape(c.get('doi', ''))}</li>"
        for c in prov.get("citations", [])
    )
    controls = prov.get("controls", {})
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>flyconn experiment: {html.escape(spec.name)}</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}"
        ".tbl{border-collapse:collapse;font-size:12px}.tbl td,.tbl th{border:1px solid #ccc;padding:2px 6px}"
        ".warn{background:#fff4e5;border-left:4px solid #e69500;padding:.5rem}.note{color:#555;font-size:13px}"
        "pre{background:#f6f6f6;padding:.5rem;overflow:auto;font-size:11px}</style></head><body>",
        f"<h1>{html.escape(spec.name)}</h1>",
        "<p class='warn'><b>Model prediction.</b> Every number on this page is a prediction of a "
        "leaky integrate-and-fire model constrained by connectome wiring and <i>predicted</i> "
        "neurotransmitters (Shiu et al. 2024 parameters). It is not measured activity and does not "
        "predict behaviour.</p>",
        cal_html,
        "<h2>Design</h2>",
        f"<p>Dataset <b>{html.escape(store.ref)}</b>; {prov['stimulated_neurons']} stimulated neurons; "
        f"{prov['silenced_neurons']} silenced; {spec.trials} trials x {spec.duration_ms:g} ms; seed {spec.seed}; "
        f"device {html.escape(spec.device)} ({html.escape(spec.dtype)}).</p>",
        f"<p>Controls (on by default): degree_preserving_rewire x{controls.get('degree_preserving_rewire')}, "
        f"sign_shuffle x{controls.get('sign_shuffle')}, unstimulated baseline: {controls.get('unstimulated_baseline')}.</p>",
        "<h2>Conditions</h2>",
        _table(result.conditions),
        "<h2>Readout effects</h2>",
        "<p class='note'>Per-trial mean rate over each readout group; difference of means with bootstrap 95% CI, "
        "Mann-Whitney p, Benjamini-Hochberg q across all rows; Cohen's d (clipped to +/-10).</p>",
        _readout_figure(result.readouts),
        _table(result.readouts),
        f"<p><b>{n_sig}</b> comparison(s) significant at q &lt; {spec.report.alpha}.</p>",
        "<h2>Top neurons (stimulated condition)</h2>",
        _top_figure(result.top_neurons),
        _table(result.top_neurons),
        "<h2>Provenance</h2>",
        f"<pre>{html.escape(json.dumps({k: v for k, v in prov.items() if k != 'citations'}, indent=1, default=str))}</pre>",
        "<h2>Citations</h2>",
        f"<ul>{cites}</ul>",
        "<p class='note'>Generated by flyconn; reports are self-contained (figures embedded).</p>",
        "</body></html>",
    ]
    return "\n".join(parts)
