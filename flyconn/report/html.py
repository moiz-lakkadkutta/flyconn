"""Render an experiment result as a single self-contained HTML page."""

from __future__ import annotations

import base64
import html
import io
import json
from typing import TYPE_CHECKING, Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from flyconn.data.store import Store
    from flyconn.experiments.runner import ExperimentResult
    from flyconn.experiments.spec import ExperimentSpec
    from flyconn.experiments.sweep import SweepResult


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


_STYLE = (
    "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}"
    ".tbl{border-collapse:collapse;font-size:12px}.tbl td,.tbl th{border:1px solid #ccc;padding:2px 6px}"
    ".warn{background:#fff4e5;border-left:4px solid #e69500;padding:.5rem}.note{color:#555;font-size:13px}"
    "pre{background:#f6f6f6;padding:.5rem;overflow:auto;font-size:11px}</style>"
)


def _header(title: str, name: str, prov: dict[str, Any]) -> list[str]:
    """Document head, title and the model-prediction and calibration banners."""
    calibration = prov.get("network", {}).get("calibration", "")
    cal_html = (
        f'<p class="warn"><b>Calibration:</b> {html.escape(str(calibration))}</p>'
        if calibration
        else ""
    )
    return [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        _STYLE,
        "</head><body>",
        f"<h1>{html.escape(name)}</h1>",
        "<p class='warn'><b>Model prediction.</b> Every number on this page is a prediction of a "
        "leaky integrate-and-fire model constrained by connectome wiring and <i>predicted</i> "
        "neurotransmitters (Shiu et al. 2024 parameters). It is not measured activity and does not "
        "predict behaviour.</p>",
        cal_html,
    ]


def _controls_line(prov: dict[str, Any]) -> str:
    controls = prov.get("controls", {})
    return (
        f"<p>Controls (on by default): degree_preserving_rewire x{controls.get('degree_preserving_rewire')}, "
        f"sign_shuffle x{controls.get('sign_shuffle')}, unstimulated baseline: {controls.get('unstimulated_baseline')}.</p>"
    )


def _footer(prov: dict[str, Any]) -> list[str]:
    """Provenance, citations and closing tags."""
    cites = "".join(
        f"<li>{html.escape(c['text'])} doi:{html.escape(c.get('doi', ''))}</li>"
        for c in prov.get("citations", [])
    )
    shown = {k: v for k, v in prov.items() if k != "citations"}
    return [
        "<h2>Provenance</h2>",
        f"<pre>{html.escape(json.dumps(shown, indent=1, default=str))}</pre>",
        "<h2>Citations</h2>",
        f"<ul>{cites}</ul>",
        "<p class='note'>Generated by flyconn; reports are self-contained (figures embedded).</p>",
        "</body></html>",
    ]


def render_report(result: ExperimentResult, spec: ExperimentSpec, store: Store) -> str:
    """Self-contained HTML: summary, effects vs controls, top neurons, provenance, citations."""
    prov = result.provenance
    if "significant" in result.readouts.columns:
        n_sig = int(result.readouts["significant"].sum())
    else:
        n_sig = 0
    parts = [
        *_header(f"flyconn experiment: {spec.name}", spec.name, prov),
        "<h2>Design</h2>",
        f"<p>Dataset <b>{html.escape(store.ref)}</b>; {prov['stimulated_neurons']} stimulated neurons; "
        f"{prov['silenced_neurons']} silenced; {spec.trials} trials x {spec.duration_ms:g} ms; seed {spec.seed}; "
        f"device {html.escape(spec.device)} ({html.escape(spec.dtype)}).</p>",
        _controls_line(prov),
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
        *_footer(prov),
    ]
    return "\n".join(parts)


_MAX_BARS = 40


def _silence_figure(sweep: pd.DataFrame) -> str:
    """One panel per readout: effect of silencing each variant, ranked, with bootstrap 95% CI."""
    if sweep.empty:
        return "<p>No variants were simulated.</p>"
    readouts = list(dict.fromkeys(sweep["readout"].tolist()))
    n_rows = min(int(sweep.groupby("readout").size().max()), _MAX_BARS)
    fig, axes = plt.subplots(
        1, len(readouts), figsize=(3.6 * len(readouts) + 1, 0.8 + 0.28 * n_rows), squeeze=False
    )
    for ax, name in zip(axes[0], readouts, strict=True):
        g = (
            cast("pd.DataFrame", sweep[sweep["readout"] == name])
            .sort_values("rank")
            .head(_MAX_BARS)
        )
        y = list(range(len(g)))
        colors = ["#c0504d" if s else "#9db3c8" for s in g["significant"]]
        ax.barh(y, g["difference"], color=colors)
        err = [g["difference"] - g["ci_low"], g["ci_high"] - g["difference"]]
        ax.errorbar(g["difference"], y, xerr=err, fmt="none", ecolor="#333", capsize=2, lw=0.8)
        ax.axvline(0, color="grey", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([str(v) for v in g["variant"]], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(str(name), fontsize=9)
        ax.set_xlabel("silenced - stimulated (Hz)", fontsize=8)
    fig.suptitle("Effect of silencing each variant (red: q < alpha); bootstrap 95% CI", fontsize=9)
    return _fig_to_img(fig)


def _rate_figure(sweep: pd.DataFrame) -> str:
    """Readout vs stimulation rate, mean with bootstrap 95% CI of the mean, and the reference."""
    if sweep.empty:
        return "<p>No variants were simulated.</p>"
    readouts = list(dict.fromkeys(sweep["readout"].tolist()))
    fig, axes = plt.subplots(1, len(readouts), figsize=(3.4 * len(readouts) + 1, 3), squeeze=False)
    for ax, name in zip(axes[0], readouts, strict=True):
        g = cast("pd.DataFrame", sweep[sweep["readout"] == name]).sort_values("rate_hz")
        ax.plot(g["rate_hz"], g["mean_hz"], "o-", color="#1f4e79")
        ax.fill_between(
            g["rate_hz"], g["mean_ci_low"], g["mean_ci_high"], color="#1f4e79", alpha=0.2
        )
        ref = float(g["mean_reference_hz"].iloc[0])
        ax.axhline(ref, color="grey", ls="--", lw=1, label=str(g["reference"].iloc[0]))
        ax.set_title(str(name), fontsize=9)
        ax.set_xlabel("stimulation rate (Hz)", fontsize=8)
        ax.set_ylabel("mean readout rate (Hz)", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Readout vs stimulation rate (band: bootstrap 95% CI of the mean)", fontsize=9)
    return _fig_to_img(fig)


def render_sweep_report(result: SweepResult, spec: ExperimentSpec, store: Store) -> str:
    """Self-contained HTML for a sweep: ranked effects, a figure, shared controls, provenance."""
    prov = result.provenance
    sw = prov.get("sweep", {})
    t = result.sweep
    n_sig = int(t["significant"].sum()) if "significant" in t.columns else 0
    n_var = int(result.variants["simulated"].sum())
    reference = html.escape(str(sw.get("reference", "")))
    if result.kind == "silence_each":
        design = (
            f"<p>Silencing screen (<code>silence_each</code>): candidates {html.escape(str(sw.get('select')))}, "
            f"grouped by <code>{html.escape(str(sw.get('group_by')))}</code>; {sw.get('candidates')} candidate "
            f"group(s), ranked by summed firing rate in the stimulated condition; the {n_var} most active "
            f"(max_items {sw.get('max_items')}) were silenced one at a time. Groups that never fire when "
            "stimulated are listed but not simulated: silencing zeroes outgoing synapses, which cannot "
            "change a run in which the group emits no spike.</p>"
        )
        figure = _silence_figure(t)
        cols = ["readout", "rank", "variant", "n_silenced"]
    else:
        design = (
            f"<p>Rate sweep (<code>rate_hz</code>): every stimulated neuron driven at "
            f"{', '.join(f'{r:g}' for r in sw.get('rates_hz', []))} Hz in turn.</p>"
        )
        figure = "<h3>Readout vs stimulation rate</h3>" + _rate_figure(t)
        cols = ["readout", "rate_hz", "rank", "variant"]
    stat_cols = [
        "mean_hz",
        "mean_ci_low",
        "mean_ci_high",
        "mean_reference_hz",
        "difference",
        "ci_low",
        "ci_high",
        "cohens_d",
        "p_value",
        "q_value",
        "significant",
    ]
    shown = cast("pd.DataFrame", t[[c for c in [*cols, *stat_cols] if c in t.columns]])
    if result.kind == "rate_hz" and not t.empty:
        shown = shown.sort_values(["readout", "rate_hz"], kind="stable")
    parts = [
        *_header(f"flyconn sweep: {spec.name}", spec.name, prov),
        "<h2>Design</h2>",
        f"<p>Dataset <b>{html.escape(store.ref)}</b>; {prov['stimulated_neurons']} stimulated neurons; "
        f"{spec.trials} trials x {spec.duration_ms:g} ms per condition; seed {spec.seed} (every variant "
        f"reuses the same seeds); device {html.escape(spec.device)} ({html.escape(spec.dtype)}).</p>",
        design,
        _controls_line(prov) + "<p class='note'>Shared conditions (baseline, stimulated, controls) "
        "were simulated once and reused by every variant.</p>",
        "<h2>Statistics</h2>",
        f"<p class='note'><b>n = {spec.trials} trials</b> per variant and per reference condition: trials "
        "share one connectome and differ only in Poisson input seeds, so n counts model trials, not "
        f"animals. Each variant is compared with the <b>{reference}</b> condition: difference of per-trial "
        "mean readout rates with bootstrap 95% CI, Mann-Whitney p (unpaired, conservative given shared "
        "seeds), Cohen's d (clipped to +/-10). Benjamini-Hochberg q is computed across the whole sweep "
        f"(all {len(t)} variant x readout rows). Rank 1 is the largest absolute effect per readout.</p>",
        figure,
        "<h2>Ranked effects</h2>",
        _table(shown),
        f"<p><b>{n_sig}</b> of {len(t)} variant x readout rows significant at q &lt; {spec.report.alpha}.</p>",
        "<h2>Variants</h2>",
        _table(result.variants),
        "<h2>Shared conditions and controls</h2>",
        _table(result.conditions),
        "<p class='note'>Stimulated vs baseline and vs null-connectome controls (own BH family).</p>",
        _table(result.readouts),
        *_footer(prov),
    ]
    return "\n".join(parts)
