"""Evidence-card report for robustness-audited screens (M13, ADR-0011)."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import pandas as pd

from flyconn.access.handoff import PULL_HINT
from flyconn.report.html import _footer, _header, _table  # pyright: ignore[reportPrivateUsage]
from flyconn.sim.params import CALIBRATIONS

if TYPE_CHECKING:
    from flyconn.experiments.robustness import RobustScreenResult
    from flyconn.experiments.spec import ExperimentSpec

TABLE_COLUMNS = [
    "label",
    "tier",
    "direction",
    "effect_hz",
    "q_value",
    "failed_axes",
    "best_line",
    "line_match",
]
_BASE_CAVEATS = (
    "Model predictions from wiring plus Shiu LIF dynamics; not measurements.",
    "NT signs are predicted; sampled-sign variants assume the published probabilities "
    "are calibrated.",
    "Line annotations are curated text and may not match EM type names one to one.",
    "'Strong' means robust within the tested axes, not true.",
)
_REPLICATE_CAVEAT = (
    "The replicate runs at an uncalibrated w_syn on a different sex; disagreement may be "
    "biology or calibration."
)


def caveats_for(spec: ExperimentSpec) -> tuple[str, ...]:
    """Caveats that apply to this spec's dataset and replicate setting."""
    ds = spec.dataset
    cal = (
        f"{ds} w_syn calibrated by protocol, not validated (ADR-0009)."
        if ds in CALIBRATIONS
        else f"{ds} runs at Shiu's published w_syn, uncalibrated for this dataset."
    )
    rep = (_REPLICATE_CAVEAT,) if spec.robustness and spec.robustness.replicate else ()
    return (_BASE_CAVEATS[0], cal, *_BASE_CAVEATS[1:2], *rep, *_BASE_CAVEATS[2:])


def _e(x: object) -> str:
    return html.escape(str(x))


def _p(x: object) -> str:
    return "n/a" if bool(pd.isna(x)) else _e(x)  # pyright: ignore[reportArgumentType]


def _null_text(hit: pd.Series) -> str:
    """Random-null line: outcome and p, or the recorded reason the null was skipped."""
    skipped = str(hit.get("random_null_skipped", "") or "")
    if skipped:
        return f"{_e(hit['random_null'])} (skipped: {_e(skipped)})"
    return f"{_e(hit['random_null'])} (p {_p(hit['random_null_p'])})"


def _card(
    row: pd.Series, hit: pd.Series, catalog_loaded: bool, caveat_list: tuple[str, ...]
) -> str:
    lines = (
        _e(row["lines"]) or "no annotated split-GAL4 in Meissner 2025"
        if catalog_loaded
        else f"lines not looked up (run <code>{_e(PULL_HINT)}</code>)"
    )
    ps = row["path_stability"]
    caveats = "".join(f"<li>{_e(c)}</li>" for c in caveat_list)
    thin = (
        "<p class='warn'>Thinly tested: fewer than two graded axes ran.</p>"
        if bool(row["thinly_tested"])
        else ""
    )
    return (
        "<div class='card'>"
        f"<h3>{_e(row['label'])} &middot; {_e(row['tier'])}</h3>"
        f"<p>Silencing moves the primary readout <b>{_e(row['direction'])}</b> by "
        f"{float(row['effect_hz']):.2f} Hz (95% CI {float(row['ci_low']):.2f} to "
        f"{float(row['ci_high']):.2f}), q {float(row['q_value']):.2g}.</p>{thin}"
        "<ul>"
        f"<li>w_syn: {_e(hit['w_syn'])} ({_e(hit['w_syn_detail'])})</li>"
        f"<li>NT signs: {_e(hit['nt'])} (holds {_e(hit['nt_holds'])}; "
        f"flips: {_e(hit['nt_flips']) or 'none'})</li>"
        f"<li>threshold: {_e(hit['min_weight'])} ({_e(hit['min_weight_detail'])})</li>"
        f"<li>random null: {_null_text(hit)}</li>"
        f"<li>replicate (supporting, not graded): {_e(row['replicate'])}</li>"
        f"<li>path stability: {'n/a' if bool(pd.isna(ps)) else f'{float(ps):.2f}'}</li>"
        f"<li>lines: {lines} [{_e(row['line_match'])}]</li>"
        "</ul>"
        f"<p><b>Suggested test:</b> {_e(row['suggested_test'])}</p>"
        f"<details><summary>Caveats</summary><ul>{caveats}</ul></details>"
        "</div>"
    )


def render_robust_report(
    result: RobustScreenResult, handoff: pd.DataFrame, *, catalog_loaded: bool
) -> str:
    spec, prov = result.spec, result.provenance
    rob = prov.get("robustness", {})
    is_ref = result.effects["axis"] == "reference"
    in_hits = result.effects["candidate"].isin(handoff["label"])
    caveats = caveats_for(spec)
    parts = [
        *_header(f"flyconn robustness screen: {spec.name}", spec.name, prov),
        "<style>.card{border:1px solid #ccc;border-radius:6px;padding:.5rem 1rem;"
        "margin:1rem 0}</style>",
        "<h2>Design</h2>",
        f"<p>{_e(rob.get('design'))}. Variants: "
        + ", ".join(_e(v["label"]) for v in rob.get("variants", []))
        + ". Every variant silences the reference run's candidates.</p>",
        f"<p class='note'>{_e(rob.get('statistics'))}.</p>",
        "<h2>Ranked hits</h2>",
    ]
    if handoff.empty:
        parts.append(
            "<p><b>No reference hits</b> at the spec's alpha; variants were still run.</p>"
        )
    else:
        parts.append(_table(pd.DataFrame(handoff.loc[:, TABLE_COLUMNS])))
        hits = result.hits.set_index("candidate")
        for _, row in handoff.iterrows():
            parts.append(_card(row, hits.loc[row["label"]], catalog_loaded, caveats))
    parts += [
        "<h2>Selection stability</h2>",
        "<p class='note'>Jaccard overlap between the reference selection and the one each "
        "variant's own ranking would have picked (candidates were fixed regardless).</p>",
        _table(result.selection_stability),
        "<h2>Not hits and controls</h2>",
        _table(pd.DataFrame(result.effects.loc[is_ref & ~in_hits])),
        *_footer(prov),
    ]
    return "\n".join(parts)
