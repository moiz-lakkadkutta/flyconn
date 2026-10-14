"""Driver lines for robustness-screen hits (Meissner et al. 2025; M13, ADR-0011).

Annotations are curated free text across nomenclatures, so every row records how the
line was matched. Lines are never invented; NeuronBridge counts come from the cache only.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from flyconn.access.lines import LineCatalog
from flyconn.access.neuronbridge import cached_off_target_count
from flyconn.data.cache import store_dir
from flyconn.data.registry import get_dataset

CATALOG_REF = "flylight_lines@meissner2025"
PULL_HINT = f"flyconn data pull {CATALOG_REF}"


def load_catalog() -> LineCatalog | None:
    """The pulled line catalog, or None when ``flylight_lines`` is not in the store."""
    d = store_dir(get_dataset(CATALOG_REF))
    return LineCatalog(d) if (d / "lines.parquet").exists() else None


def _fmt(row: Any) -> str:
    q = "" if pd.isna(row["quality"]) else f"q{int(row['quality'])}"
    return f"{row['line']} ({q}{', specific' if row['specific'] else ''})"


def attach_lines(
    hits: pd.DataFrame, catalog: LineCatalog | None, *, max_lines: int = 5
) -> pd.DataFrame:
    """Add driver-line columns for each ``candidate`` (exact, then glob ``label*`` match)."""
    out = hits.copy()
    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            "best_line",
            "line_quality",
            "line_specific",
            "line_match",
            "lines",
            "off_target_bodies",
        )
    }
    for label in out["candidate"].astype(str):
        match, found = "not looked up", pd.DataFrame()
        if catalog is not None:
            found = catalog.lines_for_type(label)
            match = "exact"
            if found.empty:
                found = catalog.lines_for_type(f"{label}*")
                match = "glob" if not found.empty else "none"
        if found.empty:
            cols["best_line"].append(None)
            cols["line_quality"].append(None)
            cols["line_specific"].append(None)
            cols["lines"].append("")
            cols["off_target_bodies"].append(None)
        else:
            found = found.sort_values(
                ["quality", "specific", "line"],
                ascending=[True, False, True],
                na_position="last",
                ignore_index=True,
            )
            best = found.iloc[0]
            cols["best_line"].append(str(best["line"]))
            cols["line_quality"].append(None if pd.isna(best["quality"]) else int(best["quality"]))
            cols["line_specific"].append(bool(best["specific"]))
            cols["lines"].append("; ".join(_fmt(r) for _, r in found.head(max_lines).iterrows()))
            cols["off_target_bodies"].append(cached_off_target_count(str(best["line"])))
        cols["line_match"].append(match)
    for k, v in cols.items():
        out[k] = v
    out["line_quality"] = out["line_quality"].astype("Int64")
    out["off_target_bodies"] = out["off_target_bodies"].astype("Int64")
    return out
