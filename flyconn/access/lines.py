"""Driver-line catalog from Meissner et al. 2025 (eLife 13:RP98405), Figure 1-source data 1."""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.registry import get_dataset

DATASET = "flylight_lines"
VERSION = "meissner2025"
XLSX = "elife-98405-fig1-data1-v2.xlsx"
_COLUMNS = {
    "Line": "line",
    "Adult/\nLarval": "adult_larval",
    "Main annotator": "annotator",
    "Lab": "lab",
    "Quality": "quality",
    "Sex difference": "sex_difference",
    "VNC only": "vnc_only",
    "Brain only": "brain_only",
    "DOI to cite": "doi",
    "First author(s) & year": "first_author_year",
    "Janelia Robot ID": "robot_id",
    "Cell types": "cell_types_raw",
    "EM Body IDs": "em_body_ids_raw",
    "Alias": "alias",
    "Genotype": "genotype",
}


def _as_text(value: object) -> str | None:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [t.strip() for t in re.split(r"[,;]", str(value)) if t.strip()]


def _ids(value: object) -> list[int]:
    return [int(t) for t in _tokens(value) if t.isdigit()]


def convert_meissner_lines(xlsx_path: Path, out_dir: Path) -> dict[str, Any]:
    """Convert the eLife xlsx to ``lines.parquet`` with token lists and write provenance."""
    xlsx_path, out_dir = Path(xlsx_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(xlsx_path)
    raw = raw.rename(columns={k: v for k, v in _COLUMNS.items() if k in raw.columns})

    def text(col: str) -> pd.Series:
        if col not in raw.columns:
            return pd.Series([None] * len(raw), dtype="object")
        values = cast("pd.Series", raw[col])
        return values.map(_as_text).astype("string")

    df = pd.DataFrame(
        {
            "line": raw["line"].astype(str),
            "adult_larval": text("adult_larval"),
            "quality": pd.to_numeric(raw.get("quality"), errors="coerce"),
            "sex_difference": text("sex_difference"),
            "vnc_only": text("vnc_only"),
            "brain_only": text("brain_only"),
            "doi": text("doi"),
            "first_author_year": text("first_author_year"),
            "lab": text("lab"),
            "alias": text("alias"),
            "genotype": text("genotype"),
            "cell_types": [_tokens(v) for v in raw["cell_types_raw"]],
            "em_body_ids": [_ids(v) for v in raw["em_body_ids_raw"]],
            "em_body_ids_raw": text("em_body_ids_raw"),
        }
    )
    adult = cast("pd.Series", df["adult_larval"]).astype(str).str.startswith("Adult")
    counts = {
        "lines": len(df),
        "adult_with_cell_types": int((adult & (df["cell_types"].map(len) > 0)).sum()),
        "with_em_body_ids": int((df["em_body_ids"].map(len) > 0).sum()),
        "with_em_body_ids_raw": int(df["em_body_ids_raw"].notna().sum()),
    }
    table = pa.Table.from_pandas(df, preserve_index=False)
    write_parquet(table, out_dir / "lines.parquet")
    return write_provenance(
        out_dir,
        dataset=DATASET,
        version=VERSION,
        converter="flyconn.access.lines",
        inputs=[xlsx_path],
        counts=counts,
        note="cell type names are free text and dataset-agnostic; EM body ids carry no dataset tag",
    )


class LineCatalog:
    """Query driver lines by cell type, line name or EM body id."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.lines = pd.read_parquet(self.root / "lines.parquet")
        self.provenance: dict[str, Any] = json.loads((self.root / "provenance.json").read_text())

    def lines_for_type(self, pattern: str, *, min_quality: int | None = None) -> pd.DataFrame:
        """Lines whose annotated cell types match ``pattern`` (glob); best quality first.

        ``specific`` is true when the line lists only that one cell type.
        """

        def hit(ts: list[str]) -> bool:
            return any(fnmatch.fnmatchcase(t, pattern) for t in ts)

        mask = self.lines["cell_types"].map(hit)
        df = cast("pd.DataFrame", self.lines[mask]).copy()
        if min_quality is not None:
            df = cast("pd.DataFrame", df[df["quality"] <= min_quality])
        df["specific"] = df["cell_types"].map(len) == 1
        df = df.sort_values(["quality", "line"], na_position="last", ignore_index=True)
        cols = ["line", "cell_types", "quality", "specific", "doi", "first_author_year", "genotype"]
        return cast("pd.DataFrame", df[cols])

    def types_for_line(self, line: str) -> list[str]:
        hit = self.lines[self.lines["line"] == line]
        if hit.empty:
            msg = f"unknown line {line!r}"
            raise KeyError(msg)
        return list(hit.iloc[0]["cell_types"])

    def lines_for_body(self, body_id: int) -> pd.DataFrame:
        def has(ids: list[int]) -> bool:
            return int(body_id) in ids

        mask = self.lines["em_body_ids"].map(has)
        df = cast("pd.DataFrame", self.lines[mask]).sort_values(
            ["quality", "line"], na_position="last", ignore_index=True
        )
        return cast("pd.DataFrame", df[["line", "cell_types", "quality", "doi"]])

    def citation_text(self) -> str:
        spec = get_dataset(f"{DATASET}@{VERSION}")
        return "\n".join(f"- {c.text} doi:{c.doi}" for c in spec.citations)
