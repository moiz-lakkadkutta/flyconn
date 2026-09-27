"""Tests for the driver-line catalog (Meissner et al. 2025 table -> Parquet -> queries)."""

from pathlib import Path

import pandas as pd
import pytest

from flyconn.access.lines import LineCatalog, convert_meissner_lines

openpyxl = pytest.importorskip("openpyxl")


@pytest.fixture
def fake_xlsx(tmp_path: Path) -> Path:
    rows = [
        [
            "SS00090",
            "Adult",
            "T. W.",
            "Rubin",
            1,
            "N",
            "No",
            "Yes",
            "10.1002/cne.24512",
            "Wolff; 2018",
            2502255.0,
            "EPG",
            None,
            "GMR_19G02-x-GMR_15C03",
            "w; ...",
        ],
        [
            "SS00727",
            "Adult",
            "S. N.",
            "Card",
            2,
            "N",
            "No",
            "No",
            "10.7554/eLife.34272",
            "Namiki; 2018",
            1.0,
            "DNp01",
            "1234567890",
            "a",
            "w; ...",
        ],
        [
            "SS02299",
            "Adult",
            "S. N.",
            "Card",
            1,
            "N",
            "No",
            "No",
            "10.7554/eLife.34272",
            "Namiki; 2018",
            2.0,
            "DNp01, DNp02",
            "1234567890, 2345678901",
            "b",
            "w; ...",
        ],
        [
            "SS99999",
            "Larval split",
            "x",
            "y",
            None,
            None,
            None,
            None,
            "10.0/x",
            "Z; 2020",
            None,
            "mooncrawler",
            None,
            None,
            None,
        ],
        [
            "SS55555",
            "Adult",
            "x",
            "y",
            3,
            "Y",
            "Yes",
            "No",
            "10.0/y",
            "Q; 2021",
            5.0,
            None,
            None,
            None,
            None,
        ],
    ]
    cols = [
        "Line",
        "Adult/\nLarval",
        "Main annotator",
        "Lab",
        "Quality",
        "Sex difference",
        "VNC only",
        "Brain only",
        "DOI to cite",
        "First author(s) & year",
        "Janelia Robot ID",
        "Cell types",
        "EM Body IDs",
        "Alias",
        "Genotype",
    ]
    p = tmp_path / "lines.xlsx"
    pd.DataFrame(rows, columns=cols).to_excel(p, index=False, sheet_name="Cell type lines")
    return p


def test_convert_writes_parquet_with_token_lists(fake_xlsx: Path, tmp_path: Path):
    out = tmp_path / "store"
    prov = convert_meissner_lines(fake_xlsx, out)
    df = pd.read_parquet(out / "lines.parquet")
    assert len(df) == 5
    assert set(df.columns) >= {
        "line",
        "adult_larval",
        "quality",
        "sex_difference",
        "vnc_only",
        "brain_only",
        "doi",
        "cell_types",
        "em_body_ids",
        "genotype",
    }
    row = df.set_index("line").loc["SS02299"]
    assert list(row["cell_types"]) == ["DNp01", "DNp02"]
    assert list(row["em_body_ids"]) == [1234567890, 2345678901]
    assert df.set_index("line").loc["SS00090", "quality"] == 1
    assert prov["counts"] == {"lines": 5, "adult_with_cell_types": 3, "with_em_body_ids": 2}
    assert (out / "provenance.json").exists()


def test_catalog_queries_by_type_line_and_body(fake_xlsx: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_meissner_lines(fake_xlsx, out)
    cat = LineCatalog(out)
    hits = cat.lines_for_type("DNp01")
    assert list(hits["line"]) == ["SS02299", "SS00727"]  # best quality first
    assert set(hits.columns) >= {"line", "cell_types", "quality", "doi", "specific"}
    assert bool(hits.set_index("line").loc["SS00727", "specific"]) is True  # only one type listed
    assert bool(hits.set_index("line").loc["SS02299", "specific"]) is False
    assert cat.lines_for_type("DNp*", min_quality=1)["line"].tolist() == ["SS02299"]
    assert cat.types_for_line("SS02299") == ["DNp01", "DNp02"]
    assert cat.lines_for_body(1234567890)["line"].tolist() == ["SS02299", "SS00727"]
    assert cat.lines_for_type("nosuchtype").empty
    assert "Meissner" in cat.citation_text()
