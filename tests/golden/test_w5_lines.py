"""W5 (exploratory): which published lines target cell type X, and what else might they hit?"""

from __future__ import annotations

import pytest

from flyconn.access import LineCatalog, fetch_line_off_targets
from flyconn.data.pull import pull

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def catalog() -> LineCatalog:
    return LineCatalog(pull("flylight_lines@meissner2025", level="meta").store_dir)


def test_meissner_table_counts_match_phase0_verification(catalog: LineCatalog):
    c = catalog.provenance["counts"]
    print(f"\nW5 catalog: {c}")
    assert c["lines"] == 4_433
    assert c["adult_with_cell_types"] == 2_667
    assert c["with_em_body_ids_raw"] == 323  # Phase 0 count of non-empty cells
    assert (
        c["with_em_body_ids"] == 320
    )  # three cells hold non-numeric text (kept in em_body_ids_raw)


def test_lines_for_known_types(catalog: LineCatalog):
    dnp01 = catalog.lines_for_type("DNp01")
    print(dnp01.to_string(index=False))
    assert {"SS00727", "SS02299"} <= set(dnp01["line"])
    epg = catalog.lines_for_type("EPG", min_quality=1)
    assert "SS00090" in set(epg["line"])
    assert catalog.types_for_line("SS00090") == ["EPG"]


@pytest.mark.network
def test_off_target_candidates_for_a_dnp01_line_from_neuronbridge():
    df = fetch_line_off_targets("SS02299", max_images=10)
    n_img = df.attrs["n_images"]
    print(f"\nW5 NeuronBridge SS02299: {len(df)} candidate bodies over {n_img} images")
    print(df.head(8).to_string(index=False))
    assert len(df) > 0
    assert df.attrs["note"].startswith("candidate")
    assert set(df["dataset"]) & {"hemibrain", "flywire", "malecns", "manc", "banc", "vnc"}
