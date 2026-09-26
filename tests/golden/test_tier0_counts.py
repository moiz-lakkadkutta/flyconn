"""Tier 0 golden tests: dataset counts from docs/GOLDEN_RESULTS.md §1 and §6b.

Every expected number here is quoted from that document, which in turn cites the
primary source. Running these converts the real files (downloads them if absent).
"""

from __future__ import annotations

import resource
import time

import pytest

from flyconn.data.pull import pull
from flyconn.data.store import Store

pytestmark = pytest.mark.golden


def _peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if rss > 1e7 else rss / 1e6  # Linux reports KB, macOS bytes


@pytest.fixture(scope="module")
def malecns() -> Store:
    t0 = time.time()
    result = pull("malecns@1.0", level="weights")
    print(
        f"\nmalecns@1.0 pull: converted={result.converted} in {time.time() - t0:.0f}s, "
        f"peak RSS {_peak_rss_gb():.1f} GB"
    )
    return Store(result.store_dir)


@pytest.fixture(scope="module")
def flywire783() -> Store:
    return Store(pull("flywire@783", level="weights").store_dir)


@pytest.fixture(scope="module")
def flywire630() -> Store:
    return Store(pull("flywire@630", level="weights").store_dir)


def test_malecns_neuron_count_matches_cell_abstract(malecns: Store):
    assert malecns.provenance["counts"]["neurons"] == 166_700


def test_malecns_distinct_types(malecns: Store):
    n = malecns.query("SELECT count(DISTINCT cell_type) AS n FROM neurons")["n"].iloc[0]
    assert n == 11_751  # v1.0 file; Cell abstract says 11,710 (GOLDEN_RESULTS §7)


def test_malecns_raw_edge_and_weight_totals(malecns: Store):
    c = malecns.provenance["counts"]
    assert c["raw_edges"] == 151_856_684
    assert c["raw_weight_sum"] == 311_833_243


def test_malecns_neuron_graph_edges_and_thresholds(malecns: Store):
    # Neuron graph over the superclass-defined universe (166,700 neurons). The edge count equals
    # the superclass->superclass count computed independently in Phase 0 (25,582,938). The other
    # values were first computed by this converter on 2026-09-26 and are pinned as regression
    # targets (GOLDEN_RESULTS.md section 6b); the traced-only file gives the lower bounds
    # 25,563,197 edges / 124,025,046 weight / 6,235,682 (>=5) / 2,749,407 (>=10).
    c = malecns.provenance["counts"]
    assert c["edges"] == 25_582_938
    q = malecns.query(
        "SELECT sum(weight) AS w, count(*) FILTER (WHERE weight >= 5) AS ge5, "
        "count(*) FILTER (WHERE weight >= 10) AS ge10 FROM edges"
    ).iloc[0]
    assert q["w"] == 124_177_617
    assert q["ge5"] == 6_242_118
    assert q["ge10"] == 2_753_975


def test_malecns_neuron_totals_are_internally_consistent(malecns: Store):
    t = malecns.query(
        "SELECT sum(input_synapses_total) AS in_tot, sum(input_synapses_neurons) AS in_n, "
        "sum(output_synapses_total) AS out_tot, sum(output_synapses_neurons) AS out_n FROM neurons"
    ).iloc[0]
    assert t["in_n"] == t["out_n"] == 124_177_617  # both equal the neuron-graph weight sum
    assert t["in_tot"] == 130_453_923  # Traced bodies alone carry 130,413,767 PSDs
    assert t["out_tot"] == 295_069_014


def test_malecns_superclass_counts(malecns: Store):
    df = malecns.query(
        "SELECT super_class_raw AS sc, count(*) AS n FROM neurons GROUP BY 1"
    ).set_index("sc")["n"]
    assert df["ol_intrinsic"] == 89_403
    assert df["cb_intrinsic"] == 32_164
    assert df["descending_neuron"] == 1_314
    assert df["vnc_motor"] == 708


def test_malecns_nt_consensus_distribution_for_traced(malecns: Store):
    df = malecns.query(
        "SELECT nt_pred, count(*) AS n FROM neurons WHERE status IN "
        "('Roughly traced','Reviewed','Prelim Roughly traced','RT Hard to trace','Leaves',"
        "'PRT Orphan','RT Orphan') GROUP BY 1"
    ).set_index("nt_pred")["n"]
    # consensus_nt over Traced bodies (docs/research/data_malecns.md §4.2), within 1% because the
    # universe here is superclass-defined
    assert abs(df["acetylcholine"] - 103_718) / 103_718 < 0.01
    assert abs(df["gaba"] - 22_055) / 22_055 < 0.01


def test_flywire_783_counts(flywire783: Store):
    c = flywire783.provenance["counts"]
    assert c["neurons"] == 139_255
    assert c["edges"] == 2_700_513  # Dorkenwald 2024
    assert c["edge_roi_rows"] == 3_869_878
    assert flywire783.query("SELECT min(weight) AS m FROM edges")["m"].iloc[0] == 5


def test_flywire_783_super_class_and_nt(flywire783: Store):
    sc = flywire783.query("SELECT super_class_raw AS s, count(*) AS n FROM neurons GROUP BY 1")
    sc = sc.set_index("s")["n"]
    assert sc["descending"] == 1_305 and sc["motor"] == 110 and sc["endocrine"] == 80
    nt = flywire783.query("SELECT nt_pred, count(*) AS n FROM neurons GROUP BY 1").set_index(
        "nt_pred"
    )["n"]
    assert nt["acetylcholine"] == 82_298 and nt["gaba"] == 16_017 and nt["glutamate"] == 19_605


def test_flywire_630_counts(flywire630: Store):
    c = flywire630.provenance["counts"]
    assert c["neurons"] == 127_978  # Lin 2024
    assert c["edges"] == 2_613_129  # Lin 2024
