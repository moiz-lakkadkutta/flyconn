# %% [markdown]
# # W1 - Circuit tracing with signs and confidence
#
# Sugar gustatory receptor neurons (GRNs) to the proboscis motor neuron MN9,
# 1-3 hops, in FlyWire (Shiu et al. 2024 v630 export) and in MaleCNS v1.0,
# with predicted signs, weights and stability under synapse thresholds and
# neurotransmitter uncertainty.
#
# Everything here is wiring plus *predicted* transmitters; nothing is measured
# activity. Run `flyconn data pull shiu@630 --level weights` and
# `flyconn data pull malecns@1.0 --level weights` first (or let `pull` do it).

# %%
import json
from pathlib import Path

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy, find_paths
from flyconn.uncertainty import path_stability

SETS = json.loads(Path("tests/fixtures/shiu_2024_neuron_sets.json").read_text())["sets"]

# %% [markdown]
# ## FlyWire: explicit sugar GRN ids (Shiu et al. 2024) → MN9

# %%
fw = ConnectivityMatrix.from_store(
    Store(pull("shiu@630", level="weights").store_dir), min_weight=5, sign_policy=SignPolicy.ARGMAX
)
paths = find_paths(fw, SETS["sugar_grn"], SETS["mn9"], max_hops=3, min_edge_fraction=0.01)
print(len(paths), "paths;", paths["path_sign"].value_counts().to_dict())
print(paths.head())

# %%
stab = path_stability(
    fw,
    SETS["sugar_grn"],
    SETS["mn9"],
    max_hops=3,
    thresholds=[5, 10, 20],
    n_samples=100,
    seed=0,
    min_edge_fraction=0.01,
)
print(stab.summary())

# %% [markdown]
# ## MaleCNS: labellar `LB3*` GRNs (sugar **and** water; no sugar-only type exists) → MN9

# %%
mc_store = Store(pull("malecns@1.0", level="weights").store_dir)
mc = ConnectivityMatrix.from_store(mc_store, min_weight=5, sign_policy=SignPolicy.ARGMAX)
meta = mc.meta
sources = meta.index[meta["cell_type"].astype(str).str.match(r"^LB3")].to_numpy()
targets = meta.index[meta["cell_type"] == "MN9"].to_numpy()
mc_paths = find_paths(mc, sources, targets, max_hops=3, min_edge_fraction=0.01, label="cell_type")
print(len(mc_paths), "paths;", mc_paths["path_sign"].value_counts().to_dict())
print(mc_paths[["labels", "hops", "weights", "path_sign", "strength"]].head(10).to_string())

# %%
mc_stab = path_stability(
    mc,
    sources,
    targets,
    max_hops=3,
    thresholds=[5, 10, 20],
    n_samples=100,
    seed=0,
    min_edge_fraction=0.01,
    label="cell_type",
)
print(mc_stab.summary())
print("NT model per neuron used for sampling:", mc_stab.provenance["nt_model"])
print("Citations:\n" + mc_store.citation_text())
