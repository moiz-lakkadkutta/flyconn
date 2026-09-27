# %% [markdown]
# # W4 - Version drift: what changed between FlyWire v630 and v783 for my neurons?
#
# Identity is by root id. Ids that no longer exist in the new release are
# reported as unmatched; following them needs a lineage lookup (CAVE), which
# flyconn does not do offline.

# %%
import json
from pathlib import Path

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.uncertainty.versions import diff_versions

SETS = json.loads(Path("tests/fixtures/shiu_2024_neuron_sets.json").read_text())["sets"]
old = Store(pull("flywire@630", level="weights").store_dir)
new = Store(pull("flywire@783", level="weights").store_dir)

# %%
ids = SETS["sugar_grn"] + SETS["mn9"] + SETS["abn1"]
d = diff_versions(old, new, neuron_ids=ids)
print(d.summary())
print("unmatched ids:", d.unmatched_ids)
for c in d.caveats:
    print("caveat:", c)

# %%
print(d.neuron_changes.to_string())
print(d.edge_changes.head(20).to_string())
