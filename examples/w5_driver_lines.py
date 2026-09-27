# %% [markdown]
# # W5 - Which published driver lines target cell type X, and what else might they hit?
#
# On-target: Meissner et al. 2025 split-GAL4 table (CC BY 4.0). Off-target candidates:
# NeuronBridge colour-depth matches (CC BY 4.0), which are similarity candidates, not
# expression calls.

# %%
from flyconn.access import LineCatalog, fetch_line_off_targets
from flyconn.data.pull import pull

catalog = LineCatalog(pull("flylight_lines@meissner2025", level="meta").store_dir)
lines = catalog.lines_for_type("DNp01")
print(lines.to_string(index=False))
print(catalog.citation_text())

# %%
best = lines.iloc[0]["line"]
candidates = fetch_line_off_targets(best, max_images=10)  # network; cached afterwards
print(candidates.attrs["note"])
print(candidates.head(15).to_string(index=False))
