# %% [markdown]
# # W3 - Is cell type Y wired differently in male (MaleCNS) and female (FlyWire) brains?
#
# Null model: left/right variability within each brain (Schlegel et al. 2024). Verdicts are
# graded against the between-brain effect size those authors report for two *female* brains
# (0.045 +/- 0.096), because two individuals of the same sex already differ by that much.

# %%
from flyconn.compare import compare_type, match_types
from flyconn.data.pull import pull
from flyconn.data.store import Store

male = Store(pull("malecns@1.0", level="weights").store_dir)
female = Store(pull("flywire@783", level="weights").store_dir)
mapping = match_types(male, female)
print(mapping["type_a"].nunique(), "MaleCNS types matched to FlyWire v783")

# %%
for cell_type in ["PFL3", "EPG", "PAM08", "DNp01"]:
    res = compare_type(
        male, female, cell_type, direction="out", n_permutations=500, seed=0, min_weight=5
    )
    print(res.summary())
    print(res.partner_differences.head(5).to_string(index=False))
    for c in res.caveats:
        print("  caveat:", c)
