"""l3_04_reshape_toolbox.py -- wide and long layouts, pivoting, stacking and encoding tools.

One sales table appears in three shapes: wide (stores x months), long (tidy) and
aggregated (pivot table). The script moves between the shapes, then demonstrates
the remaining tools from the original lecture slide: stack/unstack, explode,
get_dummies/from_dummies, cut/qcut, crosstab and factorize.
"""
import numpy as np                                                 # random draws for the sales sample
import pandas as pd                                                # reshaping and encoding tools
import matplotlib.pyplot as plt                                    # layout comparison figures
from _common_l3 import RNG, save, dump                             # shared seed, figure and data writers

STORES = ["S01", "S02", "S03", "S04"]                              # four stores in the sample
MONTHS = [f"2026-{m:02d}" for m in range(1, 7)]                    # six months of sales
wide = pd.DataFrame(                                                # wide layout: one row per store
    RNG.lognormal(mean=4.2, sigma=0.35, size=(len(STORES), len(MONTHS))).round(1),   # monthly amounts
    index=STORES, columns=MONTHS)                                  # index = store, columns = month
dump(wide.reset_index(names="store_id"), "l3_sales_wide.csv")      # publish the wide sample

long = wide.reset_index(names="store_id").melt(                    # long (tidy) layout: one row per store-month
    id_vars="store_id", var_name="month", value_name="sales")       # which column keeps the identifier
dump(long, "l3_sales_long.csv")                                    # publish the long sample
back_to_wide = long.pivot(index="store_id", columns="month", values="sales")   # pivot long back to wide
assert back_to_wide.shape == wide.shape, "pivot did not invert melt"   # round-trip check: same shape
assert np.allclose(back_to_wide.values, wide.values), "values changed in the round trip"   # same values

long["category"] = RNG.choice(["food", "drink", "home"], len(long))   # add a category to aggregate over
pivot = long.pivot_table(index="store_id", columns="category",      # aggregate sales by store x category
                         values="sales", aggfunc="mean").round(1)   # means, rounded for the slide
pivot["total"] = pivot.sum(axis=1).round(1)                        # marginal total per store
counts = pd.crosstab(long["store_id"], long["category"])           # crosstab: frequency table
long["bucket"] = pd.cut(long["sales"], bins=[0, 40, 80, 140, 1000],   # equal-width bins on a numeric column
                        labels=["<40", "40-80", "80-140", ">140"])     # readable bucket labels
long["quartile"] = pd.qcut(long["sales"], 4, labels=["Q1", "Q2", "Q3", "Q4"])   # equal-count bins
long["cat_code"] = pd.factorize(long["category"])[0]                # integer encoding of a category column
long["cat_label"] = pd.factorize(long["category"])[1][long["cat_code"]]   # decode back to show the mapping
dummies = pd.get_dummies(long["category"], prefix="cat")            # one-hot encoding of the category
restored = pd.from_dummies(dummies).rename(columns=lambda c: "category")   # pandas >= 2.1 inverts one-hot
stacked = pivot.drop(columns="total").stack()                       # stack: columns become a row level
unstacked = stacked.unstack()                                       # unstack: bring the level back
assert unstacked.shape == pivot.drop(columns="total").shape, "stack/unstack round trip failed"   # invariant
items = pd.DataFrame({"order_id": ["O1", "O2", "O3"],               # explode needs list-like cells
                      "items": [["pen", "book"], ["mug"], ["pen", "mug", "bag"]]})   # multi-item orders
exploded = items.explode("items")                                   # one row per item instead of one per order
print("wide -> long -> wide round trip OK:", back_to_wide.shape)    # report the invariant in the log
print("explode: 3 orders ->", len(exploded), "item rows")           # row count after explode
print("buckets:\n", long["bucket"].value_counts().sort_index())     # distribution across the buckets
print("quartiles:", long["quartile"].value_counts().sort_index().to_dict())   # equal-count bins
print("category codes:", dict(zip(long["cat_label"].head(3), long["cat_code"].head(3))))   # encode/decode
print("dummies restored == original:", bool((restored["category"].values   # compare the inverse encoding
                                            == dummies.idxmax(axis=1).str.replace("cat_", "").values).all()))   # compare
print("pivot table:\n", pivot.to_string())                          # aggregated view for the slide
dump(long, "l3_sales_long_enriched.csv")                            # publish the enriched long sample
dump(pivot.reset_index(), "l3_sales_pivot.csv")                     # publish the aggregated pivot table

# ---------------------------------------------------------------- figure A: the three layouts
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # wide / long / pivot side by side
im = axes[0].imshow(wide.values, cmap="RdPu", aspect="auto")        # wide layout as a heat map
axes[0].set_xticks(range(len(MONTHS)), MONTHS, rotation=45, fontsize=8)   # month labels
axes[0].set_yticks(range(len(STORES)), STORES, fontsize=9)          # store labels
axes[0].set_title("wide: stores x months")                          # panel title
axes[0].grid(False)                                                 # heat maps do not need a grid
plt.colorbar(im, ax=axes[0], fraction=0.046)                        # sales scale
axes[1].plot(range(len(long)), long["sales"], ".", ms=4, color="#8A0C3C", alpha=0.6)   # long layout: one row
axes[1].set_xlabel("row index (one row = store-month)")             # tidy data has one observation per row
axes[1].set_title("long: one row per store-month")                  # panel title
im2 = axes[2].imshow(pivot.drop(columns="total").values, cmap="YlGnBu", aspect="auto")   # pivot heat map
axes[2].set_xticks(range(pivot.shape[1] - 1), pivot.columns[:-1], fontsize=8)   # category labels
axes[2].set_yticks(range(len(STORES)), STORES, fontsize=9)          # store labels
axes[2].set_title("pivot_table: store x category")                  # panel title
axes[2].grid(False)                                                 # heat maps do not need a grid
plt.tight_layout()                                                  # balance the three panels
save(fig, "fig_reshape_wide_long.png")                              # figure for the reshape slide

# ---------------------------------------------------------------- figure B: bins, counts and codes
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # crosstab / cut / qcut panels
im3 = axes[0].imshow(counts.values, cmap="OrRd", aspect="auto")     # crosstab heat map
axes[0].set_xticks(range(counts.shape[1]), counts.columns, fontsize=8)   # category labels
axes[0].set_yticks(range(len(STORES)), STORES, fontsize=9)          # store labels
axes[0].set_title("crosstab: store x category")                     # panel title
axes[0].grid(False)                                                 # heat maps do not need a grid
axes[1].bar(long["bucket"].value_counts().sort_index().index.astype(str),   # ordered bucket labels
            long["bucket"].value_counts().sort_index().values, color="#8A0C3C", alpha=0.85)   # cut() bins
axes[1].set_title("cut(): equal-width bins")                        # panel title
axes[2].bar(long["quartile"].value_counts().sort_index().index.astype(str),   # ordered quartile labels
            long["quartile"].value_counts().sort_index().values, color="#C9A227", alpha=0.9)   # qcut() bins
axes[2].set_title("qcut(): equal-count bins")                       # panel title
plt.tight_layout()                                                  # balance the three panels
save(fig, "fig_reshape_bins.png")                                   # figure for the toolbox slide
