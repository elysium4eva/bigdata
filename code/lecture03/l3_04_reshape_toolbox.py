"""l3_04_reshape_toolbox.py —— wide and long tables, pivoting and stacking, plus the encoding toolbox.

The same sales data has three shapes: wide (store x month), long (tidy data) and pivot (aggregated result).
The script moves back and forth between the three shapes and demonstrates each tool listed on the slide
"Reshaping and pivot tables" (stack/unstack, explode, get_dummies/from_dummies, cut/qcut, crosstab, factorize).
"""
import numpy as np                                                 # random numbers for the sales sample
import pandas as pd                                                # reshaping and encoding tools
import matplotlib.pyplot as plt                                    # figure comparing the three shapes
from _common_l3 import RNG, save, dump                             # reuse shared seed, plot and dump helpers

STORES = ["S01", "S02", "S03", "S04"]                              # four stores
MONTHS = [f"2026-{m:02d}" for m in range(1, 7)]                    # six months
wide = pd.DataFrame(                                                # wide table: one row per store
    RNG.lognormal(mean=4.2, sigma=0.35, size=(len(STORES), len(MONTHS))).round(1),   # monthly sales
    index=STORES, columns=MONTHS)                                  # rows are stores, columns are months
dump(wide.reset_index(names="store_id"), "l3_sales_wide.csv")      # save the wide sample

long = wide.reset_index(names="store_id").melt(                    # long table: one store-month per row
    id_vars="store_id", var_name="month", value_name="sales")       # keep column, key name and value name
dump(long, "l3_sales_long.csv")                                    # save the long sample
back_to_wide = long.pivot(index="store_id", columns="month", values="sales")   # long back to wide
assert back_to_wide.shape == wide.shape, "pivot did not invert melt"   # round trip: same shape
assert np.allclose(back_to_wide.values, wide.values), "values changed in the round trip"   # values unchanged

long["category"] = RNG.choice(["food", "drink", "home"], len(long))   # add a category column for aggregation
pivot = long.pivot_table(index="store_id", columns="category",      # pivot store x category
                         values="sales", aggfunc="mean").round(1)   # default aggregation is the mean
pivot["total"] = pivot.sum(axis=1).round(1)                        # append a per-store total column
counts = pd.crosstab(long["store_id"], long["category"])           # crosstab counts frequencies
long["bucket"] = pd.cut(long["sales"], bins=[0, 40, 80, 140, 1000],   # equal-width binning
                        labels=["<40", "40-80", "80-140", ">140"])     # readable bin labels
long["quartile"] = pd.qcut(long["sales"], 4, labels=["Q1", "Q2", "Q3", "Q4"])   # equal-frequency binning
long["cat_code"] = pd.factorize(long["category"])[0]                # category to integer code
long["cat_label"] = pd.factorize(long["category"])[1][long["cat_code"]]   # decode the category name back
dummies = pd.get_dummies(long["category"], prefix="cat")            # one-hot encoding
restored = pd.from_dummies(dummies).rename(columns=lambda c: "category")   # invertible from pandas 2.1
stacked = pivot.drop(columns="total").stack()                       # stack: columns down to rows
unstacked = stacked.unstack()                                       # unstack: restore the shape
assert unstacked.shape == pivot.drop(columns="total").shape, "stack/unstack round trip failed"   # inverse check
items = pd.DataFrame({"order_id": ["O1", "O2", "O3"],               # explode needs list-like cells
                      "items": [["pen", "book"], ["mug"], ["pen", "mug", "bag"]]})   # several items per order
exploded = items.explode("items")                                   # one order becomes several rows
print("wide -> long -> wide round trip OK:", back_to_wide.shape)    # print the round trip result
print("explode: 3 orders ->", len(exploded), "item rows")           # print rows after explode
print("buckets:\n", long["bucket"].value_counts().sort_index())     # equal-width bin distribution
print("quartiles:", long["quartile"].value_counts().sort_index().to_dict())   # equal-frequency distribution
print("category codes:", dict(zip(long["cat_label"].head(3), long["cat_code"].head(3))))   # encoding lookup
print("dummies restored == original:", bool((restored["category"].values   # verify one-hot is invertible
                                            == dummies.idxmax(axis=1).str.replace("cat_", "").values).all()))   # row by row
print("pivot table:\n", pivot.to_string())                          # print the pivot table
dump(long, "l3_sales_long_enriched.csv")                            # save the binned long table
dump(pivot.reset_index(), "l3_sales_pivot.csv")                     # save the pivot table

# ---------------------------------------------------------------- figure A: the three shapes
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # wide, long and pivot side by side
im = axes[0].imshow(wide.values, cmap="RdPu", aspect="auto")        # the wide table as a heat map
axes[0].set_xticks(range(len(MONTHS)), MONTHS, rotation=45, fontsize=8)   # x axis: months
axes[0].set_yticks(range(len(STORES)), STORES, fontsize=9)          # y axis: stores
axes[0].set_title("wide: stores x months")                          # subplot title
axes[0].grid(False)                                                 # a heat map needs no grid
plt.colorbar(im, ax=axes[0], fraction=0.046)                        # numeric colour bar
axes[1].plot(range(len(long)), long["sales"], ".", ms=4, color="#8A0C3C", alpha=0.6)   # one row per observation
axes[1].set_xlabel("row index (one row = store-month)")             # what one row represents
axes[1].set_title("long: one row per store-month")                  # subplot title
im2 = axes[2].imshow(pivot.drop(columns="total").values, cmap="YlGnBu", aspect="auto")   # the pivot result
axes[2].set_xticks(range(pivot.shape[1] - 1), pivot.columns[:-1], fontsize=8)   # x axis: categories
axes[2].set_yticks(range(len(STORES)), STORES, fontsize=9)          # y axis: stores
axes[2].set_title("pivot_table: store x category")                  # subplot title
axes[2].grid(False)                                                 # a heat map needs no grid
plt.tight_layout()                                                  # adjust the three subplots
save(fig, "fig_reshape_wide_long.png")                              # save: reshaping slide figure

# ---------------------------------------------------------------- figure B: bins, crosstab and encoding
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # crosstab / cut / qcut
im3 = axes[0].imshow(counts.values, cmap="OrRd", aspect="auto")     # crosstab heat map
axes[0].set_xticks(range(counts.shape[1]), counts.columns, fontsize=8)   # x axis: categories
axes[0].set_yticks(range(len(STORES)), STORES, fontsize=9)          # y axis: stores
axes[0].set_title("crosstab: store x category")                     # subplot title
axes[0].grid(False)                                                 # a heat map needs no grid
axes[1].bar(long["bucket"].value_counts().sort_index().index.astype(str),   # labels in bin order
            long["bucket"].value_counts().sort_index().values, color="#8A0C3C", alpha=0.85)   # equal-width counts
axes[1].set_title("cut(): equal-width bins")                        # subplot title
axes[2].bar(long["quartile"].value_counts().sort_index().index.astype(str),   # labels in quantile order
            long["quartile"].value_counts().sort_index().values, color="#C9A227", alpha=0.9)   # equal-frequency counts
axes[2].set_title("qcut(): equal-count bins")                       # subplot title
plt.tight_layout()                                                  # adjust the three subplots
save(fig, "fig_reshape_bins.png")                                   # save: reshaping toolbox slide figure
