"""l3_03_join_types_validate.py -- inner / left / right / full outer joins and key validation.

The script rebuilds the lecture's two-dataset example, shows what each join type
keeps or drops, and demonstrates how pandas' validate= argument turns a silent
row explosion into a loud, explicit error.
"""
import numpy as np                                                 # random draws for the sample tables
import pandas as pd                                                # merge and validation logic
import matplotlib.pyplot as plt                                    # grid figure of the join results
from _common_l3 import RNG, save, dump                             # shared seed, figure and data writers

# ---------------------------------------------------------------- 1) the lecture's Dataset 1 / Dataset 2
dataset1 = pd.DataFrame({"person": ["A", "B", "C", "D"],           # Dataset 1: one row per person
                         "age": [21, 57, 35, 23]})                 # age in years
dataset2 = pd.DataFrame({"person": ["A", "B", "C", "D"],           # Dataset 2: the questionnaire part two
                         "sex": ["male", "female", "male", "female"]})   # sex of the same persons
dataset3 = pd.DataFrame({"person": ["A", "A", "B", "C", "C", "C", "E"],   # Dataset 3: repeated persons
                         "visit": [1, 2, 1, 1, 2, 3, 1],           # visit number inside the person
                         "spend": [120.0, 80.5, 240.0, 60.0, 95.5, 45.0, 300.0]})   # spend per visit
dataset4 = pd.DataFrame({"person": ["E", "F"],                     # Dataset 4: two extra persons
                         "age": [41, 29]})                         # their ages, appended below
dump(dataset1, "l3_dataset1.csv")                                  # publish the four teaching datasets
dump(dataset2, "l3_dataset2.csv")                                  # publish Dataset 2
dump(dataset3, "l3_dataset3.csv")                                  # publish Dataset 3
dump(dataset4, "l3_dataset4.csv")                                  # publish Dataset 4
print("merge 1:1 (Stata: merge 1:1 person using dataset2):")       # mirror the original Stata command
print(dataset1.merge(dataset2, on="person", how="inner", validate="1:1").to_string(index=False))   # 1:1 merge
print("merge m:1 (Stata: merge m:1 person using dataset2):")       # many visits per person
print(dataset3.merge(dataset2, on="person", how="left", validate="m:1").head().to_string(index=False))   # m:1
print("append (Stata: append using dataset4):")                    # vertical append keeps all rows
print(pd.concat([dataset1, dataset4], ignore_index=True).to_string(index=False))   # same as append

# ---------------------------------------------------------------- 2) join semantics on business keys
customers = pd.DataFrame({                                         # left table: one row per customer
    "customer_id": ["C001", "C002", "C003", "C004"],               # customer key
    "city": ["Shenzhen", "Guangzhou", "Beijing", "Chengdu"],       # customer city
    "segment": ["VIP", "Regular", "VIP", "Regular"],               # customer segment
})                                                                 # end of the left table
orders = pd.DataFrame({                                            # right table: many rows per customer
    "order_id": [f"O{i:03d}" for i in range(1, 9)],                # order key
    "customer_id": ["C001", "C001", "C002", "C003", "C003", "C005", "C005", "C006"],   # some ids unmatched
    "amount": np.round(RNG.lognormal(5.6, 0.7, 8), 2),             # order amount in yuan
})                                                                 # end of the right table
dump(customers, "l3_customers.csv")                                # publish the customers sample
dump(orders, "l3_orders.csv")                                      # publish the orders sample
JOINS = ["inner", "left", "right", "outer"]                        # the four join types of the lecture
results = {}                                                       # collect one result per join type
for how in JOINS:                                                  # run each join on the same keys
    res = customers.merge(orders, on="customer_id", how=how,       # merge on the shared business key
                          indicator=True)                          # indicator records where each row came from
    results[how] = res                                             # keep the result for the figure
    print(f"{how:6s} rows={len(res):2d}  "                        # report row count and provenance mix
          f"provenance={res['_merge'].value_counts().to_dict()}")  # both / left_only / right_only
union = pd.concat([orders, orders.iloc[:2]], ignore_index=True)    # union = row append, not a join at all
print("union rows (orders + 2 duplicated orders):", len(union))    # union keeps everything, duplicates included

# ---------------------------------------------------------------- 3) validate= turns silence into an error
dup_keys = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)   # customer C001 appears twice
try:                                                               # expect a loud failure, not a silent blow-up
    dup_keys.merge(orders, on="customer_id", how="left", validate="1:1")   # 1:1 promised but violated
    print("validate=1:1 did NOT complain -- this would be a silent row explosion")   # unreachable if correct
except pd.errors.MergeError as err:                                # pandas raises MergeError for broken promises
    print("validate=1:1 correctly raised:", str(err).splitlines()[0])   # log the first line of the error
exploded = dup_keys.merge(orders, on="customer_id", how="left")    # without validate: rows multiply quietly
print("rows without validate:", len(exploded), "vs customers:", len(dup_keys))   # the row explosion in numbers
safe = orders.merge(customers, on="customer_id", how="left", validate="m:1")   # many orders -> one customer
print("rows with validate=m:1:", len(safe), "(one row per order, customer attributes attached)")   # honest m:1

# ---------------------------------------------------------------- figure: join grid + row counts
fig, axes = plt.subplots(2, 3, figsize=(13.6, 4.6))                # five grids plus one bar chart
panels = JOINS + ["union"]                                         # the five operations shown on the slide
for ax, name in zip(axes.ravel(), panels):                         # draw each operation in its own panel
    ax.axis("off")                                                 # tables need no axes
    if name == "union":                                            # union is a row append, so show that table
        data = union[["order_id", "customer_id"]].head(6)          # first six rows of the union result
        title = "union: append rows (keeps duplicates)"            # explain the semantics in the title
    else:                                                          # otherwise render the merge result
        res = results[name]                                        # take the stored result
        data = res[["customer_id", "order_id", "amount"]].sort_values("customer_id").head(6)   # six rows
        nulls = int(res["amount"].isna().sum())                    # how many rows carry a NULL after the join
        title = f"{name} join: {len(res)} rows, {nulls} unmatched"  # row count and NULL count in the title
    tbl = ax.table(cellText=data.fillna("NULL").astype(str).values,   # render the small result table
                   colLabels=list(data.columns), loc="center", cellLoc="center")   # centred cells
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.25)   # readable but compact
    ax.set_title(title, fontsize=10.5)                             # panel title carries the semantics
axes[1, 2].axis("on")                                              # the last panel is a bar chart
counts = [len(results[j]) for j in JOINS] + [len(union)]          # row counts for all five operations
axes[1, 2].bar(JOINS + ["union"], counts, color="#8A0C3C", alpha=0.85)   # compare sizes at a glance
axes[1, 2].set_title("rows returned per operation", fontsize=10.5)   # panel title
plt.tight_layout()                                                 # balance the six panels
save(fig, "fig_join_types.png")                                    # write the figure for the deck
