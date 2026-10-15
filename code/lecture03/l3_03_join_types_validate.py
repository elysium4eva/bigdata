"""l3_03_join_types_validate.py —— inner/left/right/full outer joins, plus cardinality checks on the join keys.

The script reproduces the two teaching datasets from the slides (Dataset 1 / 2 / 3 / 4), showing one by one what each join keeps or drops,
and how the pandas validate= argument turns a silently exploding row count into an explicit error.
"""
import numpy as np                                                 # random numbers for the sample tables
import pandas as pd                                                # join and validation logic
import matplotlib.pyplot as plt                                    # figure comparing join results
from _common_l3 import RNG, save, dump                             # reuse shared seed, plot and dump helpers

# ---------------------------------------------------------------- 1) Dataset 1 / Dataset 2 from the slides
dataset1 = pd.DataFrame({"person": ["A", "B", "C", "D"],           # Dataset 1: one row per person
                         "age": [21, 57, 35, 23]})                 # age
dataset2 = pd.DataFrame({"person": ["A", "B", "C", "D"],           # Dataset 2: second questionnaire part
                         "sex": ["male", "female", "male", "female"]})   # sex of the same people
dataset3 = pd.DataFrame({"person": ["A", "A", "B", "C", "C", "C", "E"],   # Dataset 3: several rows per person
                         "visit": [1, 2, 1, 1, 2, 3, 1],           # visit number
                         "spend": [120.0, 80.5, 240.0, 60.0, 95.5, 45.0, 300.0]})   # spend per visit
dataset4 = pd.DataFrame({"person": ["E", "F"],                     # Dataset 4: two new people
                         "age": [41, 29]})                         # their ages, used for append
dump(dataset1, "l3_dataset1.csv")                                  # save the four teaching datasets
dump(dataset2, "l3_dataset2.csv")                                  # save Dataset 2
dump(dataset3, "l3_dataset3.csv")                                  # save Dataset 3
dump(dataset4, "l3_dataset4.csv")                                  # save Dataset 4
print("merge 1:1 (Stata: merge 1:1 person using dataset2):")       # the Stata syntax from the slides
print(dataset1.merge(dataset2, on="person", how="inner", validate="1:1").to_string(index=False))   # 1:1 merge
print("merge m:1 (Stata: merge m:1 person using dataset2):")       # one person, many visits
print(dataset3.merge(dataset2, on="person", how="left", validate="m:1").head().to_string(index=False))   # m:1
print("append (Stata: append using dataset4):")                    # vertical append keeps all rows
print(pd.concat([dataset1, dataset4], ignore_index=True).to_string(index=False))   # equivalent to append

# ---------------------------------------------------------------- 2) join semantics on a business key
customers = pd.DataFrame({                                         # left table: one row per customer
    "customer_id": ["C001", "C002", "C003", "C004"],               # customer key
    "city": ["Shenzhen", "Guangzhou", "Beijing", "Chengdu"],       # customer city
    "segment": ["VIP", "Regular", "VIP", "Regular"],               # customer segment
})                                                                 # end of the left table
orders = pd.DataFrame({                                            # right table: several orders per customer
    "order_id": [f"O{i:03d}" for i in range(1, 9)],                # order key
    "customer_id": ["C001", "C001", "C002", "C003", "C003", "C005", "C005", "C006"],   # some keys do not match
    "amount": np.round(RNG.lognormal(5.6, 0.7, 8), 2),             # order amount (yuan)
})                                                                 # end of the right table
dump(customers, "l3_customers.csv")                                # save the customer sample
dump(orders, "l3_orders.csv")                                      # save the order sample
JOINS = ["inner", "left", "right", "outer"]                        # the four joins from the slides
results = {}                                                       # store each join result
for how in JOINS:                                                  # run all four joins on the same key
    res = customers.merge(orders, on="customer_id", how=how,       # merge on the business key
                          indicator=True)                          # indicator records each row origin
    results[how] = res                                             # keep the result for plotting
    print(f"{how:6s} rows={len(res):2d}  "                        # print the row count
          f"provenance={res['_merge'].value_counts().to_dict()}")  # print the provenance breakdown
union = pd.concat([orders, orders.iloc[:2]], ignore_index=True)    # union appends rows, it is not a join
print("union rows (orders + 2 duplicated orders):", len(union))    # union keeps everything, duplicates included

# ---------------------------------------------------------------- 3) validate= makes errors explicit
dup_keys = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)   # customer C001 appears twice
try:                                                               # expect an error here, not silent inflation
    dup_keys.merge(orders, on="customer_id", how="left", validate="1:1")   # declared 1:1, actually not true
    print("validate=1:1 did NOT complain -- this would be a silent row explosion")   # normally unreachable
except pd.errors.MergeError as err:                                # pandas raises MergeError on violation
    print("validate=1:1 correctly raised:", str(err).splitlines()[0])   # print the first error line
exploded = dup_keys.merge(orders, on="customer_id", how="left")    # without validate rows silently double
print("rows without validate:", len(exploded), "vs customers:", len(dup_keys))   # evidence: 5 rows become 8
safe = orders.merge(customers, on="customer_id", how="left", validate="m:1")   # many-to-one: many orders, unique customers
print("rows with validate=m:1:", len(safe), "(one row per order, customer attributes attached)")   # row count unchanged

# ---------------------------------------------------------------- figure: join comparison + row counts
fig, axes = plt.subplots(2, 3, figsize=(13.6, 4.6))                # five result tables + one row-count bar chart
panels = JOINS + ["union"]                                         # the five operations shown in class
for ax, name in zip(axes.ravel(), panels):                         # one panel per operation
    ax.axis("off")                                                 # a table needs no axes
    if name == "union":                                            # union appends rows, show the result table
        data = union[["order_id", "customer_id"]].head(6)          # take the first six rows
        title = "union: append rows (keeps duplicates)"            # the title states the semantics
    else:                                                          # the other four are joins
        res = results[name]                                        # take the stored result
        data = res[["customer_id", "order_id", "amount"]].sort_values("customer_id").head(6)   # first six rows
        nulls = int(res["amount"].isna().sum())                    # how many nulls the join creates
        title = f"{name} join: {len(res)} rows, {nulls} unmatched"  # title gives rows and nulls
    tbl = ax.table(cellText=data.fillna("NULL").astype(str).values,   # render a small result table
                   colLabels=list(data.columns), loc="center", cellLoc="center")   # centre everything
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.25)   # compact but readable font
    ax.set_title(title, fontsize=10.5)                             # the title explains the join semantics
axes[1, 2].axis("on")                                              # last panel becomes a bar chart
counts = [len(results[j]) for j in JOINS] + [len(union)]          # result rows per operation
axes[1, 2].bar(JOINS + ["union"], counts, color="#8A0C3C", alpha=0.85)   # see the row count change at a glance
axes[1, 2].set_title("rows returned per operation", fontsize=10.5)   # subplot title
plt.tight_layout()                                                 # adjust the six subplots
save(fig, "fig_join_types.png")                                    # save for the slides
