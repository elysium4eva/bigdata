"""l3_02_join_concat.py —— vertically concatenate two extraction files and align their schemas.

Vertical concatenation looks simple until the two files disagree on column names, column order, types or units.
The script stacks two periods of order extracts, first shows what happens without any treatment,
then gives the clean result after explicit alignment, with a source flag column and two assertions.
"""
import numpy as np                                                 # random numbers for both extracts
import pandas as pd                                                # concat, type handling and reconciliation
from _common_l3 import RNG, dump                                   # reuse the shared seed and dump helper

N_Q1, N_Q2 = 240, 260                                              # rows in each extract

q1 = pd.DataFrame({                                                # extract one: fairly regular schema
    "order_id": [f"Q1-{1000 + i}" for i in range(N_Q1)],           # order id
    "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(RNG.integers(0, 90, N_Q1), "D"),   # date column
    "store_id": RNG.choice([f"S{i:02d}" for i in range(1, 9)], N_Q1),   # store code
    "amount": np.round(RNG.lognormal(6.0, 0.6, N_Q1), 2),          # amount (yuan)
    "qty": RNG.integers(1, 9, N_Q1),                               # quantity (integer)
})                                                                 # end of extract one
q2 = pd.DataFrame({                                                # extract two: slightly different schema
    "OrderID": [f"Q2-{2000 + i}" for i in range(N_Q2)],            # primary key has another name
    "date": (pd.to_datetime("2026-04-01") + pd.to_timedelta(RNG.integers(0, 91, N_Q2), "D")).strftime("%d/%m/%Y"),   # day/month/year
    "store": RNG.choice([f"S{i:02d}" for i in range(1, 9)], N_Q2), # store column name differs too
    "amount": [f"¥{v:,.2f}" for v in RNG.lognormal(6.1, 0.6, N_Q2)],   # amount is a symbol string
    "quantity": RNG.integers(1, 9, N_Q2),                         # quantity column name differs
    "channel": RNG.choice(["online", "store"], N_Q2),              # only this extract has it
})                                                                 # end of extract two
dump(q1, "l3_orders_2026Q1.csv")                                   # save extract one
dump(q2, "l3_orders_2026Q2.csv")                                   # save extract two

before = pd.DataFrame({"extract": ["q1", "q2", "q1+q2"],           # baseline reconciliation table
                       "rows": [len(q1), len(q2), len(q1) + len(q2)],   # row counts
                       "missing_cells": [int(q1.isna().sum().sum()),   # missing cells in extract one
                                         int(q2.isna().sum().sum()),   # missing cells in extract two
                                         0]})                      # the total row is not a real extract
print("columns q1:", list(q1.columns))                             # print extract one columns
print("columns q2:", list(q2.columns))                             # print extract two columns
print("columns only in q1:", sorted(set(q1.columns) - set(q2.columns)))   # one-way schema difference
print("columns only in q2:", sorted(set(q2.columns) - set(q1.columns)))   # reverse schema difference

# ---------------------------------------------------------------- 1) naive concat: where it breaks
naive = pd.concat([q1, q2], ignore_index=True)                     # stack the two frames directly
print("naive concat shape:", naive.shape)                          # 500 x 9: renames create empty columns
print("naive missing per column:\n", naive.isna().sum())           # every renamed column becomes a hole

# ---------------------------------------------------------------- 2) align the schemas explicitly
RENAME = {"OrderID": "order_id", "store": "store_id", "quantity": "qty"}   # old name -> standard name
def align(df, source):                                             # bring one extract to the target schema
    """Rename, unify types and tag the source so both files share one schema."""   # function contract
    out = df.rename(columns=RENAME).copy()                         # rename first, then work on a copy
    out["date"] = pd.to_datetime(out["date"], format="mixed", dayfirst=True, errors="coerce")   # unify dates
    out["amount"] = pd.to_numeric(                                 # amounts still carry currency symbols
        out["amount"].astype(str).str.replace(r"[^\d.\-]", "", regex=True),   # strip symbols and spaces first
        errors="coerce")                                           # unparsable values become NaN
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").astype("Int64")   # nullable integer type
    out["order_id"] = out["order_id"].astype("string")             # primary key as string type
    out["store_id"] = out["store_id"].astype("string")             # store column as string too
    out["channel"] = out.get("channel", pd.Series(pd.NA, index=out.index)).astype("string")   # fill in the missing column
    out["source"] = source                                         # source column: which file this row came from
    return out[["order_id", "date", "store_id", "amount", "qty", "channel", "source"]]   # unify the column order
combined = pd.concat([align(q1, "Q1"), align(q2, "Q2")], ignore_index=True)   # 500 x 7
print("combined shape:", combined.shape)                           # rows kept, columns unified
print("rows per source:", combined["source"].value_counts().to_dict())   # reconciliation: rows per extract
assert len(combined) == len(q1) + len(q2), "concat lost or invented rows"   # row count conservation
assert combined["order_id"].is_unique, "duplicate order ids after concat"   # primary key uniqueness
dump(combined, "l3_orders_combined.csv")                           # save the aligned combined table
dump(before, "l3_concat_accounting.csv")                           # save the reconciliation table
print(combined.head(3).to_string())                                # print a few rows for checking
