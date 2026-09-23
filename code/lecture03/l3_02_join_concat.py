"""l3_02_join_concat.py -- append / concatenate two extracts and align their schemas.

Vertical combination looks trivial until the two files disagree about column
names, column order, data types or units. This script concatenates two quarterly
order extracts, reports the alignment problem, and produces a clean combined
table with an explicit source column.
"""
import numpy as np                                                 # random draws for the two extracts
import pandas as pd                                                # concat, dtype handling, reporting
from _common_l3 import RNG, dump                                   # shared seed and data writer

N_Q1, N_Q2 = 240, 260                                              # rows in each quarterly extract

q1 = pd.DataFrame({                                                # extract #1: the tidy one
    "order_id": [f"Q1-{1000 + i}" for i in range(N_Q1)],           # order identifier
    "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(RNG.integers(0, 90, N_Q1), "D"),   # date column
    "store_id": RNG.choice([f"S{i:02d}" for i in range(1, 9)], N_Q1),   # store code
    "amount": np.round(RNG.lognormal(6.0, 0.6, N_Q1), 2),          # numeric amount in yuan
    "qty": RNG.integers(1, 9, N_Q1),                               # integer quantity
})                                                                 # end of quarterly extract #1
q2 = pd.DataFrame({                                                # extract #2: slightly different schema
    "OrderID": [f"Q2-{2000 + i}" for i in range(N_Q2)],            # different column name for the key
    "date": (pd.to_datetime("2026-04-01") + pd.to_timedelta(RNG.integers(0, 91, N_Q2), "D")).strftime("%d/%m/%Y"),   # dd/mm/yyyy
    "store": RNG.choice([f"S{i:02d}" for i in range(1, 9)], N_Q2), # different column name for the store
    "amount": [f"¥{v:,.2f}" for v in RNG.lognormal(6.1, 0.6, N_Q2)],   # amounts arrive as strings
    "quantity": RNG.integers(1, 9, N_Q2),                         # different column name for quantity
    "channel": RNG.choice(["online", "store"], N_Q2),              # a column only present in Q2
})                                                                 # end of quarterly extract #2
dump(q1, "l3_orders_2026Q1.csv")                                   # publish extract #1 as a sample
dump(q2, "l3_orders_2026Q2.csv")                                   # publish extract #2 as a sample

before = pd.DataFrame({"extract": ["q1", "q2", "q1+q2"],           # baseline accounting before cleaning
                       "rows": [len(q1), len(q2), len(q1) + len(q2)],   # row counts per extract
                       "missing_cells": [int(q1.isna().sum().sum()),   # missing cells in extract #1
                                         int(q2.isna().sum().sum()),   # missing cells in extract #2
                                         0]})                      # the sum row is not a real extract
print("columns q1:", list(q1.columns))                             # show the schema of extract #1
print("columns q2:", list(q2.columns))                             # show the schema of extract #2
print("columns only in q1:", sorted(set(q1.columns) - set(q2.columns)))   # alignment gap, one direction
print("columns only in q2:", sorted(set(q2.columns) - set(q1.columns)))   # alignment gap, other direction

# ---------------------------------------------------------------- 1) naive concat: what goes wrong
naive = pd.concat([q1, q2], ignore_index=True)                     # straight vertical stack
print("naive concat shape:", naive.shape)                          # 4 columns + 6 columns aligned by name
print("naive missing per column:\n", naive.isna().sum())           # every rename creates a block of NaN

# ---------------------------------------------------------------- 2) explicit schema alignment
RENAME = {"OrderID": "order_id", "store": "store_id", "quantity": "qty"}   # mapping old -> canonical names
def align(df, source):                                             # normalise one extract to the target schema
    """Rename, retype and tag one extract so that both files share one schema."""   # helper contract
    out = df.rename(columns=RENAME).copy()                         # rename first, then work on a copy
    out["date"] = pd.to_datetime(out["date"], format="mixed", dayfirst=True, errors="coerce")   # parse dates
    out["amount"] = pd.to_numeric(                                 # amounts may still hold currency symbols
        out["amount"].astype(str).str.replace(r"[^\d.\-]", "", regex=True),   # strip symbols and spaces
        errors="coerce")                                           # unparsable values become NaN
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").astype("Int64")   # nullable integer type
    out["order_id"] = out["order_id"].astype("string")             # explicit string type for the key
    out["store_id"] = out["store_id"].astype("string")             # explicit string type for the store
    out["channel"] = out.get("channel", pd.Series(pd.NA, index=out.index)).astype("string")   # fill missing col
    out["source"] = source                                         # provenance: which file did this row come from
    return out[["order_id", "date", "store_id", "amount", "qty", "channel", "source"]]   # one canonical order

combined = pd.concat([align(q1, "Q1"), align(q2, "Q2")], ignore_index=True)   # order matters for provenance
print("combined shape:", combined.shape)                           # rows kept, columns unified
print("rows per source:", combined["source"].value_counts().to_dict())   # reconciliation check
assert len(combined) == len(q1) + len(q2), "concat lost or invented rows"   # row-conservation assertion
assert combined["order_id"].is_unique, "duplicate order ids after concat"   # key uniqueness assertion
dump(combined, "l3_orders_combined.csv")                           # publish the aligned combined table
dump(before, "l3_concat_accounting.csv")                           # publish the accounting table
print(combined.head(3).to_string())                                # show the first rows in the console
