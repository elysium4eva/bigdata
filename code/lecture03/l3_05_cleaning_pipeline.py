"""l3_05_cleaning_pipeline.py -- the five cleaning steps on a dirty order table.

Step 1 removes duplicates and irrelevant rows, step 2 repairs structural errors,
step 3 classifies (and does not blindly delete) outliers, step 4 handles missing
values, step 5 validates the result. Every step is logged so the cleaning becomes
auditable and reproducible -- the point the lecture makes about templates.
"""
import numpy as np                                                 # numeric helpers and quantiles
import pandas as pd                                                # table operations
import matplotlib.pyplot as plt                                    # before/after figures
from _common_l3 import DAT, save, dump                             # paths plus figure and data writers

raw = pd.read_csv(DAT / "l3_orders_dirty.csv", dtype=str)           # read the sample produced by l3_01
raw["qty"] = pd.to_numeric(raw["qty"], errors="coerce")             # quantity comes back as float
log = []                                                            # one row per cleaning decision
def record(step, before, after, note):                              # helper that appends to the audit log
    """Append one cleaning step to the audit log."""                 # helper contract
    log.append({"step": step, "rows_before": before, "rows_after": after,   # row counts on both sides
                "rows_removed": before - after, "note": note})      # how many rows the step removed
    print(f"{step:38s} {before:5d} -> {after:5d}   {note}")         # echo the step in the console

# ---------------------------------------------------------------- Step 1: duplicates and irrelevant rows
n0 = len(raw)                                                       # baseline row count
df = raw.drop_duplicates().copy()                                   # drop fully duplicated rows
n1 = len(df)                                                        # rows after exact de-duplication
record("Step 1a exact duplicate rows", n0, n1, "identical rows removed")   # log the step
df = df.drop_duplicates(subset="order_id", keep="first")            # keep one row per business key
n2 = len(df)                                                        # rows after key de-duplication
record("Step 1b duplicate business keys", n1, n2, "same order_id kept once")   # log the step
irrelevant = df["order_id"].isna()                                  # irrelevant = no identifier at all
df = df.loc[~irrelevant].copy()                                     # drop rows that cannot be joined back
n3 = len(df)                                                        # rows after relevance filter
record("Step 1c irrelevant observations", n2, n3, "rows without a key removed")   # log the step

# ---------------------------------------------------------------- Step 2: structural errors
city_map = {"shenzhen": "Shenzhen", "shēnzhèn": "Shenzhen",         # normalise every spelling to one form
            "guangzhou city": "Guangzhou", "guang zhou": "Guangzhou"}   # alias table for the sample
def normalise_city(value):                                          # case-insensitive alias lookup
    """Trim, collapse spaces and map an alias to the canonical city name."""   # helper contract
    key = " ".join(str(value).strip().lower().split())              # trim, lower-case and collapse spaces
    return city_map.get(key, key.title())                           # fall back to title case
df["city"] = df["city_raw"].map(normalise_city)                     # apply the mapping column-wise
pay_map = {"paid": "paid", "unpaid": "unpaid", "n/a": "not_applicable",   # three spellings of one state
           "-": "not_applicable", "not applicable": "not_applicable"}   # map them all to one label
df["payment"] = (df["payment_raw"].str.strip().str.lower()          # trim and lower-case first
                 .map(pay_map).fillna("unknown"))                   # then map, unknown stays explicit
df["date"] = pd.to_datetime(df["date_raw"], format="mixed", dayfirst=True, errors="coerce")   # parse dates
bad_dates = int(df["date"].isna().sum())                            # count unparsable dates
record("Step 2 structural errors", n3, len(df), f"cities/payments normalised, {bad_dates} bad dates")   # log
print("city variants:", raw["city_raw"].nunique(), "->", df["city"].nunique())   # consistency evidence
print("payment variants:", raw["payment_raw"].nunique(), "->", df["payment"].nunique())   # uniformity

# ---------------------------------------------------------------- Step 3: outliers (classify, then decide)
amount = pd.to_numeric(df["amount_raw"].astype(str)                 # amounts still carry currency noise
                       .str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")   # strip and coerce
df["amount"] = amount                                               # attach the parsed amount
impossible = (df["amount"] <= 0) | (df["amount"] > 100000)          # rule-based: physically impossible
q1, q3 = df["amount"].quantile(.25), df["amount"].quantile(.75)     # quartiles for the IQR rule
iqr = q3 - q1                                                       # inter-quartile range
statistical = (df["amount"] < q1 - 1.5 * iqr) | (df["amount"] > q3 + 1.5 * iqr)   # statistical outliers
df["flag_impossible"] = impossible                                  # keep the flag, do not delete silently
df["flag_outlier"] = statistical & ~impossible                      # statistical but plausible values
record("Step 3 outlier classification", len(df), len(df),           # no rows are removed at this step
       f"{int(impossible.sum())} impossible, {int(statistical.sum())} statistical outliers")   # log

# ---------------------------------------------------------------- Step 4: missing values
miss_before = df[["qty", "amount", "date"]].isna().mean().mul(100).round(2)   # missing rate before
median_qty = df["qty"].median()                                     # median is robust to skew and outliers
df["qty_missing"] = df["qty"].isna().astype(int)                    # missing indicator (missingness is signal)
df["qty"] = df["qty"].fillna(median_qty)                            # fill with the median, documented
miss_after = df[["qty", "amount", "date"]].isna().mean().mul(100).round(2)   # missing rate after
record("Step 4 missing handling", len(df), len(df),                 # again no rows lost
       f"qty filled with median {median_qty:.1f}, indicator column added")   # log the decision
print("missing rate before:\n", miss_before.to_string())            # evidence for the slide
print("missing rate after:\n", miss_after.to_string())              # evidence for the slide

# ---------------------------------------------------------------- Step 5: validate and QA
assert df["order_id"].is_unique, "order_id must be unique after cleaning"   # uniqueness assertion
assert df["date"].notna().all(), "unparsable dates must be resolved"        # validity assertion
assert df["amount"].notna().all(), "amount must be numeric after cleaning"  # type assertion
assert set(df["payment"].unique()) <= {"paid", "unpaid", "not_applicable", "unknown"}, "unexpected payment"   # domain
assert df["city"].nunique() <= 5, "city normalisation failed"       # consistency assertion
clean = df[["order_id", "date", "city", "amount", "qty", "qty_missing", "payment",   # canonical column order
            "flag_impossible", "flag_outlier"]].copy()              # final analysis-ready column set
dump(clean, "l3_orders_clean.csv")                                  # publish the cleaned table
dump(pd.DataFrame(log), "l3_cleaning_log.csv")                      # publish the audit log

# ---------------------------------------------------------------- sample data for the structural-error slide
city_variants = (raw["city_raw"].value_counts().rename_axis("city_raw")   # count every raw spelling
                 .reset_index(name="rows"))                         # tidy for CSV output
dump(city_variants, "l3_city_names_raw.csv")                        # publish the raw variants
mapping = pd.DataFrame({"raw": list(city_map), "canonical": list(city_map.values())})   # alias table
dump(mapping, "l3_city_names_map.csv")                              # publish the mapping table

# ---------------------------------------------------------------- figure A: before / after
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # three before/after panels
steps = [r["step"].split()[1] for r in log[:3]] + ["Step 4"]        # short labels for the first four steps
axes[0].bar([r["step"].split()[1] for r in log[:3]],                # rows removed by each early step
            [r["rows_removed"] for r in log[:3]], color="#8A0C3C", alpha=0.85)   # bar chart of removals
axes[0].set_title("Step 1: rows removed")                           # panel title
axes[0].tick_params(axis="x", rotation=20)                          # keep labels readable
axes[0].set_ylabel("rows")                                          # y label
x = np.arange(len(miss_before))                                     # positions for the grouped bars
axes[1].bar(x - 0.2, miss_before.values, width=0.4, label="before", color="#C9A227")   # missing before
axes[1].bar(x + 0.2, miss_after.values, width=0.4, label="after", color="#8A0C3C")     # missing after
axes[1].set_xticks(x, miss_before.index)                            # column names on the x axis
axes[1].set_title("Step 4: missing rate (%)")                       # panel title
axes[1].legend()                                                    # explain the two colours
positive = df["amount"] > 0                                         # log10 needs strictly positive values
vals = [np.log10(df.loc[df[k] & positive, "amount"]) for k in ["flag_outlier", "flag_impossible"]]   # groups
axes[2].hist(np.log10(df.loc[positive, "amount"]), bins=30,         # overall distribution on a log scale
             color="#66717D", alpha=0.6, label="all rows")          # grey background histogram
for v, lab, c in zip(vals, ["statistical", "impossible"], ["#C9A227", "#8A0C3C"]):   # overlay each group
    axes[2].hist(v, bins=15, color=c, alpha=0.85, label=lab)        # histogram per flagged group
axes[2].set_title("Step 3: amount distribution (log10, positive only)")   # panel title
axes[2].legend()                                                    # group legend
plt.tight_layout()                                                  # balance the panels
save(fig, "fig_cleaning_before_after.png")                          # figure for the cleaning slide

# ---------------------------------------------------------------- figure B: structural error repair
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # variants / mapping / result
v = city_variants.set_index("city_raw")["rows"].sort_values()       # raw spelling counts
axes[0].barh(v.index, v.values, color="#8A0C3C", alpha=0.85)        # how messy the raw column is
axes[0].set_title(f"raw spellings ({len(v)} variants)")             # panel title
axes[0].tick_params(axis="y", labelsize=8)                          # smaller labels for long names
axes[1].axis("off")                                                 # the mapping is a table, not a plot
tbl = axes[1].table(cellText=mapping.values, colLabels=["raw", "canonical"],   # alias lookup table
                    loc="center", cellLoc="center")                 # centred cells
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.3)   # compact but readable
axes[1].set_title("alias mapping applied")                          # panel title
final_counts = df["city"].value_counts()                            # canonical counts after cleaning
axes[2].bar(final_counts.index, final_counts.values, color="#C9A227", alpha=0.9)   # tidy category column
axes[2].set_title(f"after cleaning ({df['city'].nunique()} cities)")   # panel title
axes[2].tick_params(axis="x", rotation=20)                          # keep labels readable
plt.tight_layout()                                                  # balance the panels
save(fig, "fig_structural_errors.png")                              # figure for the structural-error slide
print("clean shape:", clean.shape, "| audit log rows:", len(log))   # final summary in the console
