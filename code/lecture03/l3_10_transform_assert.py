"""l3_10_transform_assert.py —— tame right-skewed distributions with transforms, plus a schema assertion report.

Income-like variables are usually right skewed, and a few extreme values dominate models and tests. The script compares common transforms,
quantifies their effect on skewness and kurtosis, and finishes with a light (pandera-style) schema check
that gates the data before publication -- exactly what a production pipeline does.
"""
import numpy as np                                                 # logs and descriptive statistics
import pandas as pd                                                # tables and quantile binning
import matplotlib.pyplot as plt                                    # distribution comparison figure
from _common_l3 import RNG, save, dump                             # reuse shared seed, plot and dump helpers

# ---------------------------------------------------------------- 1) a right-skewed variable
N = 900                                                            # number of firms
revenue = RNG.lognormal(mean=4.6, sigma=0.9, size=N)                # strongly right-skewed revenue
region = RNG.choice(["East", "Central", "West"], N, p=[0.5, 0.3, 0.2])   # a categorical column
raw = pd.DataFrame({"firm_id": [f"F{i:04d}" for i in range(N)],     # firm primary key
                    "region": region,                              # region (categorical)
                    "revenue": np.round(revenue, 2)})              # revenue (right skewed)
dump(raw, "l3_income_skewed.csv")                                  # save the raw sample

# ---------------------------------------------------------------- 2) five transforms
transformed = raw.copy()                                           # transform a copy, keep the original
transformed["log1p"] = np.log1p(transformed["revenue"])             # log(1+x): handles zeros, shrinks tail
transformed["sqrt"] = np.sqrt(transformed["revenue"])               # square root: gentler than the log
transformed["zscore"] = ((transformed["revenue"] - transformed["revenue"].mean())   # centre first
                         / transformed["revenue"].std(ddof=0))      # then scale: mean 0, sd 1
transformed["minmax"] = ((transformed["revenue"] - transformed["revenue"].min())    # shift so it starts at 0
                         / (transformed["revenue"].max() - transformed["revenue"].min()))   # scale to [0,1]
transformed["quartile"] = pd.qcut(transformed["revenue"], 4, labels=["Q1", "Q2", "Q3", "Q4"])   # equal-frequency binning
transformed["decile"] = pd.qcut(transformed["revenue"], 10, labels=False)   # deciles as integer codes
summary = pd.DataFrame([                                          # shape comparison before and after
    {"variable": name, "mean": round(transformed[name].mean(), 3),   # location
     "sd": round(transformed[name].std(), 3),                      # spread
     "skew": round(transformed[name].skew(), 3),                   # skewness: 0 means symmetric
     "kurtosis": round(transformed[name].kurt(), 3)}               # excess kurtosis: 0 is normal-like
    for name in ["revenue", "log1p", "sqrt", "zscore", "minmax"]])  # the five comparable columns
print(summary.to_string(index=False))                               # print the effect of each transform
dump(summary, "l3_transform_summary.csv")                           # save the comparison table
print("quartile counts:", transformed["quartile"].value_counts().sort_index().to_dict())   # equal-frequency bin counts
print("decile means (revenue):\n",                               # deciles are common in reporting
      transformed.groupby("decile")["revenue"].mean().round(1).to_string())   # mean revenue per decile

# ---------------------------------------------------------------- 3) schema assertions before publishing
RULES = {                                                          # a minimal schema contract
    "firm_id": {"dtype": "object", "unique": True, "regex": r"^F\d{4}$"},   # key: unique with a format
    "region": {"dtype": "object", "allowed": ["East", "Central", "West"]},   # category: closed value domain
    "revenue": {"dtype": "float64", "min": 0.0, "max": 1_000_000.0, "nullable": False},   # numeric range
    "log1p": {"dtype": "float64", "min": 0.0, "nullable": False},   # derived column is non-negative
}                                                                  # end of the contract definition
def check(table, rules):                                           # run each rule and collect violations
    """Return a violation report instead of stopping at the first problem."""   # function contract
    report = []                                                    # one row per rule
    for column, spec in rules.items():                             # walk the declared columns
        present = column in table.columns                          # a missing column is itself a violation
        report.append({"column": column, "rule": "present",        # record the presence check
                       "ok": present, "detail": "" if present else "missing column"})   # state the reason on failure
        if not present:                                            # skip the rest if the column is absent
            continue                                               # move to the next column
        series = table[column]                                     # the column to check
        if spec.get("unique"):                                     # uniqueness rule
            dup = int(series.duplicated().sum())                   # number of duplicated values
            report.append({"column": column, "rule": "unique", "ok": dup == 0,   # uniqueness check result
                           "detail": f"{dup} duplicates"})         # state the duplicate count
        if spec.get("regex"):                                      # identifier format rule
            bad = int((~series.astype(str).str.match(spec["regex"])).sum())   # non-conforming values
            report.append({"column": column, "rule": "regex", "ok": bad == 0,   # format check result
                           "detail": f"{bad} non-conforming"})     # state the non-conforming count
        if spec.get("allowed"):                                    # allowed values rule
            extra = sorted(set(series.dropna()) - set(spec["allowed"]))   # unexpected categories seen
            report.append({"column": column, "rule": "allowed", "ok": not extra,   # value domain check result
                           "detail": f"unexpected: {extra}"})      # state the out-of-range values
        if spec.get("min") is not None:                            # lower bound rule
            below = int((series < spec["min"]).sum())              # values below the bound
            report.append({"column": column, "rule": "min", "ok": below == 0,   # lower bound check result
                           "detail": f"{below} below {spec['min']}"})   # state the out-of-range count
        if spec.get("max") is not None:                            # upper bound rule
            above = int((series > spec["max"]).sum())              # values above the bound
            report.append({"column": column, "rule": "max", "ok": above == 0,   # upper bound check result
                           "detail": f"{above} above {spec['max']}"})   # state the out-of-range count
        if spec.get("nullable") is False:                          # not-null rule
            miss = int(series.isna().sum())                        # number of missing values
            report.append({"column": column, "rule": "not-null", "ok": miss == 0,   # not-null check result
                           "detail": f"{miss} missing"})           # state the missing count
    return pd.DataFrame(report)                                    # return the detail table for the page
report = check(transformed, RULES)                                 # validate the transformed table
print("schema checks passed:", int(report["ok"].sum()), "of", len(report))   # print the number passed
print(report[~report["ok"]].to_string(index=False))                # print only the failures
dump(report, "l3_schema_report.csv")                               # save the validation report
assert report["ok"].all(), "schema contract failed: fix the table before publishing it"   # stop the pipeline on failure

# ---------------------------------------------------------------- figure: before and after transform
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # raw / transformed / shape metrics
axes[0].hist(transformed["revenue"], bins=40, color="#8A0C3C", alpha=0.85)   # raw distribution: strongly skewed
axes[0].set_title(f"raw revenue (skew = {transformed['revenue'].skew():.2f})")   # skewness in the title
axes[0].set_xlabel("revenue")                                       # what the x axis means
axes[1].hist(transformed["log1p"], bins=40, color="#C9A227", alpha=0.9)   # after log1p
axes[1].set_title(f"log1p (skew = {transformed['log1p'].skew():.2f})")   # skewness in the title
axes[1].set_xlabel("log(1 + revenue)")                              # what the x axis means
metrics = summary.set_index("variable")[["skew", "kurtosis"]]       # shape metrics per transform
x = np.arange(len(metrics))                                         # x positions of the bars
axes[2].bar(x - 0.2, metrics["skew"], width=0.4, label="skew", color="#8A0C3C")   # skewness comparison
axes[2].bar(x + 0.2, metrics["kurtosis"], width=0.4, label="kurtosis", color="#66717D")   # kurtosis comparison
axes[2].axhline(0, color="#1E2630", lw=1.0)                         # zero line: symmetric, near normal
axes[2].set_xticks(x, metrics.index, rotation=20, fontsize=8.5)     # transform names
axes[2].set_title("shape metrics after transformation")             # subplot title
axes[2].legend(fontsize=8.5)                                        # legend
plt.tight_layout()                                                  # adjust the three subplots
save(fig, "fig_transform_scale.png")                                # save: transform slide figure
