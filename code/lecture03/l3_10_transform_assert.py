"""l3_10_transform_assert.py -- transformations that tame skew, plus a schema assertion report.

Income-like variables are right skewed, so models and tests see a few huge values.
The script compares common transformations, measures what each one does to skew and
kurtosis, and finishes with a lightweight, pandera-style schema check that a
production pipeline would run before publishing the table.
"""
import numpy as np                                                 # logarithms and summary statistics
import pandas as pd                                                # tables and quantile binning
import matplotlib.pyplot as plt                                    # distribution comparison figure
from _common_l3 import RNG, save, dump                             # shared seed, figure and data writers

# ---------------------------------------------------------------- 1) a right-skewed variable
N = 900                                                            # number of firms
revenue = RNG.lognormal(mean=4.6, sigma=0.9, size=N)                # heavily right-skewed revenue
region = RNG.choice(["East", "Central", "West"], N, p=[0.5, 0.3, 0.2])   # a categorical column
raw = pd.DataFrame({"firm_id": [f"F{i:04d}" for i in range(N)],     # firm key
                    "region": region,                              # category column
                    "revenue": np.round(revenue, 2)})              # skewed numeric column
dump(raw, "l3_income_skewed.csv")                                  # publish the raw sample

# ---------------------------------------------------------------- 2) five transformations
transformed = raw.copy()                                           # work on a copy so the raw table survives
transformed["log1p"] = np.log1p(transformed["revenue"])             # log(1 + x): handles zeros and compresses tails
transformed["sqrt"] = np.sqrt(transformed["revenue"])               # square root: weaker than the logarithm
transformed["zscore"] = ((transformed["revenue"] - transformed["revenue"].mean())   # centre on the mean
                         / transformed["revenue"].std(ddof=0))      # standardisation: mean 0, sd 1
transformed["minmax"] = ((transformed["revenue"] - transformed["revenue"].min())    # shift to start at zero
                         / (transformed["revenue"].max() - transformed["revenue"].min()))   # scale to [0, 1]
transformed["quartile"] = pd.qcut(transformed["revenue"], 4, labels=["Q1", "Q2", "Q3", "Q4"])   # equal counts
transformed["decile"] = pd.qcut(transformed["revenue"], 10, labels=False)   # integer decile codes
summary = pd.DataFrame([                                          # compare shape before and after
    {"variable": name, "mean": round(transformed[name].mean(), 3),   # location
     "sd": round(transformed[name].std(), 3),                      # spread
     "skew": round(transformed[name].skew(), 3),                   # skewness: 0 is symmetric
     "kurtosis": round(transformed[name].kurt(), 3)}               # excess kurtosis: 0 is normal-like
    for name in ["revenue", "log1p", "sqrt", "zscore", "minmax"]])  # the five comparable columns
print(summary.to_string(index=False))                               # show the effect of each transformation
dump(summary, "l3_transform_summary.csv")                           # publish the comparison table
print("quartile counts:", transformed["quartile"].value_counts().sort_index().to_dict())   # equal-count bins
print("decile means (revenue):\n",                               # deciles are the basis of many business reports
      transformed.groupby("decile")["revenue"].mean().round(1).to_string())   # mean revenue per decile

# ---------------------------------------------------------------- 3) schema assertions before publishing
RULES = {                                                          # a minimal, pandera-style schema contract
    "firm_id": {"dtype": "object", "unique": True, "regex": r"^F\d{4}$"},   # key: unique and patterned
    "region": {"dtype": "object", "allowed": ["East", "Central", "West"]},   # category: closed domain
    "revenue": {"dtype": "float64", "min": 0.0, "max": 1_000_000.0, "nullable": False},   # numeric range
    "log1p": {"dtype": "float64", "min": 0.0, "nullable": False},   # derived column stays non-negative
}                                                                  # end of the schema contract
def check(table, rules):                                           # run the contract and collect violations
    """Return a tidy report of schema violations instead of raising on the first one."""   # helper contract
    report = []                                                    # one row per checked rule
    for column, spec in rules.items():                             # iterate over the declared columns
        present = column in table.columns                          # a missing column is itself a violation
        report.append({"column": column, "rule": "present",        # record the presence check
                       "ok": present, "detail": "" if present else "missing column"})   # detail for failures
        if not present:                                            # skip the remaining rules for that column
            continue                                               # move to the next column
        series = table[column]                                     # the column under test
        if spec.get("unique"):                                     # uniqueness rule
            dup = int(series.duplicated().sum())                   # number of duplicated values
            report.append({"column": column, "rule": "unique", "ok": dup == 0,   # uniqueness rule result
                           "detail": f"{dup} duplicates"})         # report the duplicate count
        if spec.get("regex"):                                      # pattern rule for identifiers
            bad = int((~series.astype(str).str.match(spec["regex"])).sum())   # non-conforming values
            report.append({"column": column, "rule": "regex", "ok": bad == 0,   # identifier pattern result
                           "detail": f"{bad} non-conforming"})     # report the count
        if spec.get("allowed"):                                    # closed domain rule
            extra = sorted(set(series.dropna()) - set(spec["allowed"]))   # unexpected categories
            report.append({"column": column, "rule": "allowed", "ok": not extra,   # closed domain result
                           "detail": f"unexpected: {extra}"})      # report the offenders
        if spec.get("min") is not None:                            # lower bound rule
            below = int((series < spec["min"]).sum())              # values below the bound
            report.append({"column": column, "rule": "min", "ok": below == 0,   # lower bound result
                           "detail": f"{below} below {spec['min']}"})   # report the count
        if spec.get("max") is not None:                            # upper bound rule
            above = int((series > spec["max"]).sum())              # values above the bound
            report.append({"column": column, "rule": "max", "ok": above == 0,   # upper bound result
                           "detail": f"{above} above {spec['max']}"})   # report the count
        if spec.get("nullable") is False:                          # nullability rule
            miss = int(series.isna().sum())                        # missing values in the column
            report.append({"column": column, "rule": "not-null", "ok": miss == 0,   # nullability result
                           "detail": f"{miss} missing"})           # report the count
    return pd.DataFrame(report)                                    # tidy report for the slide
report = check(transformed, RULES)                                 # validate the transformed table
print("schema checks passed:", int(report["ok"].sum()), "of", len(report))   # headline number
print(report[~report["ok"]].to_string(index=False))                # show only the failures, if any
dump(report, "l3_schema_report.csv")                               # publish the validation report
assert report["ok"].all(), "schema contract failed: fix the table before publishing it"   # fail the pipeline

# ---------------------------------------------------------------- figure: shape before and after
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # raw / transformed / shape metrics
axes[0].hist(transformed["revenue"], bins=40, color="#8A0C3C", alpha=0.85)   # raw, strongly skewed
axes[0].set_title(f"raw revenue (skew = {transformed['revenue'].skew():.2f})")   # skew in the title
axes[0].set_xlabel("revenue")                                       # axis label
axes[1].hist(transformed["log1p"], bins=40, color="#C9A227", alpha=0.9)   # after log1p
axes[1].set_title(f"log1p (skew = {transformed['log1p'].skew():.2f})")   # skew in the title
axes[1].set_xlabel("log(1 + revenue)")                              # axis label
metrics = summary.set_index("variable")[["skew", "kurtosis"]]       # shape metrics per transformation
x = np.arange(len(metrics))                                         # bar positions
axes[2].bar(x - 0.2, metrics["skew"], width=0.4, label="skew", color="#8A0C3C")   # skewness comparison
axes[2].bar(x + 0.2, metrics["kurtosis"], width=0.4, label="kurtosis", color="#66717D")   # kurtosis comparison
axes[2].axhline(0, color="#1E2630", lw=1.0)                         # zero line: symmetric, normal-like
axes[2].set_xticks(x, metrics.index, rotation=20, fontsize=8.5)     # transformation names
axes[2].set_title("shape metrics after transformation")             # panel title
axes[2].legend(fontsize=8.5)                                        # compact legend
plt.tight_layout()                                                  # balance the panels
save(fig, "fig_transform_scale.png")                                # figure for the transformation slide
