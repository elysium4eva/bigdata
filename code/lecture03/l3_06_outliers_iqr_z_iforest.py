"""l3_06_outliers_iqr_z_iforest.py -- three ways to flag outliers and three ways to treat them.

The sample mixes a log-normal core with three kinds of injected anomalies: a data
entry error, a legitimate extreme value and a small group of unusual records.
The script compares the IQR rule, the Z-score rule and an Isolation Forest, then
shows how dropping, winsorising or keeping changes the summary statistics.
"""
import numpy as np                                                 # random draws and quantile helpers
import pandas as pd                                                # tables and summary statistics
import matplotlib.pyplot as plt                                    # flag comparison figures
from sklearn.ensemble import IsolationForest                       # multivariate anomaly detection
from _common_l3 import RNG, save, dump                             # shared seed, figure and data writers

# ---------------------------------------------------------------- 1) sample with known anomalies
n_core, n_err, n_extreme, n_group = 400, 2, 1, 8                    # sizes of the four sub-populations
core = RNG.lognormal(mean=10.4, sigma=0.45, size=n_core)            # normal salaries in yuan
errors = np.array([-3200.0, 999999.0])                             # two data-entry errors
extreme = np.array([480000.0])                                     # one legitimate but extreme salary
group = RNG.normal(loc=250000, scale=4000, size=n_group)           # a small cluster of unusual records
values = np.concatenate([core, errors, extreme, group])             # one column of all salaries
kind = (["core"] * n_core + ["entry_error"] * n_err                 # record the true generating process
        + ["legitimate_extreme"] * n_extreme + ["unusual_group"] * n_group)   # only for teaching
df = pd.DataFrame({"salary": np.round(values, 2), "true_kind": kind})   # the analysis table
dump(df, "l3_income_outliers.csv")                                 # publish the sample

# ---------------------------------------------------------------- 2) three detection rules
q1, q3 = df["salary"].quantile([.25, .75])                          # quartiles of the observed column
iqr = q3 - q1                                                       # inter-quartile range
df["iqr_outlier"] = (df["salary"] < q1 - 1.5 * iqr) | (df["salary"] > q3 + 1.5 * iqr)   # Tukey fence
z = (df["salary"] - df["salary"].mean()) / df["salary"].std(ddof=0)   # classic Z-score
df["z_outlier"] = z.abs() > 3                                       # |z| > 3 rule
mad = (df["salary"] - df["salary"].median()).abs().median()         # median absolute deviation
robust_z = 0.6745 * (df["salary"] - df["salary"].median()) / mad    # robust Z (MAD based)
df["robust_z_outlier"] = robust_z.abs() > 3.5                       # threshold for the robust version
iso = IsolationForest(contamination=0.03, random_state=42)          # 3% expected anomalies
df["iforest_outlier"] = iso.fit_predict(df[["salary"]]) == -1       # -1 marks the anomalies
flags = ["iqr_outlier", "z_outlier", "robust_z_outlier", "iforest_outlier"]   # the four rule columns
print("flagged counts:", {c: int(df[c].sum()) for c in flags})      # how many rows each rule flags
print("rules agree on all flags:", int((df[flags].sum(axis=1) == len(flags)).sum()))   # unanimous rows
print("extreme value flagged by IQR/Z:", bool(df.loc[df["true_kind"] == "legitimate_extreme", "iqr_outlier"].iloc[0]))   # legit?
print("entry errors flagged by IQR:", int(df.loc[df["true_kind"] == "entry_error", "iqr_outlier"].sum()))   # count

# ---------------------------------------------------------------- 3) three treatments
keep = df["salary"]                                                 # option A: keep everything
drop = df.loc[~df["iqr_outlier"], "salary"]                         # option B: drop IQR outliers
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr                       # winsorising bounds
wins = df["salary"].clip(lower=lower, upper=upper)                  # option C: winsorise to the fences
summary = pd.DataFrame({"treatment": ["keep", "drop (IQR)", "winsorise (IQR)"],   # the three options
                        "n": [len(keep), len(drop), len(wins)],     # rows kept
                        "mean": [keep.mean(), drop.mean(), wins.mean()],   # mean after treatment
                        "median": [keep.median(), drop.median(), wins.median()],   # median is robust
                        "std": [keep.std(), drop.std(), wins.std()]}).round(1)   # spread after treatment
print(summary.to_string(index=False))                               # show the trade-off in the console
dump(summary, "l3_outlier_treatment.csv")                           # publish the comparison table

# ---------------------------------------------------------------- figure: flags and treatments
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # scatter / box / overlap panels
axes[0].scatter(np.arange(len(df)), df["salary"], s=10,             # index vs salary scatter
                c=np.where(df["iqr_outlier"], "#8A0C3C", "#B9BFC7"))   # red = flagged by IQR
axes[0].axhline(upper, color="#C9A227", lw=1.2, ls="--", label="IQR fence")   # upper fence line
axes[0].axhline(lower, color="#C9A227", lw=1.2, ls="--")            # lower fence line
axes[0].set_yscale("symlog")                                        # symlog keeps errors and extremes visible
axes[0].set_title("flagged points (IQR rule)")                      # panel title
axes[0].set_xlabel("row index"); axes[0].set_ylabel("salary (yuan)")   # axis labels
axes[0].legend()                                                    # fence legend
axes[1].boxplot([keep, drop, wins], patch_artist=True,              # compare the treatment options
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))     # filled boxes for readability
axes[1].set_xticklabels(["keep", "drop", "winsorise"])              # treatment labels
axes[1].set_title("distribution after treatment")                   # panel title
axes[2].bar([c.replace("_outlier", "") for c in flags],             # how many rows each rule flags
            [int(df[c].sum()) for c in flags], color="#66717D", alpha=0.9)   # counts per rule
axes[2].set_title("rows flagged per rule")                          # panel title
axes[2].tick_params(axis="x", rotation=20)                          # keep labels readable
for i, c in enumerate(flags):                                       # annotate the counts on the bars
    axes[2].text(i, int(df[c].sum()), str(int(df[c].sum())), ha="center", va="bottom", fontsize=9)   # label
plt.tight_layout()                                                  # balance the three panels
save(fig, "fig_outliers.png")                                       # figure for the outlier slide
