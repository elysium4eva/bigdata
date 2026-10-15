"""l3_06_outliers_iqr_z_iforest.py —— three ways to find outliers, and three ways to treat them.

The sample mixes lognormal "normal salaries" with three kinds of anomaly: two entry errors, one real but extreme value,
and a small group of unusual records. The script compares the IQR rule, the Z-score rule and the isolation forest,
then shows what dropping, winsorising and keeping do to the mean, median and standard deviation.
"""
import numpy as np                                                 # random sampling and quantiles
import pandas as pd                                                # tables and descriptive statistics
import matplotlib.pyplot as plt                                    # figure comparing the flagged points
from sklearn.ensemble import IsolationForest                       # multivariate anomaly detection
from _common_l3 import RNG, save, dump                             # reuse shared seed, plot and dump helpers

# ---------------------------------------------------------------- 1) a sample with known composition
n_core, n_err, n_extreme, n_group = 400, 2, 1, 8                    # size of each subsample
core = RNG.lognormal(mean=10.4, sigma=0.45, size=n_core)            # normal salaries (yuan)
errors = np.array([-3200.0, 999999.0])                             # two entry errors
extreme = np.array([480000.0])                                     # one real but extreme salary
group = RNG.normal(loc=250000, scale=4000, size=n_group)           # a small unusual group
values = np.concatenate([core, errors, extreme, group])             # combine into one salary column
kind = (["core"] * n_core + ["entry_error"] * n_err                 # record the true generating process
        + ["legitimate_extreme"] * n_extreme + ["unusual_group"] * n_group)   # for teaching comparison only
df = pd.DataFrame({"salary": np.round(values, 2), "true_kind": kind})   # the analysis table
dump(df, "l3_income_outliers.csv")                                 # save the sample

# ---------------------------------------------------------------- 2) three detection rules
q1, q3 = df["salary"].quantile([.25, .75])                          # quartiles of the observed column
iqr = q3 - q1                                                       # interquartile range
df["iqr_outlier"] = (df["salary"] < q1 - 1.5 * iqr) | (df["salary"] > q3 + 1.5 * iqr)   # boxplot rule
z = (df["salary"] - df["salary"].mean()) / df["salary"].std(ddof=0)   # classic Z score
df["z_outlier"] = z.abs() > 3                                       # the |z| > 3 rule
mad = (df["salary"] - df["salary"].median()).abs().median()         # median absolute deviation
robust_z = 0.6745 * (df["salary"] - df["salary"].median()) / mad    # robust Z based on MAD
df["robust_z_outlier"] = robust_z.abs() > 3.5                       # threshold for the robust version
iso = IsolationForest(contamination=0.03, random_state=42)          # expect about 3% anomalies
df["iforest_outlier"] = iso.fit_predict(df[["salary"]]) == -1       # -1 means flagged as anomaly
flags = ["iqr_outlier", "z_outlier", "robust_z_outlier", "iforest_outlier"]   # the four rule columns
print("flagged counts:", {c: int(df[c].sum()) for c in flags})      # rows flagged per rule
print("rules agree on all flags:", int((df[flags].sum(axis=1) == len(flags)).sum()))   # rows all rules agree on
print("extreme value flagged by IQR/Z:", bool(df.loc[df["true_kind"] == "legitimate_extreme", "iqr_outlier"].iloc[0]))   # the real extreme value
print("entry errors flagged by IQR:", int(df.loc[df["true_kind"] == "entry_error", "iqr_outlier"].sum()))   # entry error count

# ---------------------------------------------------------------- 3) three treatments
keep = df["salary"]                                                 # option A: keep everything
drop = df.loc[~df["iqr_outlier"], "salary"]                         # option B: drop IQR outliers
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr                       # winsorising bounds
wins = df["salary"].clip(lower=lower, upper=upper)                  # option C: clip to the bounds
summary = pd.DataFrame({"treatment": ["keep", "drop (IQR)", "winsorise (IQR)"],   # the three treatments
                        "n": [len(keep), len(drop), len(wins)],     # sample size kept
                        "mean": [keep.mean(), drop.mean(), wins.mean()],   # mean
                        "median": [keep.median(), drop.median(), wins.median()],   # median (robust)
                        "std": [keep.std(), drop.std(), wins.std()]}).round(1)   # standard deviation
print(summary.to_string(index=False))                               # print the treatment comparison
dump(summary, "l3_outlier_treatment.csv")                           # save the comparison table

# ---------------------------------------------------------------- figure: flags and treatments
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # scatter / boxplot / rule counts
axes[0].scatter(np.arange(len(df)), df["salary"], s=10,             # row index against salary
                c=np.where(df["iqr_outlier"], "#8A0C3C", "#B9BFC7"))   # red marks the IQR-flagged rows
axes[0].axhline(upper, color="#C9A227", lw=1.2, ls="--", label="IQR fence")   # upper bound
axes[0].axhline(lower, color="#C9A227", lw=1.2, ls="--")            # lower bound
axes[0].set_yscale("symlog")                                        # symlog shows errors and extremes
axes[0].set_title("flagged points (IQR rule)")                      # subplot title
axes[0].set_xlabel("row index"); axes[0].set_ylabel("salary (yuan)")   # axis meanings
axes[0].legend()                                                    # legend: the fence lines
axes[1].boxplot([keep, drop, wins], patch_artist=True,              # compare the three treatments
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))     # filled boxes are easier to read
axes[1].set_xticklabels(["keep", "drop", "winsorise"])              # x tick labels
axes[1].set_title("distribution after treatment")                   # subplot title
axes[2].bar([c.replace("_outlier", "") for c in flags],             # the rule names
            [int(df[c].sum()) for c in flags], color="#66717D", alpha=0.9)   # rows flagged per rule
axes[2].set_title("rows flagged per rule")                          # subplot title
axes[2].tick_params(axis="x", rotation=20)                          # rotate labels
for i, c in enumerate(flags):                                       # annotate the value on top
    axes[2].text(i, int(df[c].sum()), str(int(df[c].sum())), ha="center", va="bottom", fontsize=9)   # value label
plt.tight_layout()                                                  # adjust the three subplots
save(fig, "fig_outliers.png")                                       # save: outliers slide figure
