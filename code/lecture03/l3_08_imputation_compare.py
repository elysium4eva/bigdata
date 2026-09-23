"""l3_08_imputation_compare.py -- five imputation strategies judged against the known truth.

Income is deleted under a MAR pattern, then repaired with mean, median, regression,
KNN and MICE (IterativeImputer) imputation. Because the complete data is known, the
script can score every strategy on distribution, correlation and downstream
regression coefficient -- the practical test of whether imputation did harm.
"""
import numpy as np                                                 # linear algebra and summaries
import pandas as pd                                                # tables and missing-value handlers
import matplotlib.pyplot as plt                                    # distribution comparison figure
from sklearn.experimental import enable_iterative_imputer           # opt in to the experimental MICE imputer
from sklearn.impute import KNNImputer, IterativeImputer            # KNN and MICE style imputation
from sklearn.linear_model import LinearRegression                  # regression imputation and downstream model
from _common_l3 import RNG, save, dump                             # shared seed, figure and data writers

# ---------------------------------------------------------------- 1) complete data, then damage it
N = 700                                                            # surveyed employees
age = RNG.integers(23, 58, N)                                      # age in years
hours = np.round(RNG.normal(42, 6, N).clip(20, 65), 1)             # weekly working hours
edu = RNG.choice([0, 1, 2], N, p=[0.4, 0.42, 0.18])                # education level as a small integer
income = (12 + 0.30 * hours + 0.55 * edu + 0.06 * age              # the true data-generating process
          + RNG.normal(0, 2.4, N))                                 # plus idiosyncratic noise
complete = pd.DataFrame({"age": age, "hours": hours, "edu": edu,   # covariates
                         "income": np.round(income, 2)})           # the variable that will go missing
dump(complete, "l3_income_complete.csv")                           # publish the complete reference table
truth_mean, truth_sd = complete["income"].mean(), complete["income"].std()   # reference moments
truth_corr = complete["income"].corr(complete["hours"])            # reference correlation with hours
coef_truth = LinearRegression().fit(complete[["hours"]], complete["income"]).coef_[0]   # reference slope
print(f"truth: mean={truth_mean:.2f} sd={truth_sd:.2f} corr(hours)={truth_corr:.3f} slope={coef_truth:.3f}")   # baseline

# ---------------------------------------------------------------- 2) MAR deletion of income
p_missing = np.where(edu == 2, 0.55, np.where(edu == 1, 0.30, 0.10))   # higher education, less disclosure
mask = RNG.random(N) < p_missing                                   # the deletion mask
damaged = complete.copy()                                          # work on a copy
damaged.loc[mask, "income"] = np.nan                               # income is now missing for 28% of rows
dump(damaged, "l3_income_missing.csv")                             # publish the damaged table
print("missing rate in income:", round(100 * damaged['income'].isna().mean(), 1), "%")   # report the rate

# ---------------------------------------------------------------- 3) five repair strategies
filled = {"mean": damaged["income"].fillna(damaged["income"].mean()),   # global mean imputation
          "median": damaged["income"].fillna(damaged["income"].median()),   # global median imputation
          "drop": None}                                            # complete-case analysis (no repair)
features = ["hours", "edu", "age"]                                 # observed predictors for the models
known = damaged.dropna(subset=["income"])                          # rows used to fit the models
reg = LinearRegression().fit(known[features], known["income"])      # regression imputation model
reg_pred = reg.predict(damaged.loc[damaged["income"].isna(), features])   # predictions for missing rows
reg_series = damaged["income"].copy()                              # copy for regression-based filling
reg_series.loc[damaged["income"].isna()] = reg_pred                # fill with model predictions
filled["regression"] = reg_series                                  # register the strategy
knn = KNNImputer(n_neighbors=5).fit_transform(damaged[features + ["income"]])   # KNN imputation
filled["knn"] = pd.Series(knn[:, -1], index=damaged.index)         # keep the income column
mice = IterativeImputer(max_iter=10, random_state=42).fit_transform(damaged[features + ["income"]])   # MICE
filled["mice"] = pd.Series(mice[:, -1], index=damaged.index)       # MICE style iterative imputation
complete_case = damaged.dropna(subset=["income"])                  # option: analyse only complete rows

# ---------------------------------------------------------------- 4) score every strategy
rows = []                                                          # one row per strategy
for name, series in filled.items():                                # evaluate each repair
    data = complete_case if series is None else damaged.assign(income=series)   # pick the working table
    mean = data["income"].mean()                                   # estimated mean
    sd = data["income"].std()                                      # estimated spread
    corr = data["income"].corr(data["hours"])                      # correlation with hours
    slope = LinearRegression().fit(data[["hours"]], data["income"]).coef_[0]   # downstream slope
    rows.append({"strategy": name, "n_used": int(data["income"].notna().sum()),   # rows actually analysed
                 "mean": round(mean, 2), "mean_bias": round(mean - truth_mean, 2),   # bias in the mean
                 "sd": round(sd, 2), "sd_bias": round(sd - truth_sd, 2),   # bias in the spread
                 "corr_hours": round(corr, 3), "corr_error": round(corr - truth_corr, 3),   # correlation
                 "slope": round(slope, 3), "slope_error": round(slope - coef_truth, 3)})   # slope
scores = pd.DataFrame(rows)                                        # comparison table
print(scores.to_string(index=False))                               # show it in the console
dump(scores, "l3_imputation_metrics.csv")                          # publish the comparison table
strategy_means = pd.DataFrame({"strategy": ["complete_data"] + list(filled),   # truth plus every strategy
                               "mean_income": [truth_mean] + [mean for mean in scores["mean"]]})   # means
dump(strategy_means, "l3_imputation_means.csv")                    # publish a small plotting table

# ---------------------------------------------------------------- figure: distributions and errors
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                # density / box / error panels
bins = np.linspace(complete["income"].min(), complete["income"].max(), 30)   # shared bins for comparability
axes[0].hist(complete["income"], bins=bins, density=True, color="#1E2630",   # the truth as a reference
             alpha=0.35, label="complete (truth)")                 # legend label
for name, color in zip(["mean", "regression", "mice"], ["#66717D", "#C9A227", "#8A0C3C"]):
    axes[0].hist(filled[name], bins=bins, density=True, histtype="step",   # compare three repairs
                 lw=1.6, color=color, label=name)                  # curve per strategy
axes[0].set_title("income distribution after imputation")           # panel title
axes[0].set_xlabel("income (10k yuan)")                             # axis label
axes[0].legend(fontsize=8.5)                                        # compact legend
box_data = [complete["income"]] + [filled[k] for k in ["mean", "median", "regression", "knn", "mice"]]   # series
axes[1].boxplot(box_data, patch_artist=True,                        # spread comparison across strategies
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))     # filled boxes
axes[1].set_xticklabels(["truth", "mean", "median", "regr", "knn", "mice"], fontsize=8.5)   # labels
axes[1].set_title("spread is the first casualty")                   # panel title
x = np.arange(len(scores))                                          # positions for the error bars
axes[2].bar(x - 0.2, scores["mean_bias"], width=0.4, label="mean bias", color="#8A0C3C")   # level error
axes[2].bar(x + 0.2, scores["sd_bias"], width=0.4, label="spread bias", color="#C9A227")   # variance error
axes[2].axhline(0, color="#1E2630", lw=1.0)                         # zero line = no error
axes[2].set_xticks(x, scores["strategy"], rotation=20, fontsize=8.5)   # strategy names
axes[2].set_title("bias vs the complete-data truth")                # panel title
axes[2].set_ylabel("income units")                                  # same unit for both bars
axes[2].legend(fontsize=8.5)                                        # compact legend
plt.tight_layout()                                                  # balance the panels
save(fig, "fig_imputation_compare.png")                             # figure for the imputation slide
