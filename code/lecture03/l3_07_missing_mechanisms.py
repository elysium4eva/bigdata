"""l3_07_missing_mechanisms.py -- manufacture MCAR, MAR and MNAR missingness and measure the bias.

The same complete population is damaged in three different ways. Because the truth
is known, the script can show how each mechanism distorts a simple estimate such
as the mean income: MCAR stays unbiased, MAR is repairable with the right model,
MNAR cannot be fixed without assumptions.
"""
import numpy as np                                                 # random draws and bias arithmetic
import pandas as pd                                                # tables and missing-value tooling
import matplotlib.pyplot as plt                                    # missing-matrix figure
from _common_l3 import RNG, save, dump                             # shared seed, figure and data writers

# ---------------------------------------------------------------- 1) one complete population
N = 800                                                            # number of surveyed households
education = RNG.choice(["high_school", "bachelor", "master"], N,   # education drives income
                       p=[0.45, 0.40, 0.15])                       # realistic education mix
age = RNG.integers(22, 60, N)                                      # working age in years
base = {"high_school": 8.6, "bachelor": 9.6, "master": 10.2}       # log-income intercept per education level
mu = np.array([base[e] for e in education]) + 0.012 * (age - 22)   # income grows with age
income = RNG.lognormal(mean=mu, sigma=0.30)                        # complete, known monthly incomes in yuan
complete = pd.DataFrame({"household_id": [f"H{i:04d}" for i in range(N)],   # household key
                         "education": education, "age": age,       # fully observed covariates
                         "income": np.round(income, 2)})           # the variable that will go missing
dump(complete, "l3_income_complete_l3_07.csv")                     # publish the complete sample
true_mean = complete["income"].mean()                              # the target estimand: mean income
print(f"true mean income (complete data): {true_mean:.2f} yuan/month")   # baseline for the comparison

# ---------------------------------------------------------------- 2) three missingness mechanisms
mcar_mask = RNG.random(N) < 0.30                                   # MCAR: every row has the same 30% chance
mar_p = np.where(education == "master", 0.55,                  # MAR: missingness depends on education,
                 np.where(education == "bachelor", 0.30, 0.12))    # a variable that is fully observed
mar_mask = RNG.random(N) < mar_p                                   # draw the MAR deletion pattern
mnar_p = np.clip((income - income.min()) / (income.max() - income.min()), 0, 1) ** 2   # MNAR: rich refuse
mnar_mask = RNG.random(N) < 0.15 + 0.55 * mnar_p                   # high earners drop out more often
mechanisms = {"MCAR": mcar_mask, "MAR": mar_mask, "MNAR": mnar_mask}   # the three damaged versions
rows, panels, matrices = [], {}, {}                                # containers for report, panel and figure
for name, mask in mechanisms.items():                              # build one damaged table per mechanism
    damaged = complete.copy()                                      # keep the complete rows untouched
    damaged.loc[mask, "income"] = np.nan                           # delete the income values only
    naive_mean = damaged["income"].mean()                          # estimate the mean on the observed rows
    rows.append({"mechanism": name, "missing_rate_pct": round(100 * mask.mean(), 1),   # how much is missing
                 "observed_mean": round(naive_mean, 2),            # naive estimate
                 "bias_vs_truth": round(naive_mean - true_mean, 2),   # bias of the naive estimate
                 "bias_pct": round(100 * (naive_mean - true_mean) / true_mean, 1)})   # relative bias
    panels[name] = damaged                                         # store for the CSV export
    matrices[name] = damaged[["income", "age", "education"]].isna().astype(int)   # 0/1 matrix for the figure
bias = pd.DataFrame(rows)                                          # one row per mechanism
print(bias.to_string(index=False))                                 # show the bias table in the console
dump(bias, "l3_missing_bias.csv")                                  # publish the bias table
for name, damaged in panels.items():                               # export each damaged version
    dump(damaged, f"l3_income_missing_{name}.csv")                  # e.g. l3_income_missing_MNAR.csv

# ---------------------------------------------------------------- 3) panel sample with block missingness
n_ids, n_periods = 6, 12                                            # a small firm panel
grid = pd.DataFrame([(f"F{i:02d}", t) for i in range(1, n_ids + 1)  # full (firm, month) grid
                     for t in range(1, n_periods + 1)],            # twelve months per firm
                    columns=["firm_id", "month"])                   # column names
grid["revenue"] = np.round(RNG.lognormal(6.0, 0.35, len(grid)), 1)  # revenue per firm-month
drop_firm = grid["firm_id"].isin(["F03", "F05"]) & grid["month"].isin([3, 4, 5])   # a whole block disappears
drop_tail = grid["firm_id"].eq("F06") & grid["month"].gt(9)         # one firm leaves the sample early
grid.loc[drop_firm | drop_tail, "revenue"] = np.nan                 # the panel is now unbalanced
dump(grid, "l3_panel_missing.csv")                                 # publish the panel with block missingness
coverage = (grid.assign(observed=grid["revenue"].notna())           # per-firm coverage statistics
            .groupby("firm_id")["observed"].agg(["sum", "count"]))  # observed months vs total months
coverage["coverage_pct"] = (100 * coverage["sum"] / coverage["count"]).round(1)   # coverage in percent
print("panel coverage:\n", coverage.to_string())                    # coverage table for the slide
dump(coverage.reset_index(), "l3_panel_coverage.csv")               # publish the coverage table

# ---------------------------------------------------------------- figure: missing matrices and bias
fig = plt.figure(figsize=(11.0, 4.2))                               # custom grid: matrices plus two charts
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.5])         # two matrices side by side plus a chart
for i, name in enumerate(["MCAR", "MAR"]):                          # draw the first two mechanisms
    ax = fig.add_subplot(gs[0, i])                                  # one axes per mechanism
    ax.imshow(matrices[name].T.values, cmap="Greys", aspect="auto", vmin=0, vmax=1)   # 1 = missing (dark)
    ax.set_yticks(range(3), ["income", "age", "education"], fontsize=9)   # variables on the y axis
    ax.set_xlabel("households")                                     # rows of the sample
    ax.set_title(f"{name}: missing pattern "                        # panel title with the missing rate
                 f"({100 * mechanisms[name].mean():.0f}% of income missing)", fontsize=10.5)   # rate in title
    ax.grid(False)                                                  # matrices need no grid
ax = fig.add_subplot(gs[0, 2])                                      # right panel: bias of the naive mean
colors = ["#66717D", "#C9A227", "#8A0C3C"]                          # grey, gold, red for MCAR/MAR/MNAR
ax.bar(bias["mechanism"], bias["bias_pct"], color=colors, alpha=0.9)   # relative bias per mechanism
ax.axhline(0, color="#1E2630", lw=1.0)                              # zero line = unbiased
ax.set_title("bias of the naive mean income (%)", fontsize=10.5)    # panel title
for i, v in enumerate(bias["bias_pct"]):                            # annotate each bar
    ax.text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)   # value label
plt.tight_layout()                                                  # balance the figure
save(fig, "fig_missing_matrix.png")                                 # figure for the missing-data slide
