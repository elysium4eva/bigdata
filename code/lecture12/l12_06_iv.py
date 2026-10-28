"""l12_06_iv.py -- two-stage least squares, the first stage and weak instruments.

Every line carries an English comment. The script shows the omitted-variable bias of OLS on an   # continuation of the call
endogenous regressor, estimates 2SLS by hand, reports the first-stage F statistic, and simulates   # continuation of the call
how the 2SLS bias grows as the instrument weakens.                      # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.api as sm                                       # OLS machinery
from _common_l12 import (make_iv, RNG, save, dump, style_axes,      # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the sample
df = make_iv(3000)                                                 # 3,000 observations
dump(df, "l12_iv.csv")                                             # persist
print("n:", len(df), "| corr(z, x) %.3f | corr(x, y) %.3f" % (          # report the result
    np.corrcoef(df["z"], df["x"])[0, 1], np.corrcoef(df["x"], df["y"])[0, 1]))   # report

# ---------------------------------------------------------------- 2) OLS versus 2SLS
ols = sm.OLS(df["y"], sm.add_constant(df[["x"]])).fit()              # the biased OLS estimate
b_ols = float(ols.params["x"])                                      # its slope
# first stage: x on z
first = sm.OLS(df["x"], sm.add_constant(df[["z"]])).fit()             # the first stage
x_hat = first.fittedvalues                                          # the fitted values
second = sm.OLS(df["y"], sm.add_constant(x_hat)).fit()               # the second stage (manual 2SLS)
b_2sls = float(second.params.iloc[1])                                # the 2SLS slope
f_stat = float(first.fvalue)                                        # the first-stage F statistic
print("OLS %.3f | 2SLS %.3f | first-stage F %.1f (true 1.5)" % (b_ols, b_2sls, f_stat))   # report
table = pd.DataFrame([{"estimator": "OLS", "slope": round(b_ols, 3)},   # OLS
                      {"estimator": "2SLS", "slope": round(b_2sls, 3)},   # 2SLS
                      {"estimator": "true", "slope": 1.5}])            # the truth
dump(table, "l12_iv_compare.csv")                                   # persist
dump(pd.DataFrame([{"first_stage_f": round(f_stat, 2), "first_stage_coef": round(float(first.params["z"]), 3)}]),   # assemble the frame
     "l12_iv_first_stage.csv")                                      # persist

# ---------------------------------------------------------------- 3) weak instruments
def iv_once(n, strength):                                           # one simulated dataset
    """Return the OLS and 2SLS slopes for a given instrument strength."""   # short docstring
    z = RNG.binomial(1, 0.5, n)                                     # the instrument
    u = RNG.normal(0, 1, n)                                        # the confounder
    x = strength * z + 0.9 * u + RNG.normal(0, 1, n)                # the endogenous regressor
    y = 1.5 * x + 2.0 * u + RNG.normal(0, 1, n)                     # the outcome
    b_ols = float(sm.OLS(y, sm.add_constant(x)).fit().params[1])     # OLS
    f1 = sm.OLS(x, sm.add_constant(z)).fit()                        # the first stage
    b_iv = float(sm.OLS(y, sm.add_constant(f1.fittedvalues)).fit().params[1])   # 2SLS
    return b_ols, b_iv, float(f1.fvalue)                             # the estimates and the F

rows = []                                                           # one row per strength
for strength in [0.05, 0.10, 0.20, 0.40, 0.80, 1.60]:                 # from very weak to strong
    ols_list, iv_list, f_list = [], [], []                           # accumulators
    for _ in range(60):                                            # sixty replications
        b_o, b_i, f = iv_once(400, strength)                         # one replication
        ols_list.append(b_o); iv_list.append(b_i); f_list.append(f)   # collect
    rows.append({"strength": strength, "mean_ols": round(float(np.mean(ols_list)), 3),   # append the record
                 "mean_2sls": round(float(np.mean(iv_list)), 3),        # continuation of the call
                 "median_f": round(float(np.median(f_list)), 1)})     # the row
weak = pd.DataFrame(rows)                                          # assemble
dump(weak, "l12_weak_iv.csv")                                      # persist
print(weak.to_string(index=False))                                 # report

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
axes[0].scatter(df["x"], df["y"], s=6, alpha=0.25, color=OKABE[1])    # the raw scatter
grid = np.linspace(df["x"].min(), df["x"].max(), 40)                 # a grid for the lines
axes[0].plot(grid, ols.params["const"] + b_ols * grid, lw=2.0, color=OKABE[1], label="OLS %.2f" % b_ols)   # OLS
axes[0].plot(grid, second.params.iloc[0] + b_2sls * grid, lw=2.0, color=OKABE[5], label="2SLS %.2f" % b_2sls)   # 2SLS
axes[0].plot(grid, 3.0 + 1.5 * grid, lw=1.4, ls="--", color=OKABE[7], label="真值 1.50")   # the truth
axes[0].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[0], "OLS 被内生性拉高，2SLS 接近真值", "x", "y")          # shared style
axes[1].plot(weak["median_f"], weak["mean_2sls"], "-o", ms=5, color=OKABE[5], label="2SLS")   # 2SLS vs F
axes[1].axhline(1.5, ls="--", lw=1.0, color="#66717D", label="真值")    # the truth
axes[1].set_xscale("log")                                          # F spans orders of magnitude
axes[1].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[1], "弱工具：F 越小，2SLS 越偏", "第一阶段 F（对数）", "2SLS 估计均值")   # shared style
annotated(axes[1], "F<10 通常视为弱工具", "lower right")                # annotate
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_iv_compare.png")                                     # save the figure
print("done: 2SLS, first stage and weak instruments")                 # closing log
