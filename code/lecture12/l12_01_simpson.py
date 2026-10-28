"""l12_01_simpson.py -- Simpson's paradox: the pooled slope and the within-group slopes.

Every line carries an English comment. The script builds a dataset where the pooled relationship   # continuation of the call
and the within-group relationship have opposite signs, draws the pooled scatter with its slope   # continuation of the call
next to the two group-specific fits, and prints the two regressions so the deck can show that   # continuation of the call
"conditioning" is the whole story.                                      # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # OLS formulas
from _common_l12 import (make_simpson, save, dump, style_axes,     # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the data
df = make_simpson(600)                                             # 600 units in two groups
dump(df, "l12_simpson.csv")                                        # persist
print("units:", len(df), "| groups:", df["group"].value_counts().to_dict())   # report

# ---------------------------------------------------------------- 2) the two regressions
pooled = smf.ols("y ~ x", data=df).fit()                            # the pooled regression
within = smf.ols("y ~ x + C(group)", data=df).fit()                  # with a group fixed effect
by_group = {g: smf.ols("y ~ x", data=sub).fit() for g, sub in df.groupby("group")}   # one per group
b_pool = float(pooled.params["x"])                                  # the pooled slope
b_within = float(within.params["x"])                                # the within-group slope
print("pooled slope %.3f | within slope %.3f" % (b_pool, b_within))   # report
print("group A slope %.3f | group B slope %.3f" % (                     # report the result
    float(by_group["A"].params["x"]), float(by_group["B"].params["x"])))   # report
table = pd.DataFrame([{"spec": "合并 pooled", "slope": round(b_pool, 3), "r2": round(float(pooled.rsquared), 3)},   # assemble the frame
                      {"spec": "分组 A", "slope": round(float(by_group["A"].params["x"]), 3),   # continuation of the call
                       "r2": round(float(by_group["A"].rsquared), 3)},   # continuation of the call
                      {"spec": "分组 B", "slope": round(float(by_group["B"].params["x"]), 3),   # continuation of the call
                       "r2": round(float(by_group["B"].rsquared), 3)},   # continuation of the call
                      {"spec": "固定效应 FE", "slope": round(b_within, 3),   # continuation of the call
                       "r2": round(float(within.rsquared), 3)}])        # assemble
dump(table, "l12_simpson_slopes.csv")                              # persist
print(table.to_string(index=False))                                # report

# ---------------------------------------------------------------- 3) the figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
for g, colour in (("A", OKABE[1]), ("B", OKABE[5])):                 # loop over the groups
    sub = df[df["group"] == g]                                     # the group's rows
    axes[0].scatter(sub["x"], sub["y"], s=9, alpha=0.45, color=colour, label="组 %s" % g)   # the points
    grid = np.linspace(sub["x"].min(), sub["x"].max(), 40)           # a grid for the fit
    fit = by_group[g]                                              # the group regression
    axes[0].plot(grid, fit.params["Intercept"] + fit.params["x"] * grid, lw=1.8, color=colour)   # the fit
grid_all = np.linspace(df["x"].min(), df["x"].max(), 40)            # a grid for the pooled fit
axes[0].plot(grid_all, pooled.params["Intercept"] + pooled.params["x"] * grid_all,   # draw the panel
             lw=2.4, ls="--", color="#2B2B2B", label="合并（正斜率）")   # the pooled fit
axes[0].legend(fontsize=7.4)                                       # compact legend
style_axes(axes[0], "合并为正、组内为负：辛普森悖论", "x", "y")          # shared style
axes[1].bar(["合并", "组 A", "组 B", "固定效应"], [b_pool, float(by_group["A"].params["x"]),   # draw the panel
             float(by_group["B"].params["x"]), b_within],               # continuation of the call
            color=[OKABE[7], OKABE[1], OKABE[5], OKABE[2]], alpha=0.92)   # the slopes
axes[1].axhline(0, color="#2B2B2B", lw=0.9)                         # zero line
style_axes(axes[1], "四种口径的斜率", "", "斜率")                     # shared style
for i, v in enumerate([b_pool, float(by_group["A"].params["x"]),        # the loop
                       float(by_group["B"].params["x"]), b_within]):   # annotate
    axes[1].text(i, v, "%.2f" % v, ha="center", va="bottom" if v >= 0 else "top", fontsize=8.6)   # value
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_simpson.png")                                       # save the figure
print("done: Simpson's paradox, pooled versus within")               # closing log
