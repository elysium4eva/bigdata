"""l12_05_did.py -- difference-in-differences and the event study.

Every line carries an English comment. The script estimates a two-way fixed-effects DID on a   # continuation of the call
two-group panel, runs an event study to check parallel trends, and draws both the raw group means   # continuation of the call
and the event-time coefficients.                                        # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # OLS formulas
from _common_l12 import (make_panel_did, save, dump, style_axes,   # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the panel
panel = make_panel_did(400, 8, 4, -3.0)                            # 400 units, 8 periods, shock at 4
dump(panel, "l12_did_panel.csv")                                    # persist
print("rows:", len(panel), "| treated share: %.2f" % panel["treated"].mean())   # report

# ---------------------------------------------------------------- 2) the DID regression
panel["did"] = panel["treated"] * panel["post"]                     # the interaction
model = smf.ols("y ~ did + C(unit) + C(period)", data=panel).fit(       # estimate the model
    cov_type="cluster", cov_kwds={"groups": panel["unit"]})          # two-way FE with clustered SEs
print("DID coefficient: %.3f (se %.3f)" % (model.params["did"], model.bse["did"]))   # report
means = panel.groupby(["treated", "post"])["y"].mean().round(3)      # the four group-period means
print(means.to_string())                                           # report
dump(pd.DataFrame([{"estimate": "DID", "coef": round(float(model.params["did"]), 3),   # assemble the frame
                    "se": round(float(model.bse["did"]), 3)}]), "l12_did.csv")   # persist

# ---------------------------------------------------------------- 3) the event study
# With a single treatment date and two groups the event study is simply the period-by-period
# difference between the groups, re-based at the last pre-treatment period.
per = panel.groupby(["period", "treated"])["y"].agg(["mean", "std", "count"]).reset_index()   # group means
wide = per.pivot(index="period", columns="treated", values="mean")   # treated vs control means
se_wide = per.pivot(index="period", columns="treated", values="std")    # continuation of the call
n_wide = per.pivot(index="period", columns="treated", values="count")   # continuation of the call
diff = (wide[1] - wide[0])                                          # the group difference by period
se_diff = np.sqrt(se_wide[1] ** 2 / n_wide[1] + se_wide[0] ** 2 / n_wide[0])   # its standard error
ref_period = 3                                                      # the last pre-treatment period
base_value = float(diff.loc[ref_period])                             # the reference level
event = pd.DataFrame({"event_time": diff.index - ref_period,        # relative period
                      "coef": diff.values - base_value,             # deviation from the reference
                      "se": se_diff.values})                        # the standard error
event = event[event["event_time"] != 0]                              # drop the reference row
dump(event.round(4), "l12_event_study.csv")                          # persist
print(event.round(3).to_string(index=False))                         # report
pre = event[event["event_time"] < 0]                                 # the pre-treatment coefficients
print("max |pre| coefficient: %.3f" % float(pre["coef"].abs().max()))   # a parallel-trends check
# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
for treated, colour, label in ((0, OKABE[1], "对照组"), (1, OKABE[5], "处理组")):   # loop over the groups
    sub = panel[panel["treated"] == treated].groupby("period")["y"].mean()   # the group mean by period
    axes[0].plot(sub.index, sub.values, "-o", ms=5, color=colour, label=label)   # the series
axes[0].axvline(3.5, ls="--", lw=1.0, color="#66717D")               # the treatment date
axes[0].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[0], "平行趋势：处理前两条线应平行", "期", "均值 y")        # shared style
annotated(axes[0], "处理后分离", "lower left")                        # annotate
axes[1].errorbar(event["event_time"], event["coef"], yerr=1.96 * event["se"], fmt="-o",   # draw the panel
                 ms=4.5, color=OKABE[5], ecolor="#2B2B2B", capsize=2)   # the event-study coefficients
axes[1].axhline(0, color="#2B2B2B", lw=0.9)                          # zero line
axes[1].axvline(-0.5, ls="--", lw=1.0, color="#66717D")               # the treatment date
style_axes(axes[1], "事件研究：处理前系数应接近 0", "相对期", "系数")        # shared style
annotated(axes[1], "DID = %.2f" % model.params["did"], "lower left")     # annotate
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_event_study.png")                                    # save the figure
print("done: DID and the event study")                                 # closing log
