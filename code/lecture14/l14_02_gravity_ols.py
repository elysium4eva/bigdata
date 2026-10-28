"""l14_02_gravity_ols.py -- the gravity equation with OLS and with fixed effects.

Every line carries an English comment. The script builds the bilateral panel, estimates the   # continuation of the call
gravity equation three ways (pooled OLS, exporter-importer-year effects, and pair effects on a   # continuation of the call
subsample), and shows how the distance elasticity moves, plus the trade-cost index implied by   # continuation of the call
the fixed effects.                                                      # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # OLS formulas
from _common_l14 import (make_country_panel, make_geography,       # shared generators
                         make_trade_flows, TRUE_BETA, save, dump,       # shared generators
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the panel
countries = make_country_panel(60, (2000, 2019))                    # the country-year panel
geo = make_geography(60)                                           # bilateral geography
flows = make_trade_flows(countries, geo, (2000, 2019))              # the dyad-year flows
dump(flows, "l14_trade_flows.csv")                                 # persist
print("dyad-years:", len(flows), "| zero-trade share: %.3f" % (flows["trade"] == 0).mean())   # report

# ---------------------------------------------------------------- 2) the two stylised facts
positive = flows[flows["trade"] > 0].copy()                         # flows we can log
positive["ltrade"] = np.log(positive["trade"])                      # log trade
positive["lgdp_sum"] = positive["lgdp_o"] + positive["lgdp_d"]       # combined size
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].scatter(positive["lgdp_sum"], positive["ltrade"], s=4, alpha=0.18, color=OKABE[1])   # size vs trade
b1 = np.polyfit(positive["lgdp_sum"], positive["ltrade"], 1)        # a simple slope
grid = np.linspace(positive["lgdp_sum"].min(), positive["lgdp_sum"].max(), 50)   # a grid
axes[0].plot(grid, np.polyval(b1, grid), lw=2.0, color=OKABE[5])   # fitted line
style_axes(axes[0], "规模越大，贸易越多（斜率 ≈ %.2f）" % b1[0], "两国 GDP 之和（对数）", "双边贸易（对数）")   # shared style
axes[1].scatter(positive["ldist"], positive["ltrade"], s=4, alpha=0.18, color=OKABE[2])   # distance vs trade
b2 = np.polyfit(positive["ldist"], positive["ltrade"], 1)          # the distance slope
grid2 = np.linspace(positive["ldist"].min(), positive["ldist"].max(), 50)   # a grid
axes[1].plot(grid2, np.polyval(b2, grid2), lw=2.0, color=OKABE[5])   # fitted line
style_axes(axes[1], "距离越远，贸易越少（斜率 ≈ %.2f）" % b2[0], "距离（对数）", "双边贸易（对数）")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_gravity_scatter.png")                               # save the scatter figure

# ---------------------------------------------------------------- 3) pooled OLS
formula = "ltrade ~ lgdp_o + lgdp_d + ldist + contig + comlang + colony + rta + cur"   # naive gravity
ols = smf.ols(formula, data=positive).fit(cov_type="cluster", cov_kwds={"groups": positive["iso_o"]})   # clustered SEs
print("pooled OLS: ldist %.3f (se %.3f) | lgdp_o %.3f | rta %.3f" % (   # report the pooled estimates
    ols.params["ldist"], ols.bse["ldist"], ols.params["lgdp_o"], ols.params["rta"]))   # report

# ---------------------------------------------------------------- 4) exporter, importer and year fixed effects
fe_formula = formula + " + C(iso_o) + C(iso_d) + C(year)"            # country and year effects
fe = smf.ols(fe_formula, data=positive).fit(cov_type="cluster", cov_kwds={"groups": positive["iso_o"]})   # clustered
print("FE: ldist %.3f (se %.3f) | contig %.3f | rta %.3f" % (           # report the FE estimates
    fe.params["ldist"], fe.bse["ldist"], fe.params["contig"], fe.params["rta"]))   # report
print("FE: lgdp_o %.3f | lgdp_d %.3f" % (fe.params["lgdp_o"], fe.params["lgdp_d"]))   # GDP stays identified
print("FE R2: %.3f | pooled R2: %.3f" % (fe.rsquared, ols.rsquared))   # report the fits

# ---------------------------------------------------------------- 5) pair fixed effects on a subsample
sub = positive[positive["year"] >= 2015].copy()                     # recent years only, to keep it fast
pair_fe = smf.ols("ltrade ~ ldist + rta + C(iso_o) + C(iso_d) + C(year)", data=sub).fit()   # pair-level effects proxy
print("pair-FE subsample: ldist %.3f | rta %.3f" % (pair_fe.params["ldist"], pair_fe.params["rta"]))   # report

# ---------------------------------------------------------------- 6) compare the estimates
comp = pd.DataFrame([                                                   # the comparison table
    {"spec": "理论真值", "ldist": TRUE_BETA["ldist"], "rta": TRUE_BETA["rta"]},   # the DGP
    {"spec": "混合 OLS", "ldist": float(ols.params["ldist"]), "rta": float(ols.params["rta"])},   # pooled
    {"spec": "国家+年份 FE", "ldist": float(fe.params["ldist"]), "rta": float(fe.params["rta"])},   # FE
    {"spec": "近期子样本", "ldist": float(pair_fe.params["ldist"]), "rta": float(pair_fe.params["rta"])},   # subsample
])                                                                      # end of the table
dump(comp.round(4), "l14_gravity_compare.csv")                     # persist
print(comp.round(3).to_string(index=False))                        # report

# ---------------------------------------------------------------- 7) the fixed-effect figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
xs = np.arange(len(comp))                                          # bar positions
axes[0].bar(xs, comp["ldist"], color=[OKABE[7], OKABE[1], OKABE[2], OKABE[3]], alpha=0.9)   # distance elasticity
axes[0].axhline(TRUE_BETA["ldist"], ls="--", lw=1.0, color="#66717D")   # the true value
axes[0].set_xticks(xs, comp["spec"], rotation=15, fontsize=8)       # labels
style_axes(axes[0], "Distance elasticity by specification", "", "coefficient")   # shared style
axes[1].bar(xs, comp["rta"], color=[OKABE[7], OKABE[1], OKABE[2], OKABE[3]], alpha=0.9)   # RTA effect
axes[1].axhline(TRUE_BETA["rta"], ls="--", lw=1.0, color="#66717D")   # the true value
axes[1].set_xticks(xs, comp["spec"], rotation=15, fontsize=8)        # labels
style_axes(axes[1], "RTA effect by specification", "", "coefficient")   # shared style
annotated(axes[0], "FE 把遗漏的多边阻力吸掉了", "lower right")          # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_fe_compare.png")                                    # save the comparison figure

# ---------------------------------------------------------------- 8) the implied trade-cost index
# With exporter and importer effects, the pair residual is a trade-cost measure up to a constant.
positive["resid"] = fe.resid                                        # the residual
cost = positive.groupby("iso_d")["resid"].mean().sort_values()       # inward trade cost by importer
cost_index = pd.DataFrame({"iso": cost.index, "cost_index": (cost - cost.min()).round(3).values})   # a relative index
dump(cost_index, "l14_trade_cost.csv")                             # persist
fig, ax = plt.subplots(figsize=(11.6, 3.8))                         # one wide panel
ax.bar(range(len(cost_index)), cost_index["cost_index"], color=OKABE[4], alpha=0.9)   # the index
ax.set_xticks(range(len(cost_index)), cost_index["iso"], rotation=70, fontsize=6.4)   # country labels
style_axes(ax, "implied inward trade cost (relative index)", "", "relative cost")   # shared style
annotated(ax, "由 FE 残差反解，只可比较、不可当绝对水平", "upper left")   # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_trade_cost.png")                                    # save the cost figure
print("done: gravity by OLS, fixed effects and the implied trade cost")   # closing log
