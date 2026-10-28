"""l14_03_ppml.py -- PPML versus log-OLS when there are zero flows.

Every line carries an English comment. The script re-estimates the gravity equation four ways   # continuation of the call
(log OLS on positives, log(1+x), truncation, PPML), shows how the zero-trade share distorts the   # continuation of the call
log-OLS estimates, and draws the comparison that motivates Poisson pseudo-maximum likelihood.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.api as sm                                       # GLM machinery
import statsmodels.formula.api as smf                              # OLS formulas
from _common_l14 import (make_country_panel, make_geography,       # shared generators
                         make_trade_flows, TRUE_BETA, save, dump,       # shared generators
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the panel
countries = make_country_panel(60, (2000, 2019))                    # country-year panel
geo = make_geography(60)                                           # bilateral geography
flows = make_trade_flows(countries, geo, (2000, 2019))              # dyad-year flows
print("dyad-years:", len(flows), "| zeros: %.3f" % (flows["trade"] == 0).mean())   # report
X_COLS = ["lgdp_o", "lgdp_d", "ldist", "contig", "comlang", "colony", "rta", "cur"]   # covariates
formula = "y ~ " + " + ".join(X_COLS)                               # the shared right-hand side

# ---------------------------------------------------------------- 2) four estimators
rows = []                                                          # one row per estimator
# (a) log-OLS on positive flows only
pos = flows[flows["trade"] > 0].copy()                              # the truncated sample
pos["y"] = np.log(pos["trade"])                                     # log outcome
a = smf.ols(formula, data=pos).fit()                                # OLS on positives
rows.append({"estimator": "对数 OLS（仅正贸易）", "ldist": float(a.params["ldist"]),   # append the record
             "rta": float(a.params["rta"]), "n": len(pos)})          # record
# (b) log(1+x) on all flows
allf = flows.copy()                                                # the full sample
allf["y"] = np.log1p(allf["trade"])                                 # log(1+x)
b = smf.ols(formula, data=allf).fit()                               # OLS on the transformed outcome
rows.append({"estimator": "log(1+x) OLS", "ldist": float(b.params["ldist"]),   # append the record
             "rta": float(b.params["rta"]), "n": len(allf)})         # record
# (c) PPML: Poisson pseudo-maximum likelihood on the levels
c = smf.glm(formula, data=allf, family=sm.families.Poisson()).fit(cov_type="HC1")   # GLM Poisson
rows.append({"estimator": "PPML（泊松）", "ldist": float(c.params["ldist"]),   # PPML on all flows
             "rta": float(c.params["rta"]), "n": len(allf)})         # record
# (d) PPML on the truncated sample, to show the sample matters too
d = smf.glm(formula, data=pos, family=sm.families.Poisson()).fit(cov_type="HC1")   # PPML on positives
rows.append({"estimator": "PPML（仅正贸易）", "ldist": float(d.params["ldist"]),   # append the record
             "rta": float(d.params["rta"]), "n": len(pos)})          # record
comp = pd.DataFrame(rows)                                          # assemble
comp["true_ldist"] = TRUE_BETA["ldist"]                             # the DGP value
dump(comp.round(4), "l14_ppml_compare.csv")                        # persist
print(comp.round(3).to_string(index=False))                        # report

# ---------------------------------------------------------------- 3) the bias versus the zero share
shares, biases = [], []                                            # accumulators
for keep in [1.0, 0.9, 0.75, 0.6, 0.45, 0.3]:                       # keep a growing share of flows
    sub = flows.sample(frac=keep, random_state=42) if keep < 1 else flows   # subsample or all
    sub_pos = sub[sub["trade"] > 0].copy()                          # positives only
    if len(sub_pos) < 100:                                          # guard against tiny samples
        continue                                                   # skip
    sub_pos["y"] = np.log(sub_pos["trade"])                         # log outcome
    m = smf.ols(formula, data=sub_pos).fit()                        # log-OLS
    shares.append(float((sub["trade"] == 0).mean()))                # the realised zero share
    biases.append(float(m.params["ldist"] - TRUE_BETA["ldist"]))    # the bias in the distance term
curve = pd.DataFrame({"zero_share": shares, "bias": biases}).round(4)   # assemble
dump(curve, "l14_zero_bias.csv")                                   # persist
print(curve.to_string(index=False))                                # report

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].hist(np.log10(flows.loc[flows["trade"] > 0, "trade"]), bins=40,   # the flow distribution
             color=OKABE[1], alpha=0.85)                            # histogram
axes[0].set_yscale("log")                                          # counts span orders of magnitude
style_axes(axes[0], "Positive flows: log-normal-ish", "log10 贸易额", "频数（对数）")   # shared style
annotated(axes[0], "另有 %.0f%% 的零贸易没画进来" % (100 * (flows["trade"] == 0).mean()), "upper left")   # annotate
xs = np.arange(len(comp))                                          # bar positions
axes[1].bar(xs, comp["ldist"], color=[OKABE[1], OKABE[3], OKABE[2], OKABE[4]], alpha=0.92)   # estimates
axes[1].axhline(TRUE_BETA["ldist"], ls="--", lw=1.0, color=OKABE[5])   # the true value
axes[1].set_xticks(xs, ["对数 OLS", "log(1+x)", "PPML", "PPML 截尾"], rotation=12, fontsize=8)   # labels
style_axes(axes[1], "Distance elasticity by estimator", "", "coefficient")   # shared style
annotated(axes[1], "PPML 最接近真值", "lower right")                    # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_zero_trade.png")                                    # save the figure

# ---------------------------------------------------------------- 5) the bias curve
fig, ax = plt.subplots(figsize=(7.4, 3.6))                          # one panel
ax.plot(curve["zero_share"], curve["bias"], "-o", ms=5, color=OKABE[5])   # bias versus zeros
ax.axhline(0, color="#2B2B2B", lw=0.9)                              # the truth
style_axes(ax, "Zero-trade share and the log-OLS bias", "零贸易占比", "距离弹性偏差")   # shared style
annotated(ax, "零越多，偏差越大", "upper left")                       # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_zero_bias.png")                                     # save
print("done: PPML versus log-OLS with zero flows")                    # closing log
