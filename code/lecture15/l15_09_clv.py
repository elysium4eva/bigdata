"""l15_09_clv.py -- retention curves, discounted customer lifetime value and value tiers.

Every line carries an English comment. The script fits retention curves for four customer tiers,
computes CLV with a monthly discount rate, shows how the discount rate moves the answer, and
builds the value x churn-probability matrix the deck uses for retention strategy.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l15 import (RNG, save, dump, style_axes, annotated,   # shared helpers
                         OKABE)                                         # shared plotting helpers

# ---------------------------------------------------------------- 1) four tiers with different retention
tiers = {"高价值": {"m0": 260.0, "decay": 0.028, "n": 250},          # high value, slow decay
         "成长": {"m0": 150.0, "decay": 0.040, "n": 450},            # mid value
         "价格敏感": {"m0": 80.0, "decay": 0.062, "n": 550},          # fast decay
         "一次性": {"m0": 45.0, "decay": 0.090, "n": 750}}            # very fast decay
months = np.arange(0, 37)                                          # three years of months
discount = 0.01                                                    # 1% monthly discount rate
rows = []                                                          # one row per tier-month
curves = {}                                                        # retention curves for the figure
for name, cfg in tiers.items():                                    # loop over the tiers
    survival = np.exp(-cfg["decay"] * months)                       # exponential retention curve
    margin = cfg["m0"] * np.exp(-0.012 * months)                    # slowly shrinking margin
    disc = (1 + discount) ** (-months)                              # discount factors
    clv_cum = np.cumsum(survival * margin * disc)                   # cumulative discounted value
    curves[name] = {"survival": survival, "clv_cum": clv_cum, "n": cfg["n"]}   # keep for plotting
    for t in months:                                               # one row per month
        rows.append({"tier": name, "month": int(t),                  # ids
                     "survival": round(float(survival[t]), 4),       # retention
                     "margin": round(float(margin[t]), 2),           # monthly margin
                     "clv_cum": round(float(clv_cum[t]), 2)})        # cumulative CLV
clv_long = pd.DataFrame(rows)                                      # assemble
dump(clv_long, "l15_clv_curve.csv")                                # persist
summary = clv_long.groupby("tier")["clv_cum"].max().round(1)        # three-year CLV by tier
weights = pd.Series({k: v["n"] for k, v in tiers.items()})         # tier sizes
weighted = float((summary * weights).sum() / weights.sum())         # customer-weighted CLV
print(summary.to_string())                                         # report
print("weighted 36-month CLV: %.0f 元" % weighted)                  # report

# ---------------------------------------------------------------- 2) discount-rate sensitivity
rates = [0.0, 0.005, 0.01, 0.02, 0.03]                             # candidate monthly rates
sens = []                                                          # CLV per rate
for r in rates:                                                    # loop over the rates
    cfg = tiers["成长"]                                             # use the middle tier
    survival = np.exp(-cfg["decay"] * months)                       # the same retention curve
    margin = cfg["m0"] * np.exp(-0.012 * months)                    # the same margin
    total = float(np.sum(survival * margin * (1 + r) ** (-months)))   # discounted sum
    sens.append({"monthly_rate": r, "clv": round(total, 1)})         # record
sensitivity = pd.DataFrame(sens)                                   # assemble
dump(sensitivity, "l15_clv_sensitivity.csv")                       # persist
print(sensitivity.to_string(index=False))                          # report

# ---------------------------------------------------------------- 3) the value x churn matrix
grid_value = [summary.min(), summary.max()]                        # value range
churn_lo, churn_hi = 0.05, 0.45                                    # churn probability range
RNG_local = np.random.default_rng(7)                               # a local generator
points = []                                                        # simulated customers
for name, cfg in tiers.items():                                    # loop over the tiers
    n = cfg["n"]                                                   # tier size
    clv = float(summary[name])                                     # tier CLV
    churn_p = 1 - float(np.exp(-cfg["decay"] * 12))                 # annual churn implied by decay
    points.append({"tier": name, "clv": clv, "churn": churn_p, "n": n})   # one point per tier
matrix = pd.DataFrame(points)                                      # assemble
dump(matrix.round(3), "l15_value_churn_matrix.csv")                # persist
print(matrix.round(3).to_string(index=False))                      # report

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                 # three panels
for k, (name, data) in enumerate(curves.items()):                   # one retention curve per tier
    axes[0].plot(months, data["survival"], lw=1.8, color=OKABE[k], label=name)   # the curve
style_axes(axes[0], "Retention curves by tier", "month", "留存率")    # shared style
axes[0].legend(fontsize=7.2, loc="upper right")                     # compact legend
for k, (name, data) in enumerate(curves.items()):                   # cumulative CLV per tier
    axes[1].plot(months, data["clv_cum"], lw=1.8, color=OKABE[k], label=name)   # the curve
style_axes(axes[1], "Cumulative discounted CLV", "month", "CLV (元)")   # shared style
axes[1].legend(fontsize=7.2, loc="upper left")                      # compact legend
axes[2].plot(sensitivity["monthly_rate"] * 100, sensitivity["clv"], "-o", ms=5, color=OKABE[5])   # sensitivity
style_axes(axes[2], "Sensitivity to the discount rate (成长 tier)", "月折现率 (%)", "CLV (元)")   # shared style
annotated(axes[2], "折现率翻倍，CLV 明显缩水", "upper right")          # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_clv_curves.png")                                    # save the figure
print("done: retention curves, discounted CLV and the value matrix")   # closing log
