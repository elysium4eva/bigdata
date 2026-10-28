"""l14_10_gpr.py -- a geopolitical-risk index and its link to trade.

Every line carries an English comment. The script builds a monthly risk index from keyword   # continuation of the call
counts, standardises it, regresses world trade growth on the index, and draws the time series and   # continuation of the call
the scatter so the deck can show how a text-based risk measure is constructed and validated.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # the regression
from _common_l14 import (make_gpr, RNG, save, dump, style_axes,     # shared generators
                         annotated, OKABE)                              # shared plotting helpers

# ---------------------------------------------------------------- 1) the index
gpr = make_gpr(120)                                                # ten years of monthly data
gpr["index_z"] = ((gpr["index"] - gpr["index"].mean()) / gpr["index"].std()).round(3)   # standardised
dump(gpr, "l14_gpr.csv")                                           # persist
print("months:", len(gpr), "| index range: %.1f–%.1f" % (gpr["index"].min(), gpr["index"].max()))   # report
print("event months:", int(gpr["event"].sum()))                     # report

# ---------------------------------------------------------------- 2) link to trade growth
# trade growth responds negatively to risk, with a one-month lag
growth = -1.8 * gpr["index_z"].shift(1) + RNG.normal(0, 2.4, len(gpr))   # synthetic trade growth
gpr["trade_growth"] = growth.round(2)                               # attach it
model = smf.ols("trade_growth ~ index_z", data=gpr.dropna()).fit(cov_type="HC1")   # robust regression
print("risk coefficient: %.3f (se %.3f) | R2 %.3f" % (                  # report the result
    model.params["index_z"], model.bse["index_z"], model.rsquared))   # report
dump(pd.DataFrame([{"term": model.params.index[i], "coef": round(float(model.params.iloc[i]), 3),   # assemble the frame
                    "se": round(float(model.bse.iloc[i]), 3)} for i in range(len(model.params))]),   # continuation of the call
     "l14_gpr_regression.csv")                                     # persist

# ---------------------------------------------------------------- 3) the figure
fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.9))                 # two panels
axes[0].plot(gpr["month"], gpr["index"], lw=1.4, color=OKABE[5])     # the index
event_months = gpr.loc[gpr["event"] == 1, "month"].unique()          # the shock months
for m in event_months:                                             # mark the shocks
    axes[0].axvspan(m, m + 3, color=OKABE[3], alpha=0.25)            # a shaded window
axes[0].axhline(gpr["index"].mean(), ls="--", lw=1.0, color="#66717D")   # the mean
style_axes(axes[0], "地缘政治风险指数（月度，阴影=事件窗）", "月", "指数")   # shared style
annotated(axes[0], "三次冲击：制造、俄乌式、再起", "upper left")        # annotate
clean = gpr.dropna(subset=["index_z", "trade_growth"])            # rows with both variables
axes[1].scatter(clean["index_z"], clean["trade_growth"], s=18, alpha=0.7, color=OKABE[1])   # risk vs growth
b = np.polyfit(clean["index_z"], clean["trade_growth"], 1)          # a simple slope
grid = np.linspace(gpr["index_z"].min(), gpr["index_z"].max(), 50)    # a grid
axes[1].plot(grid, np.polyval(b, grid), lw=2.0, color=OKABE[5])       # fitted line
style_axes(axes[1], "风险越高，贸易增速越低（斜率 %.2f）" % b[0], "风险指数（标准化）", "贸易增速 %")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_gpr.png")                                           # save the figure
print("done: risk index construction and its trade link")             # closing log
