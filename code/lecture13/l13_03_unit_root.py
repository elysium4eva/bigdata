"""l13_03_unit_root.py -- stationarity, the ADF test and differencing.

Every line carries an English comment. The script tests the market series for a unit root, shows   # continuation of the call
what differencing does to the autocorrelation, and demonstrates the spurious-regression trap that   # continuation of the call
makes stationarity the first thing to check in finance.                 # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from statsmodels.tsa.stattools import adfuller                     # the augmented Dickey-Fuller test
from _common_l13 import (DAT, RNG, save, dump, style_axes,          # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the two series
market = pd.read_csv(DAT / "l13_market.csv")["mkt_ret"].values      # returns (already stationary)
prices = pd.read_csv(DAT / "l13_prices.csv", dtype={"ticker": str})   # the price panel (ticker as text)
one = prices[prices["ticker"] == "600519"]["price"].values           # one price path (non-stationary)

def adf_summary(series, label):                                   # run and format the ADF test
    """Return a dict with the ADF statistic, p value and the chosen lag."""   # short docstring
    stat, pval, usedlag, nobs, crit, _ = adfuller(series, autolag="AIC")   # the ADF test
    return {"series": label, "adf": round(float(stat), 3), "p_value": round(float(pval), 4),   # the statistics
            "lags": int(usedlag), "n": int(nobs), "crit_5pct": round(float(crit["5%"]), 3)}   # more detail

rows = [adf_summary(one, "价格水平"),                                # the price level
        adf_summary(np.diff(one), "价格一阶差分"),                    # the first difference
        adf_summary(market, "市场收益率")]                            # the return series
table = pd.DataFrame(rows)                                         # assemble
dump(table, "l13_adf.csv")                                         # persist
print(table.to_string(index=False))                                # report

# ---------------------------------------------------------------- 2) the spurious regression demo
n = 300                                                            # sample size
x = np.cumsum(0.20 + RNG.normal(0, 1, n))                          # a random walk with drift
y = np.cumsum(0.15 + RNG.normal(0, 1, n))                          # an independent random walk with drift
import statsmodels.api as sm                                       # the OLS machinery
res = sm.OLS(y, sm.add_constant(x)).fit()                           # regress one random walk on another
print("spurious regression R2: %.3f | t of slope: %.2f" % (res.rsquared, res.tvalues[1]))   # report
spurious = pd.DataFrame([{"r2": round(float(res.rsquared), 4), "t_slope": round(float(res.tvalues[1]), 2),   # assemble the frame
                          "adf_x": round(float(adfuller(x)[0]), 3), "adf_y": round(float(adfuller(y)[0]), 3)}])   # continuation of the call
dump(spurious, "l13_spurious.csv")                                 # persist

# ---------------------------------------------------------------- 3) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                 # three panels
axes[0].plot(one, lw=1.0, color=OKABE[4])                          # the price level
style_axes(axes[0], "价格水平：均值随时间漂移", "交易日", "价格")        # shared style
annotated(axes[0], "ADF p = %.2f" % rows[0]["p_value"], "upper left")   # annotate
axes[1].plot(np.diff(one), lw=0.8, color=OKABE[2])                 # the first difference
style_axes(axes[1], "一阶差分：回到均值附近", "交易日", "差分")          # shared style
annotated(axes[1], "ADF p = %.3f" % rows[1]["p_value"], "upper left")   # annotate
axes[2].scatter(x, y, s=8, alpha=0.7, color=OKABE[5])              # the two random walks
axes[2].plot(x, res.params[0] + res.params[1] * x, lw=1.6, color=OKABE[7])   # the fitted line
style_axes(axes[2], "伪回归：两个随机游走也能拟合出 R²=%.2f" % res.rsquared, "随机游走 X", "随机游走 Y")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_unit_root.png")                                     # save the figure
print("done: ADF, differencing and the spurious-regression trap")     # closing log
