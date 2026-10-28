"""l13_05_var_coint.py -- vector autoregressions, cointegration and a pairs spread.

Every line carries an English comment. The script estimates a VAR on two related price series,   # continuation of the call
runs the Engle-Granger cointegration test, builds the mean-reverting spread, and draws the price   # continuation of the call
paths, the spread and its autocorrelation.                              # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.api as sm                                       # OLS for the cointegration step
from statsmodels.tsa.api import VAR                                # the VAR estimator
from statsmodels.tsa.stattools import coint, adfuller              # cointegration and unit root
from _common_l13 import (DAT, RNG, save, dump, style_axes,          # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) two linked series
prices = pd.read_csv(DAT / "l13_prices.csv", dtype={"ticker": str})   # the price panel
a = prices[prices["ticker"] == "600519"].set_index("date")["price"]    # series A
b = prices[prices["ticker"] == "000858"].set_index("date")["price"]    # series B (same sector)
frame = pd.DataFrame({"a": a.values, "b": b.values})                 # the pair
print("observations:", len(frame))                                   # report

# ---------------------------------------------------------------- 2) the VAR
var = VAR(frame).fit(maxlags=5, ic="aic")                            # select lags by AIC
print("VAR lag order:", var.k_ar)                                    # report
irf = var.irf(10)                                                   # impulse responses over ten steps
print("VAR coefficients:\n", var.params.round(4))                    # report
dump(pd.DataFrame(var.params, columns=frame.columns), "l13_var_params.csv")   # persist

# ---------------------------------------------------------------- 3) cointegration
stat, pval, crit = coint(a.values, b.values)                         # the Engle-Granger test
print("cointegration stat %.2f | p %.4f | 5%% crit %.2f" % (stat, pval, crit[1]))   # report
beta_hat = float(sm.OLS(b.values, sm.add_constant(a.values)).fit().params[1])   # the hedge ratio
spread = b.values - beta_hat * a.values                              # the spread
adf_spread = adfuller(spread, autolag="AIC")                         # is the spread stationary
print("hedge ratio %.4f | spread ADF p %.4f" % (beta_hat, adf_spread[1]))   # report
z = (spread - spread.mean()) / spread.std()                          # the standardised spread
print("spread |z| > 2 on %.1f%% of days" % (100 * float((np.abs(z) > 2).mean())))   # report
dump(pd.DataFrame({"day": np.arange(1, len(z) + 1), "spread": spread.round(4), "z": z.round(3)}),   # assemble the frame
     "l13_spread.csv")                                               # persist

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                  # three panels
axes[0].plot(a.values, lw=1.0, color=OKABE[1], label="A")             # series A
axes[0].plot(b.values, lw=1.0, color=OKABE[2], label="B")             # series B
axes[0].legend(fontsize=7.6)                                         # compact legend
style_axes(axes[0], "两个同板块的股票价格", "交易日", "价格")            # shared style
axes[1].plot(z, lw=0.9, color=OKABE[5])                              # the standardised spread
axes[1].axhline(2, ls="--", lw=1.0, color=OKABE[7])                   # the upper band
axes[1].axhline(-2, ls="--", lw=1.0, color=OKABE[7])                  # the lower band
style_axes(axes[1], "配对价差（z 分数）：围绕 0 波动", "交易日", "z")      # shared style
annotated(axes[1], "ADF p = %.3f" % adf_spread[1], "upper right")      # annotate
from statsmodels.graphics.tsaplots import plot_acf                   # the ACF helper
plot_acf(spread, ax=axes[2], lags=40)                                # the spread's ACF
axes[2].set_title("价差的 ACF：快速衰减 = 均值回复")                     # panel title
plt.tight_layout()                                                   # tidy the layout
save(fig, "fig_cointegration.png")                                   # save the figure
print("done: VAR, cointegration and the pairs spread")                 # closing log
