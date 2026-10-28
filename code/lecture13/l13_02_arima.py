"""l13_02_arima.py -- ACF/PACF, ARIMA orders, forecasts and the implied-volatility smile.

Every line carries an English comment. The script reads the market series built in lesson 13.1,   # continuation of the call
plots the ACF/PACF of three simulated processes, selects and estimates an ARIMA model, draws the   # continuation of the call
forecast band, and inverts Black-Scholes to recover the volatility smile from the option chain.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf        # autocorrelation plots
from statsmodels.tsa.arima.model import ARIMA                      # the ARIMA estimator
from statsmodels.stats.diagnostic import acorr_ljungbox             # residual test
from scipy.optimize import brentq                                   # root finding for implied vol
from _common_l13 import (DAT, make_option_chain, black_scholes_call,   # shared helpers
                         save, dump, style_axes, annotated, OKABE, RNG)   # continuation of the call

# ---------------------------------------------------------------- 1) the series
market = pd.read_csv(DAT / "l13_market.csv")["mkt_ret"].values      # the market returns
print("observations:", len(market), "| sd %.4f" % market.std())      # report

# ---------------------------------------------------------------- 2) three processes for the ACF panel
n = 500                                                            # sample size
ar1 = np.zeros(n)                                                  # AR(1)
ma1 = np.zeros(n)                                                  # MA(1)
arma = np.zeros(n)                                                 # ARMA(1,1)
noise = RNG.normal(0, 1, n)                                        # the shocks
for t in range(1, n):                                              # simulate the processes
    ar1[t] = 0.7 * ar1[t - 1] + noise[t]                            # AR(1) with phi = 0.7
    ma1[t] = noise[t] + 0.6 * noise[t - 1]                          # MA(1) with theta = 0.6
    arma[t] = 0.6 * arma[t - 1] + noise[t] + 0.4 * noise[t - 1]     # ARMA(1,1)
fig, axes = plt.subplots(3, 2, figsize=(11.6, 6.6))                 # three rows, two columns
for row, (name, series) in enumerate([("AR(1)", ar1), ("MA(1)", ma1), ("ARMA(1,1)", arma)]):   # loop over the processes
    plot_acf(series, ax=axes[row, 0], lags=20, title="%s 的 ACF" % name)   # the ACF
    plot_pacf(series, ax=axes[row, 1], lags=20, title="%s 的 PACF" % name, method="ywm")   # the PACF
    axes[row, 0].set_xlabel("")                                     # drop the redundant label
    axes[row, 1].set_xlabel("")                                     # drop the redundant label
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_acf_pacf.png")                                      # save the ACF/PACF figure

# ---------------------------------------------------------------- 3) ARIMA order selection
rows = []                                                          # one row per candidate order
for p in range(3):                                                 # AR orders 0-2
    for q in range(3):                                             # MA orders 0-2
        try:                                                       # some orders fail to converge
            fit = ARIMA(market[:1200], order=(p, 0, q)).fit()       # estimate on the first part
            rows.append({"p": p, "q": q, "aic": round(float(fit.aic), 2),   # the information criteria
                         "bic": round(float(fit.bic), 2)})          # the BIC
        except Exception:                                          # skip a failed fit
            continue                                               # next order
table = pd.DataFrame(rows).sort_values("aic")                       # rank by AIC
dump(table, "l13_arima_orders.csv")                                # persist
best = table.iloc[0]                                               # the best order
print("best by AIC: ARMA(%d,%d) AIC %.2f" % (best["p"], best["q"], best["aic"]))   # report

# ---------------------------------------------------------------- 4) estimate and forecast
model = ARIMA(market[:1200], order=(1, 0, 1)).fit()                 # a fixed order for the figure
forecast = model.get_forecast(steps=120)                            # 120 steps ahead
mean_fc = forecast.predicted_mean                                   # the point forecast
ci = forecast.conf_int(alpha=0.05)                                  # the 95% band
lb = acorr_ljungbox(model.resid, lags=[10], return_df=True)          # the residual test
print("Ljung-Box p on residuals: %.3f" % float(lb["lb_pvalue"].iloc[0]))   # report
fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.9))                 # two panels
axes[0].plot(np.arange(1100, 1200), market[1100:1200], lw=1.2, color=OKABE[1], label="历史")   # history
axes[0].plot(np.arange(1200, 1320), mean_fc, lw=1.6, color=OKABE[5], label="预测")   # forecast
axes[0].fill_between(np.arange(1200, 1320), ci[:, 0], ci[:, 1], color=OKABE[5], alpha=0.18)   # the band
axes[0].legend(fontsize=7.6)                                        # compact legend
style_axes(axes[0], "ARIMA(1,0,1) 的 120 步预测与 95% 区间", "交易日", "市场日收益")   # shared style
axes[1].plot(market[:1200] - market[:1200].mean(), lw=0.8, color=OKABE[3])   # the demeaned series
style_axes(axes[1], "收益率本身几乎没有可预测的趋势", "交易日", "偏离均值")   # shared style
annotated(axes[1], "R² 通常 < 1%", "upper right")                     # annotate the honest truth
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_arima_forecast.png")                                # save the forecast figure

# ---------------------------------------------------------------- 5) the implied-volatility smile
chain = make_option_chain(100.0, 0.02, 17)                          # the option chain
dump(chain, "l13_option_chain.csv")                                # persist
spot, r = 100.0, 0.02                                              # spot and rate
def implied_vol(price, strike):                                    # invert Black-Scholes numerically
    """Return the implied volatility for one option price."""       # short docstring
    try:                                                           # the root may not exist
        return float(brentq(lambda s: black_scholes_call(spot, strike, r, s) - price, 1e-4, 3.0))   # bisection
    except ValueError:                                             # outside the bracket
        return np.nan                                              # return a missing value
chain["iv_recovered"] = [implied_vol(p, k) for p, k in zip(chain["call_price"], chain["strike"])]   # recover it
dump(chain, "l13_option_chain_iv.csv")                             # persist
print("ATM IV: %.3f | wing IV: %.3f" % (float(chain["iv"].iloc[8]), float(chain["iv"].iloc[0])))   # report
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].plot(chain["strike"], chain["iv"] * 100, "-o", ms=4, color=OKABE[5], label="真实 IV")   # the smile
axes[0].plot(chain["strike"], chain["iv_recovered"] * 100, "-s", ms=3.4, color=OKABE[2], label="反解 IV")   # the recovered
axes[0].axhline(22, ls="--", lw=1.0, color="#66717D")                # the flat BS assumption
style_axes(axes[0], "隐含波动率微笑：BS 假设常数波动率是错的", "行权价", "IV (%)")   # shared style
axes[0].legend(fontsize=7.6)                                        # compact legend
axes[1].plot(chain["strike"], chain["call_price"], "-o", ms=4, color=OKABE[1])   # the option prices
style_axes(axes[1], "看涨期权价格随行权价下降", "行权价", "价格")        # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_iv_smile.png")                                      # save the smile figure
print("done: ACF/PACF, ARIMA and the implied-volatility smile")      # closing log
