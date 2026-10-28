"""l13_04_garch.py -- a hand-written GARCH(1,1) estimated by maximum likelihood.

Every line carries an English comment. The script fits a GARCH(1,1) to the market returns with   # continuation of the call
scipy, reports persistence and the half-life of a volatility shock, compares the conditional   # continuation of the call
volatility with a rolling estimate, and draws the volatility-clustering picture.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from scipy.optimize import minimize                                # the optimiser
from _common_l13 import (DAT, save, dump, style_axes, annotated,   # shared helpers
                         OKABE)                                         # continuation of the call

# ---------------------------------------------------------------- 1) the returns
market = pd.read_csv(DAT / "l13_market.csv")["mkt_ret"].values      # the market returns
r = market * 100                                                   # work in percent for stability
print("observations:", len(r), "| sd %.3f%%" % r.std())               # report

# ---------------------------------------------------------------- 2) the likelihood
def garch_filter(omega, alpha, beta, series):                      # filter the variance recursion
    """Return the conditional variance path given the three parameters."""   # short docstring
    n = len(series)                                                # sample size
    var = np.zeros(n)                                              # the variance path
    var[0] = np.var(series)                                        # start at the unconditional variance
    for t in range(1, n):                                          # loop over time
        var[t] = omega + alpha * series[t - 1] ** 2 + beta * var[t - 1]   # the GARCH recursion
        if var[t] <= 0:                                            # guard against negative variance
            var[t] = 1e-6                                          # a tiny positive floor
    return var                                                     # the variance path

def neg_loglik(theta, series):                                     # the negative log-likelihood
    """Return the negative Gaussian log-likelihood of the GARCH model."""   # short docstring
    omega, alpha, beta = theta                                     # unpack the parameters
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:   # the stationarity constraints
        return 1e10                                                # a very bad value
    var = garch_filter(omega, alpha, beta, series)                 # the variance path
    return float(0.5 * np.sum(np.log(2 * np.pi * var) + series ** 2 / var))   # the Gaussian likelihood

# ---------------------------------------------------------------- 3) estimate
start = np.array([0.02, 0.08, 0.88])                                # starting values
bounds = [(1e-8, 1.0), (0.0, 0.5), (0.0, 0.999)]                     # parameter bounds
res = minimize(neg_loglik, start, args=(r,), method="L-BFGS-B", bounds=bounds)   # maximise the likelihood
omega, alpha, beta = res.x                                          # the estimates
var_hat = garch_filter(omega, alpha, beta, r)                       # the fitted variance
vol_hat = np.sqrt(var_hat)                                          # the conditional volatility
persistence = alpha + beta                                          # how long shocks last
half_life = float(np.log(0.5) / np.log(persistence)) if persistence < 1 else np.inf   # the half-life
print("omega %.4f | alpha %.3f | beta %.3f | persistence %.3f" % (omega, alpha, beta, persistence))   # report
print("half-life: %.1f days" % half_life)                            # report
long_run = float(np.sqrt(omega / max(1 - persistence, 1e-6)))        # the long-run volatility
print("long-run vol: %.2f%% | unconditional sd: %.2f%%" % (long_run, r.std()))   # report

# ---------------------------------------------------------------- 4) a rolling comparison
window = 20                                                        # a rolling window
rolling = pd.Series(r).rolling(window).std().bfill().values         # rolling standard deviation
err_garch = float(np.mean((vol_hat - rolling) ** 2))                # mean squared error
print("MSE vs rolling: %.4f" % err_garch)                            # report
table = pd.DataFrame([{"omega": round(float(omega), 5), "alpha": round(float(alpha), 4),   # assemble the frame
                       "beta": round(float(beta), 4), "persistence": round(float(persistence), 4),   # continuation of the call
                       "half_life_days": round(half_life, 1), "long_run_vol_pct": round(long_run, 2)}])   # continuation of the call
dump(table, "l13_garch_params.csv")                                # persist
dump(pd.DataFrame({"day": np.arange(1, len(r) + 1), "ret_pct": r.round(4),   # assemble the frame
                   "vol_garch": vol_hat.round(4), "vol_rolling": rolling.round(4)}),   # continuation of the call
     "l13_garch_vol.csv")                                          # persist the paths

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                 # three panels
axes[0].plot(r, lw=0.6, color=OKABE[1], alpha=0.8)                  # the returns
axes[0].plot(vol_hat, lw=1.4, color=OKABE[5])                       # the fitted volatility
axes[0].plot(-vol_hat, lw=1.4, color=OKABE[5])                      # the mirrored band
style_axes(axes[0], "波动聚集：大波动成串出现", "交易日", "收益率 (%)")   # shared style
axes[1].plot(vol_hat, lw=1.2, color=OKABE[5], label="GARCH")         # the GARCH volatility
axes[1].plot(rolling, lw=1.0, color=OKABE[2], label="20 日滚动")       # the rolling estimate
axes[1].legend(fontsize=7.6)                                        # compact legend
style_axes(axes[1], "GARCH 比滚动窗口更平滑", "交易日", "波动率 (%)")   # shared style
axes[2].plot(np.arange(1, 61), 0.5 ** (np.arange(1, 61) / max(half_life, 1e-9)), lw=1.8, color=OKABE[4])   # the decay
style_axes(axes[2], "冲击的半衰期 %.0f 天" % half_life, "天数", "剩余影响")   # shared style
annotated(axes[2], "持久性 %.2f" % persistence, "upper right")        # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_garch_fit.png")                                     # save the figure
print("done: GARCH(1,1) by maximum likelihood")                        # closing log
