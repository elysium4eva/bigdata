"""l13_06_portfolio.py -- the efficient frontier, constraints and factor exposures.

Every line carries an English comment. The script builds the covariance matrix from the price   # continuation of the call
panel, solves the mean-variance problem with scipy, draws the efficient frontier with and without   # continuation of the call
constraints, and regresses portfolio returns on a market factor to show the alpha/beta split.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from scipy.optimize import minimize                                # the quadratic programme
import statsmodels.api as sm                                       # the factor regression
from _common_l13 import (DAT, save, dump, style_axes, annotated,   # shared helpers
                         OKABE)                                         # continuation of the call

# ---------------------------------------------------------------- 1) inputs
prices = pd.read_csv(DAT / "l13_prices.csv", dtype={"ticker": str})   # the price panel
wide = prices.pivot(index="date", columns="ticker", values="ret")     # returns in wide form
mu = wide.mean().values * 252                                        # annualised mean returns
cov = wide.cov().values * 252                                        # annualised covariance
n = len(mu)                                                          # number of assets
print("assets:", n, "| mean range %.1f%%-%.1f%%" % (100 * mu.min(), 100 * mu.max()))   # report

def portfolio_stats(w):                                            # risk and return of a portfolio
    """Return the annualised return and volatility of a weight vector."""   # short docstring
    ret = float(w @ mu)                                            # portfolio mean
    vol = float(np.sqrt(w @ cov @ w))                              # portfolio volatility
    return ret, vol                                                # the pair

def solve(target, long_only=True, cap=1.0):                        # the mean-variance problem
    """Return the minimum-variance weights for a target return."""   # short docstring
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},          # weights sum to one
            {"type": "eq", "fun": lambda w: w @ mu - target}]      # hit the target return
    bounds = [(0.0, cap) if long_only else (-1.0, 1.0)] * n         # long-only or with shorts
    res = minimize(lambda w: float(w @ cov @ w), np.ones(n) / n,     # minimise the variance
                   bounds=bounds, constraints=cons, method="SLSQP")   # the solver
    return res.x if res.success else np.ones(n) / n                 # the weights

targets = np.linspace(float(mu.min()), float(mu.max()) * 0.9, 25)     # a grid of targets
frontier = []                                                       # one row per target
for t in targets:                                                   # loop over the targets
    w = solve(t, long_only=True)                                    # solve the problem
    ret, vol = portfolio_stats(w)                                   # its statistics
    frontier.append({"target": round(float(t), 4), "vol": round(vol, 4), "ret": round(ret, 4)})   # the row
frontier = pd.DataFrame(frontier)                                   # assemble
dump(frontier, "l13_efficient_frontier.csv")                       # persist
w_min = solve(float(mu.min()) * 0.999, long_only=True, cap=0.3)       # a capped solution
print("min-variance weights (top 5):", np.round(np.sort(w_min)[-5:], 3))   # report

# ---------------------------------------------------------------- 2) an equal-weight and a min-variance portfolio
w_eq = np.ones(n) / n                                               # equal weights
ret_eq, vol_eq = portfolio_stats(w_eq)                              # its statistics
w_mv = solve(float(mu.min()) * 0.999, long_only=True)                # the minimum-variance weights
ret_mv, vol_mv = portfolio_stats(w_mv)                              # its statistics
print("equal weight: ret %.3f vol %.3f | min var: ret %.3f vol %.3f" % (ret_eq, vol_eq, ret_mv, vol_mv))   # report

# ---------------------------------------------------------------- 3) the factor regression
port_ret = wide.values @ w_eq                                       # the equal-weight portfolio returns
mkt = pd.read_csv(DAT / "l13_market.csv")["mkt_ret"].values          # the market returns
reg = sm.OLS(port_ret[1:], sm.add_constant(mkt[1:])).fit()           # regress on the market
alpha_d = float(reg.params[0]) * 252                                # annualised alpha
beta = float(reg.params[1])                                        # the beta
print("alpha %.4f (annualised) | beta %.3f | R2 %.3f" % (alpha_d, beta, reg.rsquared))   # report
dump(pd.DataFrame([{"portfolio": "equal weight", "alpha_annual": round(alpha_d, 4),   # assemble the frame
                    "beta": round(beta, 4), "r2": round(float(reg.rsquared), 4)}]), "l13_factor.csv")   # persist

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
axes[0].plot(frontier["vol"], frontier["ret"], "-o", ms=3.6, color=OKABE[5], label="有效前沿")   # the frontier
axes[0].scatter([vol_eq], [ret_eq], s=90, color=OKABE[1], zorder=3, label="等权")   # the equal-weight point
axes[0].scatter([vol_mv], [ret_mv], s=90, color=OKABE[2], zorder=3, label="最小方差")   # the min-variance point
axes[0].legend(fontsize=7.6)                                        # compact legend
style_axes(axes[0], "有效前沿：给定风险能拿到的最高收益", "年化波动", "年化收益")   # shared style
axes[1].bar(range(10), np.sort(w_min)[::-1][:10], color=OKABE[3], alpha=0.9)   # the ten largest weights
style_axes(axes[1], "带 30% 上限的最小方差权重（前 10）", "资产（按权重排序）", "权重")   # shared style
annotated(axes[1], "没有上限时，权重会集中到 1-2 只股票", "upper right")   # annotate
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_efficient_frontier.png")                             # save the frontier figure

fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.8))                  # two panels
assets = wide.columns.values                                        # ticker labels
betas = []                                                          # betas per asset
for j in range(n):                                                  # loop over assets
    rj = wide.values[1:, j]                                          # the stock's returns
    bj = sm.OLS(rj, sm.add_constant(mkt[1:])).fit()                   # regress on the market
    betas.append(float(bj.params[1]))                                # record the beta
axes[0].bar(range(n), betas, color=OKABE[4], alpha=0.9)              # the beta histogram as bars
axes[0].axhline(1.0, ls="--", lw=1.0, color="#66717D")               # the market beta
style_axes(axes[0], "每只股票对市场的 beta", "", "beta")               # shared style
axes[0].set_xticks([])                                              # too many labels
axes[1].scatter(mkt[1:], port_ret[1:], s=6, alpha=0.4, color=OKABE[1])   # the scatter
grid = np.linspace(mkt.min(), mkt.max(), 50)                        # a grid for the fitted line
axes[1].plot(grid, reg.params[0] + reg.params[1] * grid, lw=1.8, color=OKABE[5])   # the fitted line
style_axes(axes[1], "组合收益 vs 市场收益：斜率就是 beta=%.2f" % beta, "市场日收益", "组合日收益")   # shared style
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_factor_exposure.png")                                # save the factor figure
print("done: efficient frontier, constraints and factor exposure")     # closing log
