"""l13_07_backtest.py -- a deliberately honest backtester and its performance report.

Every line carries an English comment. The script builds a simple momentum signal, runs it through   # continuation of the call
a vectorised backtest with trading costs and a one-day execution lag, draws the equity curve with   # continuation of the call
drawdowns, and computes Sharpe, Sortino and Calmar so the deck can show what "honest" means.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l13 import (DAT, save, dump, style_axes, annotated,   # shared helpers
                         OKABE)                                         # continuation of the call

# ---------------------------------------------------------------- 1) the panel
prices = pd.read_csv(DAT / "l13_prices.csv", dtype={"ticker": str})   # the price panel
wide = prices.pivot(index="date", columns="ticker", values="ret")     # returns in wide form
print("dates:", wide.shape[0], "| assets:", wide.shape[1])            # report

# ---------------------------------------------------------------- 2) the signal: 20-day momentum
lookback = 20                                                      # the momentum window
signal = wide.rolling(lookback).mean()                              # a simple momentum proxy
signal = signal.shift(1)                                           # trade on yesterday's signal only
weights = signal.div(signal.abs().sum(axis=1), axis=0).fillna(0.0)   # long-short, gross 1
weights = weights.clip(-0.2, 0.2)                                  # a position cap
turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)   # daily turnover
cost_bps = 8.0                                                      # round-trip cost in basis points
gross = (weights * wide).sum(axis=1)                                 # gross portfolio return
net = gross - turnover * cost_bps / 1e4                             # net of costs
print("mean gross %.4f%% | mean net %.4f%% | mean turnover %.3f" % (    # report the result
    100 * gross.mean(), 100 * net.mean(), turnover.mean()))          # report

# ---------------------------------------------------------------- 3) performance statistics
def performance(series, rf=0.0):                                   # the standard ratios
    """Return annualised return, volatility, Sharpe, Sortino and max drawdown."""   # short docstring
    ann_ret = float(series.mean() * 252)                            # annualised mean
    ann_vol = float(series.std() * np.sqrt(252))                    # annualised volatility
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan      # the Sharpe ratio
    downside = series[series < 0].std() * np.sqrt(252)               # downside deviation
    sortino = ann_ret / downside if downside > 0 else np.nan          # the Sortino ratio
    equity = (1 + series).cumprod()                                  # the equity curve
    drawdown = equity / equity.cummax() - 1                          # the drawdown path
    mdd = float(drawdown.min())                                      # the maximum drawdown
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan                # the Calmar ratio
    return {"ann_ret": round(ann_ret, 4), "ann_vol": round(ann_vol, 4),   # the record
            "sharpe": round(float(sharpe), 3), "sortino": round(float(sortino), 3),   # more ratios
            "max_drawdown": round(mdd, 4), "calmar": round(float(calmar), 3)}   # the rest

stats = performance(net)                                            # the net statistics
gross_stats = performance(gross)                                    # the gross statistics
print("gross Sharpe %.2f | net Sharpe %.2f" % (gross_stats["sharpe"], stats["sharpe"]))   # report
table = pd.DataFrame([{"version": "含成本", **stats}, {"version": "不含成本", **gross_stats}])   # assemble
dump(table, "l13_backtest_stats.csv")                              # persist
dump(pd.DataFrame({"day": np.arange(1, len(net) + 1), "net": net.round(6),   # assemble the frame
                   "gross": gross.round(6), "turnover": turnover.round(4)}), "l13_backtest_returns.csv")   # persist

# ---------------------------------------------------------------- 4) the figure
equity_net = (1 + net).cumprod()                                    # net equity
equity_gross = (1 + gross).cumprod()                                # gross equity
drawdown = equity_net / equity_net.cummax() - 1                      # the drawdown path
fig, axes = plt.subplots(2, 1, figsize=(12.0, 5.2), sharex=True,      # two stacked panels
                         gridspec_kw={"height_ratios": [3, 1]})       # give the equity more room
axes[0].plot(equity_gross.values if hasattr(equity_gross, "values") else equity_gross,   # draw the panel
             lw=1.4, color=OKABE[2], label="不含成本")                 # the gross curve
axes[0].plot(equity_net.values if hasattr(equity_net, "values") else equity_net,   # draw the panel
             lw=1.6, color=OKABE[5], label="含成本")                   # the net curve
axes[0].legend(fontsize=7.6)                                        # compact legend
style_axes(axes[0], "回测净值：8bp 的成本吃掉多少", "交易日", "净值")      # shared style
annotated(axes[0], "Sharpe %.2f → %.2f" % (gross_stats["sharpe"], stats["sharpe"]), "upper left")   # annotate
axes[1].fill_between(range(len(drawdown)), drawdown.values, 0, color=OKABE[1], alpha=0.55)   # the drawdown
style_axes(axes[1], "回撤路径（最大回撤 %.1f%%）" % (100 * stats["max_drawdown"]), "交易日", "回撤")   # shared style
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_equity_curve.png")                                   # save the figure
print("done: an honest backtest with costs and drawdowns")             # closing log
