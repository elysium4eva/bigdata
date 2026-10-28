"""l13_01_data.py -- the asset panel, the yield curve and two data-quality checks.

Every line carries an English comment. The script builds the daily price panel and the government   # continuation of the call
bond curve, draws the three shapes of the curve, the fat-tailed return distribution and the   # continuation of the call
data-quality panel, and persists the market series that later scripts reuse.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from scipy import stats                                            # QQ plot and kurtosis
from _common_l13 import (make_prices, make_yield_curve, save,       # shared generators
                         dump, style_axes, annotated, OKABE)            # continuation of the call

# ---------------------------------------------------------------- 1) the price panel
prices, market = make_prices(2000, 20)                             # 20 stocks x 2,000 days
dump(prices, "l13_prices.csv")                                     # persist
dump(pd.DataFrame({"day": np.arange(1, len(market) + 1), "mkt_ret": market.round(6)}),   # assemble the frame
     "l13_market.csv")                                             # persist the market series
print("stocks:", prices["ticker"].nunique(), "| days:", prices["date"].nunique())   # report
print("market annualised vol: %.3f" % (market.std() * np.sqrt(252)))   # report

# ---------------------------------------------------------------- 2) the yield curve
curve = make_yield_curve(500)                                      # 500 days
dump(curve, "l13_yield_curve.csv")                                 # persist
tenors = ["1Y", "2Y", "5Y", "10Y", "30Y"]                           # the five tenors
print("10Y range: %.2f-%.2f%%" % (curve["10Y"].min(), curve["10Y"].max()))   # report
spread = curve["10Y"] - curve["1Y"]                                 # the term spread
print("term spread range: %.2f to %.2f" % (spread.min(), spread.max()))   # report

# ---------------------------------------------------------------- 3) the returns of one stock
one = prices[prices["ticker"] == "600519"]["ret"].values             # one stock's returns
kurt = float(stats.kurtosis(one, fisher=False))                     # excess kurtosis
jb = stats.jarque_bera(one)                                         # normality test
print("kurtosis %.2f | Jarque-Bera p %.4f" % (kurt, jb.pvalue))      # report

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].plot(curve["day"], curve["1Y"], lw=1.2, color=OKABE[1], label="1Y")   # short end
axes[0].plot(curve["day"], curve["10Y"], lw=1.4, color=OKABE[5], label="10Y")   # long end
axes[0].plot(curve["day"], curve["30Y"], lw=1.2, color=OKABE[2], label="30Y")   # the long bond
axes[0].legend(fontsize=7.6)                                        # compact legend
style_axes(axes[0], "收益率随时间：中间有一段倒挂", "交易日", "收益率 (%)")   # shared style
axes[1].plot(tenors, curve[tenors].iloc[100], "-o", ms=5, color=OKABE[5], label="正常")   # normal day
axes[1].plot(tenors, curve[tenors].iloc[330], "-s", ms=5, color=OKABE[1], label="倒挂")   # inverted day
style_axes(axes[1], "同一张曲线，两种形状", "期限", "收益率 (%)")        # shared style
axes[1].legend(fontsize=7.6)                                        # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_yield_curve.png")                                   # save the curve figure

fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].hist(one, bins=80, color=OKABE[1], alpha=0.85, density=True)   # the histogram
grid = np.linspace(one.min(), one.max(), 200)                       # a grid for the normal curve
axes[0].plot(grid, stats.norm.pdf(grid, one.mean(), one.std()), lw=1.8, color=OKABE[5], label="正态")   # the normal
axes[0].set_yscale("log")                                          # fat tails need a log scale
axes[0].legend(fontsize=7.6)                                        # compact legend
style_axes(axes[0], "收益率分布：尖峰厚尾（对数纵轴）", "日对数收益率", "密度（对数）")   # shared style
stats.probplot(one, dist="norm", plot=axes[1])                      # the QQ plot
axes[1].get_lines()[0].set_markerfacecolor(OKABE[2])                # colour the points
axes[1].get_lines()[0].set_markersize(2.4)                          # smaller points
axes[1].get_lines()[1].set_color(OKABE[5])                          # colour the line
axes[1].set_title("QQ 图：两端偏离直线 = 厚尾")                       # panel title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_return_dist.png")                                   # save the distribution figure

# ---------------------------------------------------------------- 5) the data-quality panel
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.7))                 # three panels
susp = prices.copy()                                               # a working copy
susp.loc[susp.index[:90], "price"] = np.nan                         # simulate a trading halt
susp.loc[susp.index[200:205], "price"] = susp["price"].iloc[200]     # sticky prices (a data error)
axes[0].plot(susp["price"].iloc[:400], lw=1.0, color=OKABE[4])       # the flawed series
style_axes(axes[0], "停牌与粘滞价：前 90 天缺失", "交易日", "价格")      # shared style
gaps = prices.groupby("ticker")["ret"].apply(lambda s: float((s.abs() > 0.09).mean()))   # jump share
axes[1].bar(range(len(gaps)), gaps.values, color=OKABE[2], alpha=0.9)   # jump share by stock
style_axes(axes[1], "跳空占比（>9% 的日收益）", "股票", "占比")           # shared style
dups = pd.DataFrame({"check": ["缺失", "重复", "跳空", "时区错位"], "share": [0.045, 0.012, 0.031, 0.008]})   # a summary
axes[2].barh(dups["check"], dups["share"], color=OKABE[5], alpha=0.9)   # the checklist
style_axes(axes[2], "四类数据缺陷的典型占比", "占比", "")               # shared style
annotated(axes[2], "不检查就回测 = 自欺欺人", "lower right")            # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_data_quality.png")                                  # save the quality figure
print("done: price panel, yield curve, return distribution and data quality")   # closing log
