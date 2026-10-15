"""01_datatype_numeric.py -- numeric data (continuous vs discrete): samples and plots."""  # module comment
import numpy as np                                                    # numeric computation
import pandas as pd                                                   # tabular data handling
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, save, dump_csv                               # reuse shared helpers (seed / save / dump)

# ---------------------------------------------------------------- 1) generate samples
n = 800                                                               # sample size: 800 trading days
mu, sigma = 0.0004, 0.018                                             # mean and volatility of daily log returns
ret = RNG.normal(mu, sigma, n)                                        # continuous sample: daily log return, roughly normal
volume = RNG.lognormal(mean=14.0, sigma=0.5, size=n)                  # continuous sample: trading volume, lognormal and right skewed
orders = RNG.poisson(lam=12, size=n)                                  # discrete sample: daily order count, Poisson

df = pd.DataFrame({"day": np.arange(1, n + 1),                       # assemble a table: trading day index
                   "log_return": ret,                                 # continuous variable 1: daily log return
                   "volume": volume,                                  # continuous variable 2: volume, long right tail
                   "orders": orders})                                 # discrete variable: order count, non-negative integer
dump_csv(df, "numeric_samples.csv")                                   # dump the samples into the repo data/ folder

# ---------------------------------------------------------------- 2) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                   # one row, three columns: histogram / boxplot / bar chart
axes[0].hist(ret, bins=40, color="#8A0C3C", alpha=0.85)               # left: histogram of a continuous variable
axes[0].set_title("连续：日对数收益率分布")                            # set the left panel title
axes[0].set_xlabel("收益率")                                          # set the left panel x label
axes[0].set_ylabel("频数")                                            # set the left panel y label
axes[0].axvline(ret.mean(), color="#C9A227", lw=1.6,                 # draw a mean reference line (gold)
                label="均值 %.4f" % ret.mean())                       # show the mean value in the legend
axes[0].legend(fontsize=8)                                            # show the legend

axes[1].boxplot([ret, np.log(volume)], showfliers=True,               # middle: boxplot comparing two continuous variables
                patch_artist=True,                                    # allow a custom box fill colour
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))       # box colour and transparency
axes[1].set_xticklabels(["收益率", "log 成交量"])                       # readable labels for the two boxplots
axes[1].set_title("连续：箱线图（离群点）")                             # set the middle panel title

axes[2].bar(np.arange(1, 26), [np.sum(orders == k) for k in range(1, 26)],  # right: discrete count distribution, 1 to 25 orders
            color="#66717D", width=0.8)                               # grey bars give the frequency of each value
axes[2].set_title("离散：每日订单数（计数）")                           # set the right panel title
axes[2].set_xlabel("订单数（整数取值）")                                # set the right panel x label
plt.tight_layout()                                                    # tighten subplot spacing to avoid overlap
save(fig, "fig_numeric.png")                                          # save the figure into website/figures/
