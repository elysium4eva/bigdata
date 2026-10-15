"""l4_10_uncertainty.py -- how to draw uncertainty: error bars, bootstrap intervals, bands and fan forecasts.

Not drawing uncertainty is itself misleading. The script bootstraps the same A/B data to estimate
intervals for conversion and order value, shows four presentations, and stores them in a table.
"""
import numpy as np                                                 # numeric computation and resampling
import pandas as pd                                                # data processing
import matplotlib.pyplot as plt                                    # four-panel figure
from _common_l4 import RNG, DAT, save, dump                        # reuse shared helpers

ab = pd.read_csv(DAT / "l4_ab_test.csv")                           # read the A/B experiment data (from l4_09)
B = 2000                                                           # number of bootstrap repetitions
def bootstrap_ci(values, stat=np.mean, level=0.95, boot=B):         # generic bootstrap interval function
    """Resample the sample and return the point estimate and percentile interval."""   # function contract
    values = np.asarray(values, dtype=float)                       # convert to a float array
    idx = RNG.integers(0, len(values), size=(boot, len(values)))    # draw the same number of indices with replacement
    stats_ = stat(values[idx], axis=1)                             # compute the statistic for every resample
    lo, hi = np.percentile(stats_, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])   # percentile interval
    return float(stat(values)), float(lo), float(hi), stats_       # return point estimate, interval and distribution

rows = []                                                          # summary table
boot_store = {}                                                    # keep the bootstrap distributions for plotting
for g in ["对照组 A", "实验组 B"]:                                  # compute group by group
    sub = ab[ab["group"] == g]                                     # take that group's sample
    m, lo, hi, dist = bootstrap_ci(sub["converted"])               # bootstrap interval for the conversion rate
    rows.append({"指标": f"{g}·转化率", "点估计": round(m, 4),       # record the point estimate
                 "CI下界": round(lo, 4), "CI上界": round(hi, 4),    # record the confidence interval
                 "样本量": int(len(sub))})                          # record the sample size
    boot_store[g] = (m, lo, hi, dist)                              # stash the bootstrap distribution
    rev = sub.loc[sub["converted"], "revenue"]                     # order value of converters only
    m2, lo2, hi2, dist2 = bootstrap_ci(rev)                        # bootstrap interval for the order value
    rows.append({"指标": f"{g}·客单价", "点估计": round(m2, 2),      # record the order value point estimate
                 "CI下界": round(lo2, 2), "CI上界": round(hi2, 2),  # order value interval
                 "样本量": int(len(rev))})                          # order value sample size
    boot_store[g + "·客单价"] = (m2, lo2, hi2, dist2)               # stash it
diff_dist = boot_store["实验组 B"][3] - boot_store["对照组 A"][3]     # bootstrap distribution of the group difference
d_point, d_lo, d_hi = float(diff_dist.mean()), float(np.percentile(diff_dist, 2.5)), float(np.percentile(diff_dist, 97.5))   # mean and 95% interval of the difference
rows.append({"指标": "转化率差值（B−A）", "点估计": round(d_point, 4),   # difference point estimate
             "CI下界": round(d_lo, 4), "CI上界": round(d_hi, 4),     # difference interval
             "样本量": int(len(ab))})                              # total sample size
unc = pd.DataFrame(rows)                                           # collect into a table
dump(unc, "l4_uncertainty_summary.csv")                            # write the uncertainty summary
print(unc.to_string(index=False))                                  # print to the terminal
crosses_zero = d_lo < 0 < d_hi                                     # does the difference interval cross 0
print(f"差值 95% CI = [{d_lo:+.4f}, {d_hi:+.4f}]，跨 0：{crosses_zero}")   # print the conclusion hint

# ---------------------------------------------------------------- figure: four views of uncertainty
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.7))                # error bars / bootstrap distribution / confidence band / fan forecast
mA, loA, hiA, _ = boot_store["对照组 A"]                            # group A conversion rate
mB, loB, hiB, _ = boot_store["实验组 B"]                            # group B conversion rate
axes[0].errorbar([0, 1], [mA * 100, mB * 100],                      # point estimate plus error bars
                 yerr=[[(mA - loA) * 100, (mB - loB) * 100],        # lower error
                       [(hiA - mA) * 100, (hiB - mB) * 100]],       # upper error
                 fmt="o", capsize=5, color="#8A0C3C", ms=7)         # marker and caps
axes[0].set_xticks([0, 1], ["A 组", "B 组"])                        # group labels
axes[0].set_ylim(0, max(hiA, hiB) * 100 * 1.25)                    # leave room
axes[0].set_title("① 点估计 + 95% 误差棒")                          # subplot title
axes[0].set_ylabel("转化率 %")                                      # axis label
axes[1].hist(diff_dist * 100, bins=40, color="#2E6F8E", alpha=0.85)   # (2) bootstrap distribution of the difference
axes[1].axvline(0, color="#1E2630", lw=1.4, ls="--")                # zero line
axes[1].axvline(d_lo * 100, color="#C9A227", lw=1.2)                # interval lower bound
axes[1].axvline(d_hi * 100, color="#C9A227", lw=1.2, label="95% 区间")   # interval upper bound
axes[1].set_title("② 自助法：差值的分布")                            # subplot title
axes[1].set_xlabel("转化率差值（百分点）"); axes[1].legend(fontsize=8.4)   # axis label and legend
daily = ab.assign(day=RNG.integers(1, 21, len(ab))).groupby(["day", "group"])["converted"].mean().unstack()   # conversion rate by day (for the confidence band)
band_lo, band_hi = [], []                                          # (3) per-day bootstrap band
for d in daily.index:                                              # compute the interval day by day
    v = daily.loc[d, "实验组 B"]                                   # group B conversion rate that day
    m, lo, hi, _ = bootstrap_ci(ab.loc[(ab["group"] == "实验组 B"), "converted"], level=0.8)   # estimate the interval from the pooled sample
    band_lo.append(max(lo * 100, 0)); band_hi.append(hi * 100)     # record the bounds (percent)
axes[2].plot(daily.index, daily["实验组 B"] * 100, "-o", ms=3, color="#8A0C3C", lw=1.4)   # line
axes[2].fill_between(daily.index, band_lo, band_hi, color="#8A0C3C", alpha=0.18, label="80% 区间")   # confidence band
axes[2].set_title("③ 时间序列 + 置信带")                            # subplot title
axes[2].set_xlabel("实验天数"); axes[2].set_ylabel("转化率 %")        # axis labels
axes[2].legend(fontsize=8.4)                                       # legend
months = np.arange(1, 25)                                          # (4) forecast for the next 24 periods
center = 12.0 + 0.18 * months                                      # central forecast (slowly rising)
width = 0.9 + 0.42 * months                                        # uncertainty grows with the horizon
axes[3].plot(months, center, color="#8A0C3C", lw=1.8, label="中心预测")   # center line
axes[3].fill_between(months, center - width, center + width, color="#C9A227", alpha=0.25, label="80% 预测区间")   # 80% band
axes[3].fill_between(months, center - 2 * width, center + 2 * width, color="#C9A227", alpha=0.15, label="95% 预测区间")   # 95% band
axes[3].set_title("④ 扇形预测：越远越不确定")                        # subplot title
axes[3].set_xlabel("未来第 n 期"); axes[3].set_ylabel("指标值")      # axis labels
axes[3].legend(fontsize=8.2)                                       # legend
plt.tight_layout()                                                 # adjust the four panels
save(fig, "fig_uncertainty.png")                                   # save: four views of uncertainty
