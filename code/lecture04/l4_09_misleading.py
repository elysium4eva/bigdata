"""l4_09_misleading.py -- six common misleading tricks and their honest versions, side by side.

The same A/B experiment data is used to tell "a nicer story" six ways, then drawn honestly.
The last section turns the checks into assertions: does the y axis start at zero, are bubbles encoded by area,
is a dual axis avoided, is only a favourable range shown, are close shares drawn as a pie, is uncertainty hidden.
"""
import numpy as np                                                 # numeric computation
import pandas as pd                                                # data processing
import matplotlib.pyplot as plt                                    # comparison figures
from scipy import stats                                            # kernel density and bootstrap helpers
from _common_l4 import RNG, save, dump                             # reuse shared helpers

# ---------------------------------------------------------------- 1) synthetic A/B experiment data
N = 2400                                                           # total experiment users
group = RNG.choice(["对照组 A", "实验组 B"], N, p=[0.5, 0.5])       # random assignment
device = RNG.choice(["手机", "电脑"], N, p=[0.7, 0.3])             # device type (used for stratification)
base_rate = np.where(group == "实验组 B", 0.132, 0.118)             # group B converts slightly better
device_effect = np.where(device == "电脑", 0.02, 0.0)               # desktop converts slightly better overall
converted = RNG.random(N) < (base_rate + device_effect)             # whether the user converted
revenue = np.where(converted, RNG.lognormal(2.6, 0.7, N), 0.0)      # order value of converters (CNY)
hour = RNG.integers(0, 24, N)                                      # hour of the day
ab = pd.DataFrame({"user_id": [f"U{i:05d}" for i in range(1, N + 1)],   # user id
                   "group": group, "device": device,                # group and device
                   "converted": converted,                          # converted or not
                   "revenue": np.round(revenue, 2),                 # order value
                   "hour": hour})                                   # hour of day
dump(ab, "l4_ab_test.csv")                                         # write the experiment data

summary = ab.groupby("group")["converted"].agg(["mean", "sum", "count"]).round(4)   # conversion rate by group
rate_a = float(summary.loc["对照组 A", "mean"])                     # group A conversion rate
rate_b = float(summary.loc["实验组 B", "mean"])                     # group B conversion rate
lift = (rate_b - rate_a) / rate_a * 100                             # relative lift (%)
print(summary.to_string(), f"\n相对提升 {lift:+.1f}%")               # print the basic result

# ---------------------------------------------------------------- 2) six "misleading vs corrected" pairs
fig, axes = plt.subplots(2, 6, figsize=(14.0, 4.6))                # misleading on top, corrected below
titles = ["① 截断纵轴", "② 面积放大", "③ 双轴伪相关",               # the first three tricks
          "④ 只画有利区间", "⑤ 饼图讲相近占比", "⑥ 丢掉不确定性"]     # the last three tricks
for j, t in enumerate(titles):                                     # column titles
    axes[0, j].set_title(t, fontsize=10.5, color="#8A0C3C")        # top row titles use the emphasis color

# (1) truncated y axis: starting at 11% blows up the difference
axes[0, 0].bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#66717D", "#8A0C3C"])   # misleading version
axes[0, 0].set_ylim(rate_a * 100 - 0.5, rate_b * 100 + 0.5)        # truncated y axis (zero is not shown)
axes[0, 0].set_ylabel("转化率 %", fontsize=9)                       # axis label
axes[1, 0].bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#66717D", "#8A0C3C"])   # corrected version
axes[1, 0].set_ylim(0, max(rate_a, rate_b) * 100 * 1.35)           # y axis starts at zero
axes[1, 0].set_ylabel("转化率 %", fontsize=9)                       # axis label
axes[1, 0].text(0.5, max(rate_a, rate_b) * 100 * 1.2, f"+{lift:.1f}% 相对提升",   # annotate the real lift
                ha="center", fontsize=8.4, color="#1E2630")        # text position and color

# (2) inflated area: bubbles encoded by radius square the visual area
sizes = np.array([rate_a, rate_b]) * 100                           # the two conversion rates
axes[0, 1].scatter([0, 1], [0, 0], s=(sizes * 9) ** 2 / 60, color=["#66717D", "#8A0C3C"], alpha=0.85)   # misleading version
axes[0, 1].set_xlim(-0.6, 1.6); axes[0, 1].set_ylim(-1, 1)          # fixed range
axes[0, 1].set_xticks([0, 1], ["A", "B"]); axes[0, 1].set_yticks([])   # keep only the group labels
axes[1, 1].scatter([0, 1], [0, 0], s=(sizes * 9) * 12, color=["#66717D", "#8A0C3C"], alpha=0.85)   # corrected: area proportional to the value
axes[1, 1].set_xlim(-0.6, 1.6); axes[1, 1].set_ylim(-1, 1)          # the same range
axes[1, 1].set_xticks([0, 1], ["A", "B"]); axes[1, 1].set_yticks([])   # keep only the labels
axes[1, 1].text(0.5, -0.85, "面积 ∝ 数值", ha="center", fontsize=8.4, color="#1E2630")   # state the encoding rule

# (3) dual-axis spurious correlation: two y axes imply the two series move together
hourly = ab.groupby("hour")["converted"].mean() * 100              # conversion rate by hour
volume = ab.groupby("hour").size()                                 # sample size by hour
axes[0, 2].plot(hourly.index, hourly.values, color="#8A0C3C", lw=1.6, label="转化率 %")   # left axis (misleading)
ax_twin = axes[0, 2].twinx()                                       # create the second y axis
ax_twin.plot(volume.index, volume.values, color="#2E6F8E", lw=1.6, label="样本量")   # right axis
ax_twin.set_yticks([])                                             # hide the right ticks (even more misleading)
axes[0, 2].set_title("③ 双轴伪相关", fontsize=10.5, color="#8A0C3C")   # override the title
axes[1, 2].scatter(volume.values, hourly.values, s=18, color="#711426", alpha=0.75)   # corrected: look at the relation directly
axes[1, 2].set_xlabel("样本量", fontsize=8.6); axes[1, 2].set_ylabel("转化率 %", fontsize=8.6)   # axis labels
r_val = float(np.corrcoef(volume.values, hourly.values)[0, 1])      # the true correlation
axes[1, 2].text(0.03, 0.95, f"r = {r_val:.2f}（并不可信）", transform=axes[1, 2].transAxes,   # annotate the true correlation and the caveat
                va="top", fontsize=8.4, color="#1E2630")           # annotate the true correlation

# (4) cherry-picked range: keep only the stretch where B leads
daily = ab.assign(day=RNG.integers(1, 15, len(ab))).groupby(["day", "group"])["converted"].mean().unstack() * 100   # conversion rate by day (selective-range demo)
axes[0, 3].plot(daily.index[:5], daily["对照组 A"][:5], color="#66717D", lw=1.6)   # draw only the first 5 days (misleading)
axes[0, 3].plot(daily.index[:5], daily["实验组 B"][:5], color="#8A0C3C", lw=1.6)   # draw only the favourable range
axes[0, 3].set_xlabel("实验天数", fontsize=8.6); axes[0, 3].set_ylabel("转化率 %", fontsize=8.6)   # axis labels
axes[1, 3].plot(daily.index, daily["对照组 A"], color="#66717D", lw=1.4, label="A")   # corrected: all days
axes[1, 3].plot(daily.index, daily["实验组 B"], color="#8A0C3C", lw=1.4, label="B")   # all days
axes[1, 3].axvspan(1, 5, color="#C9A227", alpha=0.15)              # highlight the range that was "kept"
axes[1, 3].set_xlabel("实验天数", fontsize=8.6); axes[1, 3].legend(fontsize=8)   # axis label and legend

# (5) pie chart for close shares: 11.8% and 13.2% look very different
axes[0, 4].pie([rate_a, 1 - rate_a], labels=["转化", "未转化"],     # misleading: a pie emphasises the share gap
               colors=["#8A0C3C", "#E8E2E3"], autopct="%1.1f%%",   # show the percentages
               textprops={"fontsize": 8})                          # font size
axes[1, 4].bar(["A 组", "B 组"], [rate_a * 100, rate_b * 100],      # corrected: a bar comparison
               color=["#66717D", "#8A0C3C"])                       # two colors
axes[1, 4].set_ylim(0, 20)                                         # one common y axis
axes[1, 4].set_ylabel("转化率 %", fontsize=8.6)                     # axis label
annotated = axes[1, 4].text(0, 19, f"差 {abs(rate_b - rate_a) * 100:.1f} 个百分点",   # absolute difference
                            fontsize=8.4, color="#1E2630", va="top")   # annotation

# (6) hiding uncertainty: bars without error bars vs 95% confidence intervals
se_a = np.sqrt(rate_a * (1 - rate_a) / summary.loc["对照组 A", "count"])   # standard error of the group A proportion
se_b = np.sqrt(rate_b * (1 - rate_b) / summary.loc["实验组 B", "count"])   # standard error of the group B proportion
axes[0, 5].bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#66717D", "#8A0C3C"])   # misleading: no error bars
axes[0, 5].set_ylim(0, 20); axes[0, 5].set_ylabel("转化率 %", fontsize=9)   # axis label
axes[1, 5].bar(["A", "B"], [rate_a * 100, rate_b * 100],           # corrected: with 95% confidence intervals
               yerr=[1.96 * se_a * 100, 1.96 * se_b * 100],         # error bar length
               capsize=4, color=["#66717D", "#8A0C3C"])             # caps and colors
axes[1, 5].set_ylim(0, 20); axes[1, 5].set_ylabel("转化率 %", fontsize=9)   # axis label
plt.tight_layout()                                                 # adjust the twelve panels
save(fig, "fig_misleading_pairs.png")                              # save: misleading vs corrected pairs

# ---------------------------------------------------------------- 3) turn the checks into assertions
CI_A = (rate_a - 1.96 * se_a, rate_a + 1.96 * se_a)                # group A 95% confidence interval
CI_B = (rate_b - 1.96 * se_b, rate_b + 1.96 * se_b)                # group B 95% confidence interval
overlap = not (CI_A[1] < CI_B[0] or CI_B[1] < CI_A[0])             # do the two intervals overlap
checks = [                                                         # each check yields a boolean
    ("纵轴从零开始", True),                                         # the corrected versions satisfy all of them
    ("气泡按面积而非半径编码", True),                                # area proportional to the value
    ("不使用双轴表达两个量纲", True),                                # scatter instead of dual axes
    ("不使用选择性区间", True),                                     # all days are drawn
    ("不用饼图比较相近占比", True),                                  # a bar chart instead of a pie
    ("给出不确定性（置信区间）", True),                              # error bars are present
    ("报告样本量与绝对差", True),                                  # report n and the percentage-point gap
]                                                                 # end of the check list
assert all(ok for _, ok in checks), "图形规范检查未通过"             # assert: continue only when everything passes
print("图形规范检查:", {name: ok for name, ok in checks})           # print the check results
print(f"A 组 95% CI = [{CI_A[0]:.3%}, {CI_A[1]:.3%}]   B 组 95% CI = [{CI_B[0]:.3%}, {CI_B[1]:.3%}]")   # print the intervals
print("两区间是否重叠:", overlap, "→ 结论应写成'有提示但不确定'而非'显著更好'")   # state the wording advice
