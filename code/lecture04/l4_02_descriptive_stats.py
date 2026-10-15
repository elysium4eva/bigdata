"""l4_02_descriptive_stats.py -- center, spread and distribution shape: compute them and draw them.

The script first reproduces the slide example "5, 19, 24, 62, 91, 100 -> range 95",
then compares mean/median/mode and variance/MAD/IQR on a right-skewed income sample,
and plots skewness and kurtosis (heavy vs light tails) side by side.
"""
import numpy as np                                                 # numeric computation and distribution sampling
import pandas as pd                                                # statistics and tables
import matplotlib.pyplot as plt                                    # descriptive statistics figures
from scipy import stats                                            # mode and kernel density (descriptive use only)
from _common_l4 import RNG, save, dump, style_axes, annotated      # reuse shared helpers and the annotation helper

# ---------------------------------------------------------------- the slide's numeric example
DEMO = [5, 19, 24, 62, 91, 100]                                    # the six observations from the original slides
demo_range = max(DEMO) - min(DEMO)                                 # range = maximum minus minimum
print("原课件例子:", DEMO, "→ 极差 =", demo_range, "（应为 95）")     # print and self-check
assert demo_range == 95, "原课件例子的极差应为 95"                    # assert it matches the slides

# ---------------------------------------------------------------- a right-skewed income sample
N = 1000                                                           # sample size
income = RNG.lognormal(mean=2.6, sigma=0.55, size=N)               # monthly income (thousand CNY): typically right-skewed
outliers = np.array([98.0, 132.5, 210.0])                          # add three artificial high-income outliers
income_all = np.concatenate([income, outliers])                     # merge into a sample "with outliers"
df = pd.DataFrame({"income_k": np.round(income_all, 2)})            # assemble into a table
df["is_outlier"] = df["income_k"] > 60                             # flag high incomes (teaching only)
dump(df, "l4_income_skewed.csv")                                   # write the sample

def describe_series(s):                                            # compute the descriptive statistics
    """Return central tendency, spread and shape metrics for the slides."""   # function contract
    return {                                                       # compute each item
        "n": int(s.notna().sum()),                                 # valid sample size
        "mean": float(s.mean()),                                   # mean
        "median": float(s.median()),                               # median
        "mode": float(s.round(1).mode().iloc[0]),                  # mode (most common value after rounding to 0.1)
        "std": float(s.std(ddof=1)),                               # standard deviation
        "var": float(s.var(ddof=1)),                               # variance
        "min": float(s.min()),                                     # minimum
        "max": float(s.max()),                                     # maximum
        "range": float(s.max() - s.min()),                         # range
        "q1": float(s.quantile(0.25)),                             # first quartile
        "q3": float(s.quantile(0.75)),                             # third quartile
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),         # interquartile range
        "mad": float((s - s.median()).abs().median()),             # median absolute deviation (robust spread)
        "skew": float(s.skew()),                                   # skewness
        "kurt": float(s.kurt()),                                   # excess kurtosis
    }                                                             # end of the describe_series return dict
rows = []                                                          # three comparison groups
rows.append({"样本": "无离群点", **describe_series(df.loc[~df["is_outlier"], "income_k"])})   # clean sample
rows.append({"样本": "含离群点", **describe_series(df["income_k"])})                          # all observations
rows.append({"样本": "原课件例子", **describe_series(pd.Series(DEMO).astype(float))})         # original slide data
table = pd.DataFrame(rows)                                         # collect into a table
table = table.round(3)                                             # keep three decimals
print(table.to_string(index=False))                                # print to the terminal
dump(table, "l4_descriptive_table.csv")                            # write the descriptive table

# robustness demo: add outliers one by one and watch mean and median move
steps = np.arange(0, 11)                                           # add 0 to 10 outliers
base = df.loc[~df["is_outlier"], "income_k"].values                 # the base sample
means, medians = [], []                                            # record the two measures of center
for k in steps:                                                    # compute each scenario
    sample = np.concatenate([base, np.full(k, 60.0 + 15.0 * k)])    # each added outlier is higher than the last
    means.append(np.mean(sample))                                  # mean
    medians.append(np.median(sample))                              # median
d_mean = (means[-1] - means[0]) / means[0] * 100                    # relative move of the mean (%)
d_med = (medians[-1] - medians[0]) / medians[0] * 100               # relative move of the median (%)
print(f"加入 10 个离群点后：均值移动 {d_mean:+.1f}%，中位数移动 {d_med:+.1f}%")   # print the robustness conclusion

# ---------------------------------------------------------------- figure A: center and robustness
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))                # distribution / robustness / metric comparison
axes[0].hist(df["income_k"], bins=40, color="#8A0C3C", alpha=0.8)   # income distribution histogram
axes[0].axvline(df["income_k"].mean(), color="#C9A227", lw=1.8,    # mean line (pulled by outliers)
                label=f"均值 {df['income_k'].mean():.1f}")          # legend with the value
axes[0].axvline(df["income_k"].median(), color="#2E6F8E", lw=1.8,  # median line (robust)
                label=f"中位数 {df['income_k'].median():.1f}")      # legend with the value
style_axes(axes[0], "收入分布：均值 vs 中位数", "月收入（千元）", "人数")   # shared style
axes[0].legend()                                                   # show the legend
axes[1].plot(steps, means, "-o", ms=4, color="#C9A227", label="均值")   # mean moves with the number of outliers
axes[1].plot(steps, medians, "-s", ms=4, color="#2E6F8E", label="中位数")   # the median barely moves
style_axes(axes[1], "离群点对中心趋势的影响", "加入的离群点个数", "月收入（千元）")   # shared style
axes[1].legend()                                                   # legend
annotated(axes[1], f"均值 {d_mean:+.1f}% / 中位数 {d_med:+.1f}%", "upper left")   # in-figure note
metrics = ["mean", "median", "std", "iqr", "mad"]                  # pick five common metrics
clean = table.loc[table["样本"] == "无离群点", metrics].iloc[0]      # values for the clean sample
dirty = table.loc[table["样本"] == "含离群点", metrics].iloc[0]      # values with outliers
x = np.arange(len(metrics))                                        # bar chart x coordinates
axes[2].bar(x - 0.2, clean.values, width=0.4, label="无离群点", color="#2E6F8E")   # left bars
axes[2].bar(x + 0.2, dirty.values, width=0.4, label="含离群点", color="#8A0C3C")   # right bars
axes[2].set_xticks(x, ["均值", "中位数", "标准差", "IQR", "MAD"])     # Chinese metric names
style_axes(axes[2], "稳健性对照：谁被离群点带偏", "指标", "数值")      # shared style
axes[2].legend()                                                   # legend
plt.tight_layout()                                                 # adjust the three subplots
save(fig, "fig_central_tendency.png")                              # save: one-page central tendency figure

# ---------------------------------------------------------------- figure B: spread and shape
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.7))                # same mean different spread / spread metrics / skew / kurtosis
grid = np.linspace(-4, 4, 400)                                     # x coordinates of the density curves
for sd, c in zip([0.5, 1.0, 1.8], ["#2E6F8E", "#C9A227", "#8A0C3C"]):   # three different standard deviations
    axes[0].plot(grid, stats.norm.pdf(grid, 0, sd), lw=1.8, color=c, label=f"σ = {sd}")   # same mean, different spread
style_axes(axes[0], "同一均值、不同离散程度", "取值", "概率密度")       # shared style
axes[0].legend()                                                   # legend
disp = table.loc[table["样本"] == "含离群点", ["std", "iqr", "mad"]].iloc[0]   # three spread metrics
axes[1].bar(["标准差", "IQR", "MAD"], disp.values,                 # spread metric comparison
            color=["#8A0C3C", "#C9A227", "#2E6F8E"], alpha=0.9)     # three distinct colors
style_axes(axes[1], "三个离散指标（含离群点，千元）", "指标", "数值")    # shared style
skew_sets = [("左偏", -4.0, "#2E6F8E"), ("对称", 0.0, "#66717D"), ("右偏", 4.0, "#8A0C3C")]   # three skewness cases
for name, a, c in skew_sets:                                       # draw three densities with different skew
    axes[2].plot(grid, stats.skewnorm.pdf(grid, a), lw=1.8, color=c, label=f"{name}（偏度 {a:+.0f}）")   # three density curves with different skew
style_axes(axes[2], "偏度：尾巴朝向哪边", "取值", "概率密度")          # shared style
axes[2].legend(fontsize=8.4)                                       # legend
kurt_sets = [("厚尾 leptokurtic", 2.0), ("正态 mesokurtic", 0.0), ("薄尾 platykurtic", -0.9)]   # three kurtosis cases
t = np.linspace(-5, 5, 400)                                        # x coordinates for the kurtosis panel
axes[3].plot(t, stats.norm.pdf(t), lw=1.6, color="#66717D", label="标准正态")   # reference curve
axes[3].plot(t, stats.t.pdf(t, 3), lw=1.8, color="#8A0C3C", label="厚尾 t(3)")   # heavy tail: extremes are more frequent
axes[3].plot(t, stats.uniform.pdf(t, -1.7, 3.4), lw=1.8, color="#2E6F8E", label="薄尾 均匀")   # light tail
style_axes(axes[3], "峰度：尾部有多重", "取值", "概率密度")            # shared style
axes[3].legend(fontsize=8.4)                                       # legend
plt.tight_layout()                                                 # adjust the four subplots
save(fig, "fig_dispersion_shape.png")                              # save: one-page spread and shape figure
