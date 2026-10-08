"""l4_02_descriptive_stats.py —— 中心趋势、离散程度与分布形状：算出来，也画出来。

脚本先复现原课件"5, 19, 24, 62, 91, 100 → 极差 95"的例子，
再用一份右偏的收入样本对比均值/中位数/众数与方差/MAD/IQR 的稳健性，
并画出偏度与峰度（厚尾/薄尾）的可视化对照。
"""
import numpy as np                                                 # 数值计算与分布构造
import pandas as pd                                                # 统计量与表格
import matplotlib.pyplot as plt                                    # 描述统计图
from scipy import stats                                            # 众数与核密度（仅用于描述统计）
from _common_l4 import RNG, save, dump, style_axes, annotated      # 复用公共工具与图注工具

# ---------------------------------------------------------------- 原课件的数字例子
DEMO = [5, 19, 24, 62, 91, 100]                                    # 原课件给出的六个观测
demo_range = max(DEMO) - min(DEMO)                                 # 极差 = 最大值 − 最小值
print("原课件例子:", DEMO, "→ 极差 =", demo_range, "（应为 95）")     # 打印并自检
assert demo_range == 95, "原课件例子的极差应为 95"                    # 断言：与原课件一致

# ---------------------------------------------------------------- 一份右偏的收入样本
N = 1000                                                           # 样本量
income = RNG.lognormal(mean=2.6, sigma=0.55, size=N)               # 月收入（千元）：典型右偏
outliers = np.array([98.0, 132.5, 210.0])                          # 人为加入三个高收入离群点
income_all = np.concatenate([income, outliers])                     # 合并成"含离群点"的样本
df = pd.DataFrame({"income_k": np.round(income_all, 2)})            # 组装为表
df["is_outlier"] = df["income_k"] > 60                             # 标注高收入点（仅用于教学）
dump(df, "l4_income_skewed.csv")                                   # 落盘样本

def describe_series(s):                                            # 计算描述统计量
    """返回中心趋势、离散程度与形状指标，覆盖课件要求的全部统计量。"""     # 函数约定
    return {                                                       # 逐项计算
        "n": int(s.notna().sum()),                                 # 有效样本量
        "mean": float(s.mean()),                                   # 均值
        "median": float(s.median()),                               # 中位数
        "mode": float(s.round(1).mode().iloc[0]),                  # 众数（按 0.1 分箱取最常见的值）
        "std": float(s.std(ddof=1)),                               # 标准差
        "var": float(s.var(ddof=1)),                               # 方差
        "min": float(s.min()),                                     # 最小值
        "max": float(s.max()),                                     # 最大值
        "range": float(s.max() - s.min()),                         # 极差
        "q1": float(s.quantile(0.25)),                             # 第一四分位
        "q3": float(s.quantile(0.75)),                             # 第三四分位
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),         # 四分位距
        "mad": float((s - s.median()).abs().median()),             # 中位数绝对偏差（稳健离散度）
        "skew": float(s.skew()),                                   # 偏度
        "kurt": float(s.kurt()),                                   # 超额峰度
    }                                                             # describe_series 的返回字典结束
rows = []                                                          # 三个对照组
rows.append({"样本": "无离群点", **describe_series(df.loc[~df["is_outlier"], "income_k"])})   # 纯净样本
rows.append({"样本": "含离群点", **describe_series(df["income_k"])})                          # 全部样本
rows.append({"样本": "原课件例子", **describe_series(pd.Series(DEMO).astype(float))})         # 原课件数据
table = pd.DataFrame(rows)                                         # 汇总成表
table = table.round(3)                                             # 保留三位小数
print(table.to_string(index=False))                                # 终端打印
dump(table, "l4_descriptive_table.csv")                            # 落盘描述统计表

# 稳健性演示：逐个加入离群点，观察均值与中位数的移动幅度
steps = np.arange(0, 11)                                           # 加入 0—10 个离群点
base = df.loc[~df["is_outlier"], "income_k"].values                 # 基础样本
means, medians = [], []                                            # 记录两种中心趋势
for k in steps:                                                    # 逐个场景计算
    sample = np.concatenate([base, np.full(k, 60.0 + 15.0 * k)])    # 每加一个更高的离群点
    means.append(np.mean(sample))                                  # 均值
    medians.append(np.median(sample))                              # 中位数
d_mean = (means[-1] - means[0]) / means[0] * 100                    # 均值的相对移动（%）
d_med = (medians[-1] - medians[0]) / medians[0] * 100               # 中位数的相对移动（%）
print(f"加入 10 个离群点后：均值移动 {d_mean:+.1f}%，中位数移动 {d_med:+.1f}%")   # 打印稳健性结论

# ---------------------------------------------------------------- 图 A：中心趋势与稳健性
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))                # 分布 / 稳健性 / 指标对照
axes[0].hist(df["income_k"], bins=40, color="#8A0C3C", alpha=0.8)   # 收入分布直方图
axes[0].axvline(df["income_k"].mean(), color="#C9A227", lw=1.8,    # 均值线（受离群点拉动）
                label=f"均值 {df['income_k'].mean():.1f}")          # 图例带数值
axes[0].axvline(df["income_k"].median(), color="#2E6F8E", lw=1.8,  # 中位数线（稳健）
                label=f"中位数 {df['income_k'].median():.1f}")      # 图例带数值
style_axes(axes[0], "收入分布：均值 vs 中位数", "月收入（千元）", "人数")   # 统一风格
axes[0].legend()                                                   # 显示图例
axes[1].plot(steps, means, "-o", ms=4, color="#C9A227", label="均值")   # 均值随离群点数量变化
axes[1].plot(steps, medians, "-s", ms=4, color="#2E6F8E", label="中位数")   # 中位数几乎不动
style_axes(axes[1], "离群点对中心趋势的影响", "加入的离群点个数", "月收入（千元）")   # 统一风格
axes[1].legend()                                                   # 图例
annotated(axes[1], f"均值 {d_mean:+.1f}% / 中位数 {d_med:+.1f}%", "upper left")   # 图内注释
metrics = ["mean", "median", "std", "iqr", "mad"]                  # 选取五个常用指标
clean = table.loc[table["样本"] == "无离群点", metrics].iloc[0]      # 纯净样本的取值
dirty = table.loc[table["样本"] == "含离群点", metrics].iloc[0]      # 含离群点的取值
x = np.arange(len(metrics))                                        # 柱状图横坐标
axes[2].bar(x - 0.2, clean.values, width=0.4, label="无离群点", color="#2E6F8E")   # 左侧柱
axes[2].bar(x + 0.2, dirty.values, width=0.4, label="含离群点", color="#8A0C3C")   # 右侧柱
axes[2].set_xticks(x, ["均值", "中位数", "标准差", "IQR", "MAD"])     # 中文指标名
style_axes(axes[2], "稳健性对照：谁被离群点带偏", "指标", "数值")      # 统一风格
axes[2].legend()                                                   # 图例
plt.tight_layout()                                                 # 调整三个子图
save(fig, "fig_central_tendency.png")                              # 保存：中心趋势一页用图

# ---------------------------------------------------------------- 图 B：离散程度与形状
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.7))                # 同均值不同方差 / 离散指标 / 偏度 / 峰度
grid = np.linspace(-4, 4, 400)                                     # 密度曲线的横坐标
for sd, c in zip([0.5, 1.0, 1.8], ["#2E6F8E", "#C9A227", "#8A0C3C"]):   # 三个不同标准差
    axes[0].plot(grid, stats.norm.pdf(grid, 0, sd), lw=1.8, color=c, label=f"σ = {sd}")   # 同均值不同离散
style_axes(axes[0], "同一均值、不同离散程度", "取值", "概率密度")       # 统一风格
axes[0].legend()                                                   # 图例
disp = table.loc[table["样本"] == "含离群点", ["std", "iqr", "mad"]].iloc[0]   # 三个离散指标
axes[1].bar(["标准差", "IQR", "MAD"], disp.values,                 # 离散指标对比
            color=["#8A0C3C", "#C9A227", "#2E6F8E"], alpha=0.9)     # 三色区分
style_axes(axes[1], "三个离散指标（含离群点，千元）", "指标", "数值")    # 统一风格
skew_sets = [("左偏", -4.0, "#2E6F8E"), ("对称", 0.0, "#66717D"), ("右偏", 4.0, "#8A0C3C")]   # 三种偏度
for name, a, c in skew_sets:                                       # 画三条不同偏度的密度
    axes[2].plot(grid, stats.skewnorm.pdf(grid, a), lw=1.8, color=c, label=f"{name}（偏度 {a:+.0f}）")   # 三条不同偏度的密度曲线
style_axes(axes[2], "偏度：尾巴朝向哪边", "取值", "概率密度")          # 统一风格
axes[2].legend(fontsize=8.4)                                       # 图例
kurt_sets = [("厚尾 leptokurtic", 2.0), ("正态 mesokurtic", 0.0), ("薄尾 platykurtic", -0.9)]   # 三种峰度
t = np.linspace(-5, 5, 400)                                        # 峰度对照的横坐标
axes[3].plot(t, stats.norm.pdf(t), lw=1.6, color="#66717D", label="标准正态")   # 参照曲线
axes[3].plot(t, stats.t.pdf(t, 3), lw=1.8, color="#8A0C3C", label="厚尾 t(3)")   # 厚尾：极端值更频繁
axes[3].plot(t, stats.uniform.pdf(t, -1.7, 3.4), lw=1.8, color="#2E6F8E", label="薄尾 均匀")   # 薄尾
style_axes(axes[3], "峰度：尾部有多重", "取值", "概率密度")            # 统一风格
axes[3].legend(fontsize=8.4)                                       # 图例
plt.tight_layout()                                                 # 调整四个子图
save(fig, "fig_dispersion_shape.png")                              # 保存：离散与形状一页用图
