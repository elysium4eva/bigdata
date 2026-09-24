"""l3_06_outliers_iqr_z_iforest.py —— 三种识别异常值的方法，以及三种处置方式。

样本把对数正态的"正常薪资"与三类异常混在一起：两个录入错误、一个真实但极端的值、
一小撮特殊的记录。脚本对比 IQR 规则、Z-score 规则与隔离森林的识别结果，
再展示删除、缩尾与保留对均值/中位数/标准差的影响。
"""
import numpy as np                                                 # 随机抽样与分位数
import pandas as pd                                                # 表格与描述统计
import matplotlib.pyplot as plt                                    # 标记对照图
from sklearn.ensemble import IsolationForest                       # 多变量异常检测
from _common_l3 import RNG, save, dump                             # 复用公共种子、出图与落盘函数

# ---------------------------------------------------------------- 1) 已知组成的样本
n_core, n_err, n_extreme, n_group = 400, 2, 1, 8                    # 四类子样本的规模
core = RNG.lognormal(mean=10.4, sigma=0.45, size=n_core)            # 正常薪资（元）
errors = np.array([-3200.0, 999999.0])                             # 两个录入错误
extreme = np.array([480000.0])                                     # 一个真实但极端的薪资
group = RNG.normal(loc=250000, scale=4000, size=n_group)           # 一小撮特殊记录
values = np.concatenate([core, errors, extreme, group])             # 拼成一列薪资
kind = (["core"] * n_core + ["entry_error"] * n_err                 # 记录真实生成过程
        + ["legitimate_extreme"] * n_extreme + ["unusual_group"] * n_group)   # 仅用于教学对照
df = pd.DataFrame({"salary": np.round(values, 2), "true_kind": kind})   # 分析用的表
dump(df, "l3_income_outliers.csv")                                 # 落盘样本

# ---------------------------------------------------------------- 2) 三种识别规则
q1, q3 = df["salary"].quantile([.25, .75])                          # 观测列的四分位数
iqr = q3 - q1                                                       # 四分位距
df["iqr_outlier"] = (df["salary"] < q1 - 1.5 * iqr) | (df["salary"] > q3 + 1.5 * iqr)   # 箱线规则
z = (df["salary"] - df["salary"].mean()) / df["salary"].std(ddof=0)   # 经典 Z 分数
df["z_outlier"] = z.abs() > 3                                       # |z| > 3 规则
mad = (df["salary"] - df["salary"].median()).abs().median()         # 中位数绝对偏差
robust_z = 0.6745 * (df["salary"] - df["salary"].median()) / mad    # 稳健 Z（基于 MAD）
df["robust_z_outlier"] = robust_z.abs() > 3.5                       # 稳健版的阈值
iso = IsolationForest(contamination=0.03, random_state=42)          # 预期约 3% 是异常
df["iforest_outlier"] = iso.fit_predict(df[["salary"]]) == -1       # -1 表示被判为异常
flags = ["iqr_outlier", "z_outlier", "robust_z_outlier", "iforest_outlier"]   # 四条规则列
print("flagged counts:", {c: int(df[c].sum()) for c in flags})      # 每种规则标出多少行
print("rules agree on all flags:", int((df[flags].sum(axis=1) == len(flags)).sum()))   # 四法一致的数
print("extreme value flagged by IQR/Z:", bool(df.loc[df["true_kind"] == "legitimate_extreme", "iqr_outlier"].iloc[0]))   # 真实极端值
print("entry errors flagged by IQR:", int(df.loc[df["true_kind"] == "entry_error", "iqr_outlier"].sum()))   # 录入错误数

# ---------------------------------------------------------------- 3) 三种处置方式
keep = df["salary"]                                                 # 方案 A：全部保留
drop = df.loc[~df["iqr_outlier"], "salary"]                         # 方案 B：删掉 IQR 异常
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr                       # 缩尾的上下界
wins = df["salary"].clip(lower=lower, upper=upper)                  # 方案 C：缩尾到边界
summary = pd.DataFrame({"treatment": ["keep", "drop (IQR)", "winsorise (IQR)"],   # 三种处置
                        "n": [len(keep), len(drop), len(wins)],     # 保留的样本量
                        "mean": [keep.mean(), drop.mean(), wins.mean()],   # 均值
                        "median": [keep.median(), drop.median(), wins.median()],   # 中位数（稳健）
                        "std": [keep.std(), drop.std(), wins.std()]}).round(1)   # 标准差
print(summary.to_string(index=False))                               # 终端打印取舍对比
dump(summary, "l3_outlier_treatment.csv")                           # 落盘对比表

# ---------------------------------------------------------------- 图：标记与处置
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # 散点 / 箱线 / 规则计数
axes[0].scatter(np.arange(len(df)), df["salary"], s=10,             # 行号对薪资的散点图
                c=np.where(df["iqr_outlier"], "#8A0C3C", "#B9BFC7"))   # 红色为 IQR 标出的行
axes[0].axhline(upper, color="#C9A227", lw=1.2, ls="--", label="IQR fence")   # 上界
axes[0].axhline(lower, color="#C9A227", lw=1.2, ls="--")            # 下界
axes[0].set_yscale("symlog")                                        # symlog 让错误值与极端值都可见
axes[0].set_title("flagged points (IQR rule)")                      # 子图标题
axes[0].set_xlabel("row index"); axes[0].set_ylabel("salary (yuan)")   # 坐标轴含义
axes[0].legend()                                                    # 图例：边界线
axes[1].boxplot([keep, drop, wins], patch_artist=True,              # 三种处置的分布对比
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))     # 箱体填充便于阅读
axes[1].set_xticklabels(["keep", "drop", "winsorise"])              # 横轴标签
axes[1].set_title("distribution after treatment")                   # 子图标题
axes[2].bar([c.replace("_outlier", "") for c in flags],             # 各规则名称
            [int(df[c].sum()) for c in flags], color="#66717D", alpha=0.9)   # 各规则标出的行数
axes[2].set_title("rows flagged per rule")                          # 子图标题
axes[2].tick_params(axis="x", rotation=20)                          # 标签旋转
for i, c in enumerate(flags):                                       # 在柱顶标注具体数值
    axes[2].text(i, int(df[c].sum()), str(int(df[c].sum())), ha="center", va="bottom", fontsize=9)   # 数值标签
plt.tight_layout()                                                  # 调整三个子图
save(fig, "fig_outliers.png")                                       # 保存：异常值一页用图
