"""l3_08_imputation_compare.py —— 五种插补策略，用已知的真值给它们打分。

先在 MAR 机制下删掉一部分收入，再用均值、中位数、回归、KNN 与 MICE（迭代插补）五种方式补回来。
因为完整数据已知，脚本可以从分布、相关性到下游回归系数逐项评估每种策略——
这就是"插补有没有把事情搞坏"的实操检验。
"""
import numpy as np                                                 # 线性代数与描述统计
import pandas as pd                                                # 表格与缺失处理
import matplotlib.pyplot as plt                                    # 分布对照图
from sklearn.experimental import enable_iterative_imputer           # 先显式启用实验性的 MICE 插补器
from sklearn.impute import KNNImputer, IterativeImputer            # KNN 与 MICE 两种插补
from sklearn.linear_model import LinearRegression                  # 回归插补与下游模型
from _common_l3 import RNG, save, dump                             # 复用公共种子、出图与落盘函数

# ---------------------------------------------------------------- 1) 先造完整数据，再破坏它
N = 700                                                            # 受调查的员工数
age = RNG.integers(23, 58, N)                                      # 年龄（岁）
hours = np.round(RNG.normal(42, 6, N).clip(20, 65), 1)             # 每周工作时长
edu = RNG.choice([0, 1, 2], N, p=[0.4, 0.42, 0.18])                # 学历（用 0/1/2 表示）
income = (12 + 0.30 * hours + 0.55 * edu + 0.06 * age              # 真实的数据生成过程
          + RNG.normal(0, 2.4, N))                                 # 再加上个体噪声
complete = pd.DataFrame({"age": age, "hours": hours, "edu": edu,   # 协变量
                         "income": np.round(income, 2)})           # 将被"弄缺"的收入
dump(complete, "l3_income_complete.csv")                           # 落盘完整参照表
truth_mean, truth_sd = complete["income"].mean(), complete["income"].std()   # 参照均值与标准差
truth_corr = complete["income"].corr(complete["hours"])            # 参照相关系数
coef_truth = LinearRegression().fit(complete[["hours"]], complete["income"]).coef_[0]   # 参照回归系数
print(f"truth: mean={truth_mean:.2f} sd={truth_sd:.2f} corr(hours)={truth_corr:.3f} slope={coef_truth:.3f}")   # 打印基准

# ---------------------------------------------------------------- 2) 按 MAR 机制删掉收入
p_missing = np.where(edu == 2, 0.55, np.where(edu == 1, 0.30, 0.10))   # 学历越高越不愿披露
mask = RNG.random(N) < p_missing                                   # 缺失掩码
damaged = complete.copy()                                          # 在副本上动手
damaged.loc[mask, "income"] = np.nan                               # 收入变成缺失
dump(damaged, "l3_income_missing.csv")                             # 落盘被破坏的表
print("missing rate in income:", round(100 * damaged['income'].isna().mean(), 1), "%")   # 打印缺失率

# ---------------------------------------------------------------- 3) 五种补法
filled = {"mean": damaged["income"].fillna(damaged["income"].mean()),   # 全局均值插补
          "median": damaged["income"].fillna(damaged["income"].median()),   # 全局中位数插补
          "drop": None}                                            # 完整样本分析（不插补）
features = ["hours", "edu", "age"]                                 # 用来预测的观测变量
known = damaged.dropna(subset=["income"])                          # 用于拟合模型的行
reg = LinearRegression().fit(known[features], known["income"])      # 回归插补模型
reg_pred = reg.predict(damaged.loc[damaged["income"].isna(), features])   # 对缺失行做预测
reg_series = damaged["income"].copy()                              # 复制一列准备填值
reg_series.loc[damaged["income"].isna()] = reg_pred                # 用预测值填充
filled["regression"] = reg_series                                  # 登记该策略
knn = KNNImputer(n_neighbors=5).fit_transform(damaged[features + ["income"]])   # KNN 插补
filled["knn"] = pd.Series(knn[:, -1], index=damaged.index)         # 取出收入那一列
mice = IterativeImputer(max_iter=10, random_state=42).fit_transform(damaged[features + ["income"]])   # MICE
filled["mice"] = pd.Series(mice[:, -1], index=damaged.index)       # 迭代插补的结果
complete_case = damaged.dropna(subset=["income"])                  # 只分析完整行的方案

# ---------------------------------------------------------------- 4) 给每种策略打分
rows = []                                                          # 每种策略一行
for name, series in filled.items():                                # 逐个评估
    data = complete_case if series is None else damaged.assign(income=series)   # 选出要分析的表
    mean = data["income"].mean()                                   # 均值估计
    sd = data["income"].std()                                      # 离散度估计
    corr = data["income"].corr(data["hours"])                      # 与工作时长的相关性
    slope = LinearRegression().fit(data[["hours"]], data["income"]).coef_[0]   # 下游回归系数
    rows.append({"strategy": name, "n_used": int(data["income"].notna().sum()),   # 实际参与分析的行数
                 "mean": round(mean, 2), "mean_bias": round(mean - truth_mean, 2),   # 均值的偏差
                 "sd": round(sd, 2), "sd_bias": round(sd - truth_sd, 2),   # 标准差的偏差
                 "corr_hours": round(corr, 3), "corr_error": round(corr - truth_corr, 3),   # 相关性误差
                 "slope": round(slope, 3), "slope_error": round(slope - coef_truth, 3)})   # 回归系数误差
scores = pd.DataFrame(rows)                                        # 汇总对比表
print(scores.to_string(index=False))                               # 终端打印
dump(scores, "l3_imputation_metrics.csv")                          # 落盘对比表
strategy_means = pd.DataFrame({"strategy": ["complete_data"] + list(filled),   # 真值 + 各策略
                               "mean_income": [truth_mean] + [mean for mean in scores["mean"]]})   # 均值序列
dump(strategy_means, "l3_imputation_means.csv")                    # 落盘画图用的小表

# ---------------------------------------------------------------- 图：分布与误差
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                # 密度 / 箱线 / 误差
bins = np.linspace(complete["income"].min(), complete["income"].max(), 30)   # 统一分箱便于比较
axes[0].hist(complete["income"], bins=bins, density=True, color="#1E2630",   # 真值作为参照
             alpha=0.35, label="complete (truth)")                 # 图例：完整数据
for name, color in zip(["mean", "regression", "mice"], ["#66717D", "#C9A227", "#8A0C3C"]):
    axes[0].hist(filled[name], bins=bins, density=True, histtype="step",   # 三种补法的分布
                 lw=1.6, color=color, label=name)                  # 每种一条曲线
axes[0].set_title("income distribution after imputation")           # 子图标题
axes[0].set_xlabel("income (10k yuan)")                             # 横轴含义
axes[0].legend(fontsize=8.5)                                        # 紧凑图例
box_data = [complete["income"]] + [filled[k] for k in ["mean", "median", "regression", "knn", "mice"]]   # 待比较序列
axes[1].boxplot(box_data, patch_artist=True,                        # 各策略的离散度对比
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))     # 箱体填充
axes[1].set_xticklabels(["truth", "mean", "median", "regr", "knn", "mice"], fontsize=8.5)   # 标签
axes[1].set_title("spread is the first casualty")                   # 子图标题
x = np.arange(len(scores))                                          # 柱状图横坐标
axes[2].bar(x - 0.2, scores["mean_bias"], width=0.4, label="mean bias", color="#8A0C3C")   # 水平偏差
axes[2].bar(x + 0.2, scores["sd_bias"], width=0.4, label="spread bias", color="#C9A227")   # 离散偏差
axes[2].axhline(0, color="#1E2630", lw=1.0)                         # 零线 = 无误差
axes[2].set_xticks(x, scores["strategy"], rotation=20, fontsize=8.5)   # 策略名称
axes[2].set_title("bias vs the complete-data truth")                # 子图标题
axes[2].set_ylabel("income units")                                  # 两种柱子同单位
axes[2].legend(fontsize=8.5)                                        # 紧凑图例
plt.tight_layout()                                                  # 调整三个子图
save(fig, "fig_imputation_compare.png")                             # 保存：插补一页用图
