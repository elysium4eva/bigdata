"""l3_07_missing_mechanisms.py —— 造出 MCAR / MAR / MNAR 三种缺失，并量化它们带来的偏差。

同一份完整总体被以三种方式破坏。因为真值已知，脚本可以直接展示每种机制如何扭曲
一个简单估计（人均收入）：MCAR 基本无偏，MAR 在正确建模下可修正，MNAR 则无法在不做
假设的前提下修好。
"""
import numpy as np                                                 # 随机抽样与偏差计算
import pandas as pd                                                # 表格与缺失处理工具
import matplotlib.pyplot as plt                                    # 缺失矩阵图
from _common_l3 import RNG, save, dump                             # 复用公共种子、出图与落盘函数

# ---------------------------------------------------------------- 1) 一份完整的总体
N = 800                                                            # 受调查的家庭数
education = RNG.choice(["high_school", "bachelor", "master"], N,   # 学历影响收入
                       p=[0.45, 0.40, 0.15])                       # 贴近现实的学历结构
age = RNG.integers(22, 60, N)                                      # 劳动年龄（岁）
base = {"high_school": 8.6, "bachelor": 9.6, "master": 10.2}       # 各学历的对数收入截距
mu = np.array([base[e] for e in education]) + 0.012 * (age - 22)   # 收入随年龄增长
income = RNG.lognormal(mean=mu, sigma=0.30)                        # 完整且已知的月收入（元）
complete = pd.DataFrame({"household_id": [f"H{i:04d}" for i in range(N)],   # 家庭主键
                         "education": education, "age": age,       # 完全观测的协变量
                         "income": np.round(income, 2)})           # 将被"弄缺"的变量
dump(complete, "l3_income_complete_l3_07.csv")                     # 落盘完整样本
true_mean = complete["income"].mean()                              # 真正要估计的目标量：平均收入
print(f"true mean income (complete data): {true_mean:.2f} yuan/month")   # 作为对比的基准

# ---------------------------------------------------------------- 2) 三种缺失机制
mcar_mask = RNG.random(N) < 0.30                                   # MCAR：每行都有同样的 30% 概率
mar_p = np.where(education == "master", 0.55,                  # MAR：缺失取决于学历，
                 np.where(education == "bachelor", 0.30, 0.12))    # 而学历是完整观测的变量
mar_mask = RNG.random(N) < mar_p                                   # 按概率抽出 MAR 缺失
mnar_p = np.clip((income - income.min()) / (income.max() - income.min()), 0, 1) ** 2   # 越富越不愿填
mnar_mask = RNG.random(N) < 0.15 + 0.55 * mnar_p                   # MNAR：高收入更容易缺失
mechanisms = {"MCAR": mcar_mask, "MAR": mar_mask, "MNAR": mnar_mask}   # 三种被破坏的版本
rows, panels, matrices = [], {}, {}                                # 报告、样本与画图用的容器
for name, mask in mechanisms.items():                              # 每种机制各造一份表
    damaged = complete.copy()                                      # 完整行保持不动
    damaged.loc[mask, "income"] = np.nan                           # 只把收入删掉
    naive_mean = damaged["income"].mean()                          # 用观测到的行估计均值
    rows.append({"mechanism": name, "missing_rate_pct": round(100 * mask.mean(), 1),   # 缺失比例
                 "observed_mean": round(naive_mean, 2),            # 朴素估计
                 "bias_vs_truth": round(naive_mean - true_mean, 2),   # 与真值的绝对差
                 "bias_pct": round(100 * (naive_mean - true_mean) / true_mean, 1)})   # 相对偏差
    panels[name] = damaged                                         # 留作 CSV 导出
    matrices[name] = damaged[["income", "age", "education"]].isna().astype(int)   # 0/1 矩阵用于画图
bias = pd.DataFrame(rows)                                          # 每种机制一行
print(bias.to_string(index=False))                                 # 终端打印偏差表
dump(bias, "l3_missing_bias.csv")                                  # 落盘偏差表
for name, damaged in panels.items():                               # 三份被破坏的版本各存一份
    dump(damaged, f"l3_income_missing_{name}.csv")                  # 例如 l3_income_missing_MNAR.csv

# ---------------------------------------------------------------- 3) 含块状缺失的面板样本
n_ids, n_periods = 6, 12                                            # 一个小型公司面板
grid = pd.DataFrame([(f"F{i:02d}", t) for i in range(1, n_ids + 1)  # 完整的（公司, 月份）网格
                     for t in range(1, n_periods + 1)],            # 每家公司 12 个月
                    columns=["firm_id", "month"])                   # 列名
grid["revenue"] = np.round(RNG.lognormal(6.0, 0.35, len(grid)), 1)  # 每个公司-月一个营收值
drop_firm = grid["firm_id"].isin(["F03", "F05"]) & grid["month"].isin([3, 4, 5])   # 整块数据消失
drop_tail = grid["firm_id"].eq("F06") & grid["month"].gt(9)         # 一家公司提前退出样本
grid.loc[drop_firm | drop_tail, "revenue"] = np.nan                 # 面板从此不平衡
dump(grid, "l3_missing_panel_sample.csv")                          # 落盘含块状缺失的面板（区别于 l3_09 的面板）
coverage = (grid.assign(observed=grid["revenue"].notna())           # 逐家公司的覆盖情况
            .groupby("firm_id")["observed"].agg(["sum", "count"]))  # 有观测的月数 / 总月数
coverage["coverage_pct"] = (100 * coverage["sum"] / coverage["count"]).round(1)   # 覆盖率（%）
print("panel coverage:\n", coverage.to_string())                    # 终端打印覆盖率表
dump(coverage.reset_index(), "l3_missing_panel_coverage.csv")       # 落盘覆盖率表（避免与 l3_09 同名覆盖）

# ---------------------------------------------------------------- 图：缺失矩阵与偏差
fig = plt.figure(figsize=(11.0, 4.2))                               # 自定义布局：两张矩阵 + 一张柱状图
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.5])         # 左中右三块
for i, name in enumerate(["MCAR", "MAR"]):                          # 画前两种机制
    ax = fig.add_subplot(gs[0, i])                                  # 每种机制一个坐标区
    ax.imshow(matrices[name].T.values, cmap="Greys", aspect="auto", vmin=0, vmax=1)   # 1 = 缺失（深色）
    ax.set_yticks(range(3), ["income", "age", "education"], fontsize=9)   # 纵轴：变量
    ax.set_xlabel("households")                                     # 横轴：家庭
    ax.set_title(f"{name}: missing pattern "                        # 标题里带上缺失比例
                 f"({100 * mechanisms[name].mean():.0f}% of income missing)", fontsize=10.5)   # 比例数值
    ax.grid(False)                                                  # 矩阵不需要网格
ax = fig.add_subplot(gs[0, 2])                                      # 右图：朴素均值的偏差
colors = ["#66717D", "#C9A227", "#8A0C3C"]                          # 灰 / 金 / 红 对应三种机制
ax.bar(bias["mechanism"], bias["bias_pct"], color=colors, alpha=0.9)   # 相对偏差柱状图
ax.axhline(0, color="#1E2630", lw=1.0)                              # 零线 = 无偏
ax.set_title("bias of the naive mean income (%)", fontsize=10.5)    # 子图标题
for i, v in enumerate(bias["bias_pct"]):                            # 在柱子上标注数值
    ax.text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)   # 数值标签
plt.tight_layout()                                                  # 调整布局
save(fig, "fig_missing_matrix.png")                                 # 保存：缺失数据一页用图
