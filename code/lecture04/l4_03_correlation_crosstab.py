"""l4_03_correlation_crosstab.py —— 相关与列联表：从统计量到图形。

脚本用一份"贸易引力"风格的合成数据复现原课件的结论：
当 X 的取值范围被截断（例如只留下距离较小的一小段），
Pearson 与 Spearman 相关系数都会明显减弱。
同时用调研样本演示列联表（交叉表）与条件均值。
"""
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 表格与交叉表
import matplotlib.pyplot as plt                                    # 散点与对照图
from scipy.stats import chi2_contingency                           # 列联表的卡方检验
from _common_l4 import RNG, DAT, save, dump, style_axes, annotated   # 复用公共工具与图注工具

# ---------------------------------------------------------------- 1) 合成"贸易对"数据（引力模型风格）
N = 220                                                            # 国家对数量
gdp_x = RNG.lognormal(mean=26.0, sigma=1.0, size=N)                # 出口国 GDP（美元，右偏）
gdp_m = RNG.lognormal(mean=26.3, sigma=1.0, size=N)                # 进口国 GDP
dist = RNG.lognormal(mean=8.6, sigma=0.7, size=N)                  # 双边距离（公里）
trade = (0.9 * np.log(gdp_x) + 0.85 * np.log(gdp_m)                # 引力模型：贸易额随两边规模上升
         - 1.15 * np.log(dist) + RNG.normal(0, 0.55, N))           # 随距离下降 + 随机扰动
trade_value = np.exp(trade + 6.0)                                  # 还原为美元量级
pairs = pd.DataFrame({"exporter": [f"E{i:03d}" for i in range(1, N + 1)],   # 出口国编号
                      "importer": [f"I{i:03d}" for i in range(1, N + 1)],   # 进口国编号
                      "distance_km": np.round(dist, 0),            # 双边距离
                      "gdp_exporter": np.round(gdp_x, 0),          # 出口国 GDP
                      "gdp_importer": np.round(gdp_m, 0),          # 进口国 GDP
                      "trade_usd": np.round(trade_value, 0)})      # 贸易额
dump(pairs, "l4_trade_pairs.csv")                                  # 落盘样本

log_trade = np.log(pairs["trade_usd"])                             # 取对数：右偏变量对称化
log_dist = np.log(pairs["distance_km"])                            # 距离同样取对数
r_full = np.corrcoef(log_dist, log_trade)[0, 1]                    # Pearson 相关（全样本）
rho_full = pd.Series(log_dist).corr(pd.Series(log_trade), method="spearman")   # Spearman（全样本）
# 受限取值范围：只保留距离最小的 25%，模拟"只研究近邻贸易伙伴"的选样
cut = np.quantile(log_dist, 0.25)                                  # 25% 分位作为截断点
mask = log_dist <= cut                                             # 截断掩码
r_restricted = np.corrcoef(log_dist[mask], log_trade[mask])[0, 1]   # 受限后的 Pearson
rho_restricted = pd.Series(log_dist[mask]).corr(pd.Series(log_trade[mask]), method="spearman")   # 受限后的 Spearman
print(f"全样本 Pearson={r_full:.3f} Spearman={rho_full:.3f}")        # 打印全样本相关
print(f"受限范围 Pearson={r_restricted:.3f} Spearman={rho_restricted:.3f}")   # 打印受限后的相关

# ---------------------------------------------------------------- 2) 相关系数矩阵
analyze = pairs[["gdp_exporter", "gdp_importer", "distance_km", "trade_usd"]].copy()   # 选取分析列
analyze["log_trade"] = log_trade                                   # 加入对数贸易额
analyze["log_distance"] = log_dist                                 # 加入对数距离
corr_pearson = analyze.corr(method="pearson").round(3)             # Pearson 相关矩阵
corr_spearman = analyze.corr(method="spearman").round(3)           # Spearman 相关矩阵
dump(corr_pearson.reset_index(names="变量"), "l4_correlation_pearson.csv")   # 落盘 Pearson 矩阵
dump(corr_spearman.reset_index(names="变量"), "l4_correlation_spearman.csv")   # 落盘 Spearman 矩阵
print("Pearson 矩阵（前 3 行）:\n", corr_pearson.head(3).to_string())   # 终端打印片段

# ---------------------------------------------------------------- 3) 列联表与条件均值（用 L4 调研样本）
survey = pd.read_csv(DAT / "l4_survey_raw.csv")                    # 读取 l4_01 生成的调研样本
survey["income_band"] = pd.cut(survey["income_k"],                 # 把收入切成三档
                               bins=[0, 8, 16, 100],               # 分档边界（千元）
                               labels=["低收入", "中等收入", "高收入"])   # 三档名称
ct = pd.crosstab(survey["income_band"], survey["channel"],         # 交叉表：收入档 × 渠道
                 margins=True, margins_name="合计")                # 带边际合计
ct_pct = pd.crosstab(survey["income_band"], survey["channel"], normalize="index").round(3)   # 行百分比
cond = survey.groupby("income_band", observed=True)["satisfaction"].agg(["mean", "median", "count"]).round(2)   # 收入档的条件统计（均值/中位数/样本量）
print("收入档 × 渠道（行百分比）:\n", ct_pct.to_string())            # 打印行百分比
dump(ct.reset_index(), "l4_crosstab_income_channel.csv")           # 落盘交叉表
dump(cond.reset_index(), "l4_conditional_satisfaction.csv")        # 落盘条件均值
# 卡方与 Cramér's V：列联表强度的直觉指标
core = ct.iloc[:-1, :-1]                                           # 去掉边际合计后的核心表
chi2, p, dof, expected = chi2_contingency(core)                     # 卡方检验：是否独立
n_obs = core.values.sum()                                          # 样本量
cramers_v = float(np.sqrt(chi2 / (n_obs * (min(core.shape) - 1))))  # Cramér's V：0—1 的关联强度
print(f"卡方={chi2:.1f}  p={p:.3f}  Cramér's V={cramers_v:.3f}")     # 打印关联强度

# ---------------------------------------------------------------- 图：受限范围对相关的影响
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))                # 全样本 / 受限范围 / 系数对照
axes[0].scatter(log_dist, log_trade, s=14, color="#8A0C3C", alpha=0.55)   # 全样本散点
axes[0].axvline(cut, color="#66717D", ls="--", lw=1.2, label="25% 分位截断线")   # 标出截断位置
z = np.polyfit(log_dist, log_trade, 1)                             # 全样本回归线
xs = np.linspace(log_dist.min(), log_dist.max(), 50)               # 直线横坐标
axes[0].plot(xs, np.polyval(z, xs), color="#2E6F8E", lw=1.8)        # 画回归线
style_axes(axes[0], "全样本：距离与贸易额", "ln(距离)", "ln(贸易额)")   # 统一风格
annotated(axes[0], f"Pearson {r_full:.2f}\nSpearman {rho_full:.2f}")   # 图内注释相关系数
axes[0].legend(fontsize=8.4)                                       # 图例
axes[1].scatter(log_dist[mask], log_trade[mask], s=16, color="#2E6F8E", alpha=0.7)   # 受限范围散点
z2 = np.polyfit(log_dist[mask], log_trade[mask], 1)                # 受限样本回归线
xs2 = np.linspace(log_dist[mask].min(), log_dist[mask].max(), 50)   # 直线横坐标
axes[1].plot(xs2, np.polyval(z2, xs2), color="#8A0C3C", lw=1.8)     # 画回归线
style_axes(axes[1], "受限范围：只留最近 25% 的伙伴", "ln(距离)", "ln(贸易额)")   # 统一风格
annotated(axes[1], f"Pearson {r_restricted:.2f}\nSpearman {rho_restricted:.2f}")   # 图内注释
bars = ["Pearson 全样本", "Pearson 受限", "Spearman 全样本", "Spearman 受限"]   # 四个对照柱
vals = [r_full, r_restricted, rho_full, rho_restricted]            # 对应数值（本例均为负相关）
vals_abs = [abs(v) for v in vals]                                  # 取绝对值画柱（纵轴即"相关强度"）
axes[2].bar(bars, vals_abs, color=["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"], alpha=0.9)   # 柱状对照
axes[2].set_ylim(0, max(vals_abs) * 1.3)                           # 纵轴从 0 起，并留出标注空间
style_axes(axes[2], "截断范围后相关强度变小", "", "相关系数绝对值 |r|")   # 统一风格
for i, v in enumerate(vals_abs):                                   # 柱顶标注数值
    axes[2].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5)   # 数值标签
axes[2].tick_params(axis="x", labelsize=8.0)                       # 缩小轴标签字号，避免与边距冲突
plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.17, wspace=0.30)   # 手工留边距（比 tight_layout 更可控）
save(fig, "fig_restricted_range.png")                              # 保存：相关与截断一页用图
