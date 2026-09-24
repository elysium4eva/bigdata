"""l3_09_panel_balance.py —— 平衡与非平衡面板、缺口与插值补齐。

课件里的面板例子在这里变成真实数据：一个 T=11、n=2 的平衡面板，一个各公司 T_i 不同的
"名义平衡"面板，以及一个含块状缺失的面板。脚本先诊断结构，再按个体插值补齐，
并把每一个被插补的格子标出来，让"修补"始终可见。
"""
import numpy as np                                                 # 随机游走生成面板取值
import pandas as pd                                                # 面板重塑与插值
import matplotlib.pyplot as plt                                    # 面板热力图
from _common_l3 import RNG, save, dump                             # 复用公共种子、出图与落盘函数

MONTHS = list(range(1, 12))                                        # T = 11 期，与课件一致

# ---------------------------------------------------------------- 1) 平衡面板：n=2，T=11
def make_series(n, start):                                         # 生成一条公司层面的时间序列
    """带个体水平与趋势的随机游走，每期一个取值。"""                     # 函数约定
    level = start + np.cumsum(RNG.normal(0.6, 1.1, n))             # 逐期累加随机变化
    return np.round(level, 2)                                      # 四舍五入便于阅读
balanced = pd.DataFrame({"firm_id": np.repeat(["F1", "F2"], len(MONTHS)),   # 两家公司
                         "month": MONTHS * 2,                      # 两家都有这 11 期
                         "revenue": np.concatenate([make_series(len(MONTHS), 100),   # F1 的序列
                                                    make_series(len(MONTHS), 130)])})   # F2 的序列
dump(balanced, "l3_panel_balanced.csv")                            # 落盘平衡面板
wide_balanced = balanced.pivot(index="firm_id", columns="month", values="revenue")   # 公司 × 月矩阵
print("balanced panel shape (n x T):", wide_balanced.shape)        # 打印形状：n=2、T=11

# ---------------------------------------------------------------- 2) 非平衡面板：各公司 T_i 不同
spec = {"F1": 9, "F2": 11, "F3": 8}                                # T1=9、T2=11、T3=8，n=3
frames = []                                                        # 每家公司一个数据框
for i, (firm, periods) in enumerate(spec.items()):                 # 逐家公司生成长度不同的序列
    frames.append(pd.DataFrame({"firm_id": firm,                   # 公司标识
                                "month": list(range(1, periods + 1)),   # 这家公司自己的时间范围
                                "revenue": make_series(periods, 90 + 25 * i)}))   # 取值序列
unbalanced = pd.concat(frames, ignore_index=True)                  # 拼成非平衡面板
unbalanced.loc[(unbalanced["firm_id"] == "F2") &                # 再埋一块缺失
               (unbalanced["month"].isin([4, 5])), "revenue"] = np.nan   # F2 的第 4、5 月缺观测
dump(unbalanced, "l3_panel_unbalanced.csv")                        # 落盘非平衡面板
wide_unbalanced = unbalanced.pivot(index="firm_id", columns="month", values="revenue")   # 含缺口矩阵
coverage = (unbalanced.assign(observed=unbalanced["revenue"].notna())   # 逐公司覆盖统计
            .groupby("firm_id")["observed"].agg(periods="count", observed="sum"))   # 应有期数 / 有观测期数
coverage["T_i"] = [spec[f] for f in coverage.index]                # 每家公司声明的期数
coverage["coverage_pct"] = (100 * coverage["observed"] / coverage["periods"]).round(1)   # 覆盖率
coverage["balanced"] = coverage["periods"] == coverage["periods"].max()   # 期数是否都一样
print("panel coverage table:\n", coverage.to_string())              # 终端打印结构诊断
dump(coverage.reset_index(), "l3_panel_coverage.csv")               # 落盘覆盖率表

# ---------------------------------------------------------------- 3) 补齐缺口并做标记
repaired = unbalanced.sort_values(["firm_id", "month"]).copy()      # 先排序，插值才有时间顺序
repaired["revenue_interpolated"] = (repaired.groupby("firm_id")["revenue"]   # 分组：按个体分别插值
                                    .transform(lambda s: s.interpolate(limit_direction="both")))   # 前后都填
repaired["filled_flag"] = repaired["revenue"].isna() & repaired["revenue_interpolated"].notna()   # 标记被补的格子
repaired["revenue"] = repaired["revenue_interpolated"]             # 把补好的值提升为主列
dump(repaired, "l3_panel_repaired.csv")                            # 落盘补齐后的面板
wide_repaired = repaired.pivot(index="firm_id", columns="month", values="revenue")   # 补齐后的矩阵
still_missing = int(wide_repaired.isna().sum().sum())              # 插值补不了的格子还剩几个
print("cells still missing after interpolation:", still_missing)   # 插值无法凭空造出期间
print("imputed cells:", int(repaired["filled_flag"].sum()))        # 实际补出来的格子数

# ---------------------------------------------------------------- 图 A：平衡面板
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))                 # 折线 + 热力图
for firm in balanced["firm_id"].unique():                          # 每家公司一条线
    sub = balanced[balanced["firm_id"] == firm]                    # 取出该公司各期数据
    axes[0].plot(sub["month"], sub["revenue"], "-o", ms=3.5, lw=1.4, label=firm)   # 折线图
axes[0].set_title("balanced panel: n = 2, T = 11")                  # 子图标题
axes[0].set_xlabel("month"); axes[0].set_ylabel("revenue")          # 坐标轴含义
axes[0].legend(title="firm")                                        # 图例
im = axes[1].imshow(wide_balanced.values, cmap="RdPu", aspect="auto")   # 公司 × 月热力图
axes[1].set_xticks(range(len(MONTHS)), MONTHS, fontsize=8)          # 横轴：月份
axes[1].set_yticks(range(wide_balanced.shape[0]), wide_balanced.index, fontsize=10)   # 纵轴：公司
axes[1].set_title("heat map: every cell is observed")               # 子图标题
axes[1].grid(False)                                                 # 热力图不需要网格
plt.colorbar(im, ax=axes[1], fraction=0.046)                        # 数值色带
plt.tight_layout()                                                  # 调整两格
save(fig, "fig_panel_balanced.png")                                 # 保存：平衡面板一页用图

# ---------------------------------------------------------------- 图 B：补齐前后
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # 两张热力图 + 覆盖率
for ax, matrix, title in [(axes[0], wide_unbalanced, "before: gaps and shorter T_i"),   # 原始结构
                          (axes[1], wide_repaired, "after: interpolated (marked cells)")]:   # 补齐之后
    ax.imshow(matrix.values, cmap="RdPu", aspect="auto")            # 热力图
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, fontsize=8)   # 横轴：月份
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=10)    # 纵轴：公司
    ax.set_title(title, fontsize=10.5)                              # 子图标题
    ax.grid(False)                                                  # 热力图不需要网格
for _, row in repaired[repaired["filled_flag"]].iterrows():         # 把每个插补格子圈出来
    axes[1].scatter(row["month"] - 1, list(wide_repaired.index).index(row["firm_id"]),   # 格子坐标
                    marker="o", s=90, facecolors="none", edgecolors="#1E2630", lw=1.2)   # 空心圈
axes[2].bar(coverage.index, coverage["coverage_pct"], color="#8A0C3C", alpha=0.85)   # 各公司覆盖率
axes[2].axhline(100, color="#C9A227", ls="--", lw=1.2, label="full coverage")   # 100% 参考线
axes[2].set_ylim(0, 115)                                            # 给参考线留出空间
axes[2].set_title("coverage per firm (%)")                          # 子图标题
axes[2].legend(fontsize=8.5)                                        # 图例
plt.tight_layout()                                                  # 调整三格
save(fig, "fig_panel_heatmap.png")                                  # 保存：面板补齐一页用图
