"""l4_10_uncertainty.py —— 不确定性怎么画：误差棒、自助法区间、置信带与扇形预测。

"不画不确定性"本身就是一种误导。脚本用自助法（bootstrap）从同一份 A/B 数据里
估出转化率与客单价的区间，展示四种常见的表达方式，并把结果存成表便于写进报告。
"""
import numpy as np                                                 # 数值计算与重采样
import pandas as pd                                                # 数据处理
import matplotlib.pyplot as plt                                    # 四视图
from _common_l4 import RNG, DAT, save, dump                        # 复用公共工具

ab = pd.read_csv(DAT / "l4_ab_test.csv")                           # 读取 A/B 实验数据（来自 l4_09）
B = 2000                                                           # 自助法重复次数
def bootstrap_ci(values, stat=np.mean, level=0.95, boot=B):         # 通用自助法区间函数
    """对给定样本重复重采样，返回点估计与百分位置信区间。"""             # 函数约定
    values = np.asarray(values, dtype=float)                       # 转成浮点数组
    idx = RNG.integers(0, len(values), size=(boot, len(values)))    # 每次重采样抽取同样大小的下标
    stats_ = stat(values[idx], axis=1)                             # 计算每次重采样的统计量
    lo, hi = np.percentile(stats_, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])   # 百分位区间
    return float(stat(values)), float(lo), float(hi), stats_       # 返回点估计、区间与分布

rows = []                                                          # 汇总表
boot_store = {}                                                    # 保存自助分布供画图
for g in ["对照组 A", "实验组 B"]:                                  # 逐个组计算
    sub = ab[ab["group"] == g]                                     # 取出该组样本
    m, lo, hi, dist = bootstrap_ci(sub["converted"])               # 转化率的自助区间
    rows.append({"指标": f"{g}·转化率", "点估计": round(m, 4),       # 记录点估计
                 "CI下界": round(lo, 4), "CI上界": round(hi, 4),    # 记录置信区间
                 "样本量": int(len(sub))})                          # 记录样本量
    boot_store[g] = (m, lo, hi, dist)                              # 暂存自助分布
    rev = sub.loc[sub["converted"], "revenue"]                     # 只取转化者的客单价
    m2, lo2, hi2, dist2 = bootstrap_ci(rev)                        # 客单价的自助区间
    rows.append({"指标": f"{g}·客单价", "点估计": round(m2, 2),      # 记录客单价点估计
                 "CI下界": round(lo2, 2), "CI上界": round(hi2, 2),  # 客单价区间
                 "样本量": int(len(rev))})                          # 客单价样本量
    boot_store[g + "·客单价"] = (m2, lo2, hi2, dist2)               # 暂存
diff_dist = boot_store["实验组 B"][3] - boot_store["对照组 A"][3]     # 两组差值的自助分布
d_point, d_lo, d_hi = float(diff_dist.mean()), float(np.percentile(diff_dist, 2.5)), float(np.percentile(diff_dist, 97.5))   # 差值自助分布的均值与 95% 分位区间
rows.append({"指标": "转化率差值（B−A）", "点估计": round(d_point, 4),   # 差值点估计
             "CI下界": round(d_lo, 4), "CI上界": round(d_hi, 4),     # 差值区间
             "样本量": int(len(ab))})                              # 总样本量
unc = pd.DataFrame(rows)                                           # 汇总成表
dump(unc, "l4_uncertainty_summary.csv")                            # 落盘不确定性汇总
print(unc.to_string(index=False))                                  # 终端打印
crosses_zero = d_lo < 0 < d_hi                                     # 差值区间是否跨 0
print(f"差值 95% CI = [{d_lo:+.4f}, {d_hi:+.4f}]，跨 0：{crosses_zero}")   # 打印结论提示

# ---------------------------------------------------------------- 图：不确定性四视图
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.7))                # 误差棒 / 自助分布 / 置信带 / 扇形预测
mA, loA, hiA, _ = boot_store["对照组 A"]                            # A 组转化率
mB, loB, hiB, _ = boot_store["实验组 B"]                            # B 组转化率
axes[0].errorbar([0, 1], [mA * 100, mB * 100],                      # 点估计 + 误差棒
                 yerr=[[(mA - loA) * 100, (mB - loB) * 100],        # 下误差
                       [(hiA - mA) * 100, (hiB - mB) * 100]],       # 上误差
                 fmt="o", capsize=5, color="#8A0C3C", ms=7)         # 圆点与端帽
axes[0].set_xticks([0, 1], ["A 组", "B 组"])                        # 分组标签
axes[0].set_ylim(0, max(hiA, hiB) * 100 * 1.25)                    # 留出空间
axes[0].set_title("① 点估计 + 95% 误差棒")                          # 子图标题
axes[0].set_ylabel("转化率 %")                                      # 轴标签
axes[1].hist(diff_dist * 100, bins=40, color="#2E6F8E", alpha=0.85)   # ② 差值的自助分布
axes[1].axvline(0, color="#1E2630", lw=1.4, ls="--")                # 零线
axes[1].axvline(d_lo * 100, color="#C9A227", lw=1.2)                # 区间下界
axes[1].axvline(d_hi * 100, color="#C9A227", lw=1.2, label="95% 区间")   # 区间上界
axes[1].set_title("② 自助法：差值的分布")                            # 子图标题
axes[1].set_xlabel("转化率差值（百分点）"); axes[1].legend(fontsize=8.4)   # 轴标签与图例
daily = ab.assign(day=RNG.integers(1, 21, len(ab))).groupby(["day", "group"])["converted"].mean().unstack()   # 按天聚合的转化率（用于置信带）
band_lo, band_hi = [], []                                          # ③ 逐日自助带
for d in daily.index:                                              # 逐天计算区间
    v = daily.loc[d, "实验组 B"]                                   # 当天 B 组转化率
    m, lo, hi, _ = bootstrap_ci(ab.loc[(ab["group"] == "实验组 B"), "converted"], level=0.8)   # 用整体样本估区间
    band_lo.append(max(lo * 100, 0)); band_hi.append(hi * 100)     # 记录上下界（百分比）
axes[2].plot(daily.index, daily["实验组 B"] * 100, "-o", ms=3, color="#8A0C3C", lw=1.4)   # 折线
axes[2].fill_between(daily.index, band_lo, band_hi, color="#8A0C3C", alpha=0.18, label="80% 区间")   # 置信带
axes[2].set_title("③ 时间序列 + 置信带")                            # 子图标题
axes[2].set_xlabel("实验天数"); axes[2].set_ylabel("转化率 %")        # 轴标签
axes[2].legend(fontsize=8.4)                                       # 图例
months = np.arange(1, 25)                                          # ④ 未来 24 期的预测
center = 12.0 + 0.18 * months                                      # 中心预测（缓慢上升）
width = 0.9 + 0.42 * months                                        # 预测不确定性随时间扩大
axes[3].plot(months, center, color="#8A0C3C", lw=1.8, label="中心预测")   # 中心线
axes[3].fill_between(months, center - width, center + width, color="#C9A227", alpha=0.25, label="80% 预测区间")   # 80% 带
axes[3].fill_between(months, center - 2 * width, center + 2 * width, color="#C9A227", alpha=0.15, label="95% 预测区间")   # 95% 带
axes[3].set_title("④ 扇形预测：越远越不确定")                        # 子图标题
axes[3].set_xlabel("未来第 n 期"); axes[3].set_ylabel("指标值")      # 轴标签
axes[3].legend(fontsize=8.2)                                       # 图例
plt.tight_layout()                                                 # 调整四格
save(fig, "fig_uncertainty.png")                                   # 保存：不确定性四视图
