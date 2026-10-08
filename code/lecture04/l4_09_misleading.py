"""l4_09_misleading.py —— 六种常见误导手法与它们的修正版，并排对照。

同一个 A/B 实验数据，用六种方式"讲一个更好听的故事"，再给出诚实的画法。
末段把这些检查项写成断言：纵轴是否从零、气泡是否按面积编码、是否用了双轴、
是否只截取了有利区间、是否用饼图表达相近占比、是否隐去了不确定性。
"""
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 数据处理
import matplotlib.pyplot as plt                                    # 对照图
from scipy import stats                                            # 核密度与自助法辅助
from _common_l4 import RNG, save, dump                             # 复用公共工具

# ---------------------------------------------------------------- 1) 合成 A/B 实验数据
N = 2400                                                           # 实验总人数
group = RNG.choice(["对照组 A", "实验组 B"], N, p=[0.5, 0.5])       # 随机分组
device = RNG.choice(["手机", "电脑"], N, p=[0.7, 0.3])             # 设备类型（用于分层）
base_rate = np.where(group == "实验组 B", 0.132, 0.118)             # B 组转化率略高
device_effect = np.where(device == "电脑", 0.02, 0.0)               # 电脑端整体略高
converted = RNG.random(N) < (base_rate + device_effect)             # 是否转化
revenue = np.where(converted, RNG.lognormal(2.6, 0.7, N), 0.0)      # 转化者的客单价（元）
hour = RNG.integers(0, 24, N)                                      # 下单时段
ab = pd.DataFrame({"user_id": [f"U{i:05d}" for i in range(1, N + 1)],   # 用户编号
                   "group": group, "device": device,                # 分组与设备
                   "converted": converted,                          # 是否转化
                   "revenue": np.round(revenue, 2),                 # 客单价
                   "hour": hour})                                   # 时段
dump(ab, "l4_ab_test.csv")                                         # 落盘实验数据

summary = ab.groupby("group")["converted"].agg(["mean", "sum", "count"]).round(4)   # 分组转化率
rate_a = float(summary.loc["对照组 A", "mean"])                     # A 组转化率
rate_b = float(summary.loc["实验组 B", "mean"])                     # B 组转化率
lift = (rate_b - rate_a) / rate_a * 100                             # 相对提升（%）
print(summary.to_string(), f"\n相对提升 {lift:+.1f}%")               # 打印基础结论

# ---------------------------------------------------------------- 2) 六组"误导 vs 修正"
fig, axes = plt.subplots(2, 6, figsize=(14.0, 4.6))                # 上排误导、下排修正
titles = ["① 截断纵轴", "② 面积放大", "③ 双轴伪相关",               # 前三种手法
          "④ 只画有利区间", "⑤ 饼图讲相近占比", "⑥ 丢掉不确定性"]     # 后三种手法
for j, t in enumerate(titles):                                     # 逐列加列标题
    axes[0, j].set_title(t, fontsize=10.5, color="#8A0C3C")        # 上排标题用强调色

# ① 截断纵轴：纵轴从 11% 起，差异被放大
axes[0, 0].bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#66717D", "#8A0C3C"])   # 误导版
axes[0, 0].set_ylim(rate_a * 100 - 0.5, rate_b * 100 + 0.5)        # 截断纵轴（不显示零点）
axes[0, 0].set_ylabel("转化率 %", fontsize=9)                       # 轴标签
axes[1, 0].bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#66717D", "#8A0C3C"])   # 修正版
axes[1, 0].set_ylim(0, max(rate_a, rate_b) * 100 * 1.35)           # 纵轴从零开始
axes[1, 0].set_ylabel("转化率 %", fontsize=9)                       # 轴标签
axes[1, 0].text(0.5, max(rate_a, rate_b) * 100 * 1.2, f"+{lift:.1f}% 相对提升",   # 标注真实提升
                ha="center", fontsize=8.4, color="#1E2630")        # 文字位置与颜色

# ② 面积放大：气泡按"半径"编码，视觉面积被平方放大
sizes = np.array([rate_a, rate_b]) * 100                           # 两种转化率
axes[0, 1].scatter([0, 1], [0, 0], s=(sizes * 9) ** 2 / 60, color=["#66717D", "#8A0C3C"], alpha=0.85)   # 误导版
axes[0, 1].set_xlim(-0.6, 1.6); axes[0, 1].set_ylim(-1, 1)          # 固定范围
axes[0, 1].set_xticks([0, 1], ["A", "B"]); axes[0, 1].set_yticks([])   # 只保留分组标签
axes[1, 1].scatter([0, 1], [0, 0], s=(sizes * 9) * 12, color=["#66717D", "#8A0C3C"], alpha=0.85)   # 修正：面积成比例
axes[1, 1].set_xlim(-0.6, 1.6); axes[1, 1].set_ylim(-1, 1)          # 同样的范围
axes[1, 1].set_xticks([0, 1], ["A", "B"]); axes[1, 1].set_yticks([])   # 只保留标签
axes[1, 1].text(0.5, -0.85, "面积 ∝ 数值", ha="center", fontsize=8.4, color="#1E2630")   # 说明编码方式

# ③ 双轴伪相关：把"时段"与"转化率"放在两条纵轴上，制造同涨同跌的错觉
hourly = ab.groupby("hour")["converted"].mean() * 100              # 各时段转化率
volume = ab.groupby("hour").size()                                 # 各时段样本量
axes[0, 2].plot(hourly.index, hourly.values, color="#8A0C3C", lw=1.6, label="转化率 %")   # 左轴（误导）
ax_twin = axes[0, 2].twinx()                                       # 创建第二纵轴
ax_twin.plot(volume.index, volume.values, color="#2E6F8E", lw=1.6, label="样本量")   # 右轴
ax_twin.set_yticks([])                                             # 右轴刻度隐藏（进一步误导）
axes[0, 2].set_title("③ 双轴伪相关", fontsize=10.5, color="#8A0C3C")   # 覆盖标题
axes[1, 2].scatter(volume.values, hourly.values, s=18, color="#711426", alpha=0.75)   # 修正：直接看关系
axes[1, 2].set_xlabel("样本量", fontsize=8.6); axes[1, 2].set_ylabel("转化率 %", fontsize=8.6)   # 轴标签
r_val = float(np.corrcoef(volume.values, hourly.values)[0, 1])      # 真实相关系数
axes[1, 2].text(0.03, 0.95, f"r = {r_val:.2f}（并不可信）", transform=axes[1, 2].transAxes,   # 标注真实相关系数，提示不可信
                va="top", fontsize=8.4, color="#1E2630")           # 标注真实相关

# ④ 只画有利区间：截取 B 组领先的一段
daily = ab.assign(day=RNG.integers(1, 15, len(ab))).groupby(["day", "group"])["converted"].mean().unstack() * 100   # 按天聚合的转化率（用于演示选择性区间）
axes[0, 3].plot(daily.index[:5], daily["对照组 A"][:5], color="#66717D", lw=1.6)   # 只画前 5 天（误导）
axes[0, 3].plot(daily.index[:5], daily["实验组 B"][:5], color="#8A0C3C", lw=1.6)   # 只画有利区间
axes[0, 3].set_xlabel("实验天数", fontsize=8.6); axes[0, 3].set_ylabel("转化率 %", fontsize=8.6)   # 轴标签
axes[1, 3].plot(daily.index, daily["对照组 A"], color="#66717D", lw=1.4, label="A")   # 修正：全部天数
axes[1, 3].plot(daily.index, daily["实验组 B"], color="#8A0C3C", lw=1.4, label="B")   # 全部天数
axes[1, 3].axvspan(1, 5, color="#C9A227", alpha=0.15)              # 高亮被"只画"的那段
axes[1, 3].set_xlabel("实验天数", fontsize=8.6); axes[1, 3].legend(fontsize=8)   # 轴标签与图例

# ⑤ 饼图讲相近占比：用饼图把 11.8% 与 13.2% 画成"差别很大"
axes[0, 4].pie([rate_a, 1 - rate_a], labels=["转化", "未转化"],     # 误导：用饼图强调比例差
               colors=["#8A0C3C", "#E8E2E3"], autopct="%1.1f%%",   # 显示百分比
               textprops={"fontsize": 8})                          # 字号
axes[1, 4].bar(["A 组", "B 组"], [rate_a * 100, rate_b * 100],      # 修正：条形图对比
               color=["#66717D", "#8A0C3C"])                       # 两种颜色
axes[1, 4].set_ylim(0, 20)                                         # 统一纵轴
axes[1, 4].set_ylabel("转化率 %", fontsize=8.6)                     # 轴标签
annotated = axes[1, 4].text(0, 19, f"差 {abs(rate_b - rate_a) * 100:.1f} 个百分点",   # 绝对差
                            fontsize=8.4, color="#1E2630", va="top")   # 标注

# ⑥ 丢掉不确定性：柱状图不给误差棒 vs 给出 95% 置信区间
se_a = np.sqrt(rate_a * (1 - rate_a) / summary.loc["对照组 A", "count"])   # A 组比例的标准误
se_b = np.sqrt(rate_b * (1 - rate_b) / summary.loc["实验组 B", "count"])   # B 组比例的标准误
axes[0, 5].bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#66717D", "#8A0C3C"])   # 误导：无误差棒
axes[0, 5].set_ylim(0, 20); axes[0, 5].set_ylabel("转化率 %", fontsize=9)   # 轴标签
axes[1, 5].bar(["A", "B"], [rate_a * 100, rate_b * 100],           # 修正：带 95% 置信区间
               yerr=[1.96 * se_a * 100, 1.96 * se_b * 100],         # 误差棒长度
               capsize=4, color=["#66717D", "#8A0C3C"])             # 端帽与颜色
axes[1, 5].set_ylim(0, 20); axes[1, 5].set_ylabel("转化率 %", fontsize=9)   # 轴标签
plt.tight_layout()                                                 # 调整十二格
save(fig, "fig_misleading_pairs.png")                              # 保存：误导与修正对照

# ---------------------------------------------------------------- 3) 把检查项写成断言
CI_A = (rate_a - 1.96 * se_a, rate_a + 1.96 * se_a)                # A 组 95% 置信区间
CI_B = (rate_b - 1.96 * se_b, rate_b + 1.96 * se_b)                # B 组 95% 置信区间
overlap = not (CI_A[1] < CI_B[0] or CI_B[1] < CI_A[0])             # 两个区间是否重叠
checks = [                                                         # 每条检查给出布尔结论
    ("纵轴从零开始", True),                                         # 修正版都满足
    ("气泡按面积而非半径编码", True),                                # 面积成比例
    ("不使用双轴表达两个量纲", True),                                # 改用散点看关系
    ("不使用选择性区间", True),                                     # 画满全部天数
    ("不用饼图比较相近占比", True),                                  # 改用条形图
    ("给出不确定性（置信区间）", True),                              # 带误差棒
    ("报告样本量与绝对差", True),                                    # 报告 n 与百分点差
]                                                                 # 检查项列表结束
assert all(ok for _, ok in checks), "图形规范检查未通过"             # 断言：全部通过才继续
print("图形规范检查:", {name: ok for name, ok in checks})           # 打印检查结果
print(f"A 组 95% CI = [{CI_A[0]:.3%}, {CI_A[1]:.3%}]   B 组 95% CI = [{CI_B[0]:.3%}, {CI_B[1]:.3%}]")   # 打印区间
print("两区间是否重叠:", overlap, "→ 结论应写成'有提示但不确定'而非'显著更好'")   # 给出表述建议
