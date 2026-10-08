"""l4_12_report_pipeline.py —— 报告流水线：统计表、图与图注一起产出。

把前面几步的结果按"报告顺序"组装：描述统计表 → 分组对照表 → 图 → 图注文字，
全部落盘到 data/lecture04/report/，复制进报告即可，图注里写清数据来源、
样本量、口径与不确定性——这正是"图注自足性"的要求。
"""
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 表格处理
import matplotlib.pyplot as plt                                    # 报告配图
from _common_l4 import DAT, save, dump                             # 复用公共工具

REPORT = DAT / "report"                                            # 报告素材目录
REPORT.mkdir(parents=True, exist_ok=True)                          # 不存在则创建

# ---------------------------------------------------------------- 1) 读入前面生成的样本
survey = pd.read_csv(DAT / "l4_survey_raw.csv")                    # 调研样本（l4_01）
ab = pd.read_csv(DAT / "l4_ab_test.csv")                           # 实验数据（l4_09）
unc = pd.read_csv(DAT / "l4_uncertainty_summary.csv")              # 不确定性汇总（l4_10）

# ---------------------------------------------------------------- 2) 表一：描述统计
desc = survey[["age", "income_k", "satisfaction", "spend_k"]].describe().T.round(3)   # 常用描述统计
desc["缺失率"] = survey[["age", "income_k", "satisfaction", "spend_k"]].isna().mean().round(4)   # 补缺失率
desc = desc.reset_index(names="变量")                               # 变量名成列
desc.to_csv(REPORT / "table1_descriptive.csv", index=False, encoding="utf-8-sig")   # 落盘表一
print("[data]", REPORT / "table1_descriptive.csv")                 # 打印产物路径

# ---------------------------------------------------------------- 3) 表二：分组对照（渠道 × 满意度）
by_channel = (survey.groupby("channel")                              # 按渠道分组
              .agg(样本量=("respondent_id", "count"),                 # 每组样本量
                   满意度均值=("satisfaction", "mean"),                # 满意度均值
                   收入中位数=("income_k", "median"),                  # 收入中位数（稳健）
                   复购率=("revisit", "mean"))                         # 复购率
              .round(3).reset_index())                               # 保留三位小数
by_channel.to_csv(REPORT / "table2_channel.csv", index=False, encoding="utf-8-sig")   # 落盘表二
print("[data]", REPORT / "table2_channel.csv")                     # 打印产物路径

# ---------------------------------------------------------------- 4) 图：一页报告主图（四联）
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.6))                # 报告主图四个面板
axes[0].bar(by_channel["channel"], by_channel["满意度均值"],        # ① 渠道满意度
            color="#8A0C3C", alpha=0.88)                            # 单色，避免颜色噪声
axes[0].set_ylim(2.5, 5.0)                                          # 聚焦差异区间并标注起点
axes[0].set_title("各渠道满意度均值"); axes[0].set_ylabel("满意度（1—5）")   # 标题与轴标签
axes[0].text(0.02, 4.9, "纵轴从 2.5 起，仅用于看差异", fontsize=7.8, color="#66717D", va="top")   # 明示截断
axes[1].hist(survey["income_k"].dropna(), bins=30, color="#2E6F8E", alpha=0.85)   # ② 收入分布
axes[1].axvline(survey["income_k"].median(), color="#C9A227", lw=1.6, label="中位数")   # 中位数线
axes[1].set_title("收入分布（缺失 12 条，未插补）")                   # 标题写明缺失处理
axes[1].set_xlabel("月收入（千元）"); axes[1].legend(fontsize=8)     # 轴标签与图例
rate = ab.groupby("group")["converted"].mean() * 100                # ③ 实验转化率
se = np.sqrt((rate / 100) * (1 - rate / 100) / ab.groupby("group").size()) * 100 * 1.96   # 95% 误差
axes[2].bar(rate.index, rate.values, yerr=se.values, capsize=5,      # 带误差棒的柱状图
            color=["#66717D", "#8A0C3C"], alpha=0.9)                # 两组两色
axes[2].set_ylim(0, 20); axes[2].set_title("A/B 转化率（含 95% 误差棒）")   # 标题
axes[2].set_ylabel("转化率 %")                                       # 轴标签
axes[2].text(0.02, 19, f"n={len(ab):,}", fontsize=8, color="#66717D", va="top")   # 标注样本量
ci = unc[unc["指标"].str.contains("客单价")]                          # ④ 客单价的区间对照
ypos = np.arange(len(ci))                                           # 纵坐标
axes[3].errorbar(ci["点估计"], ypos,                               # 点估计 + 区间
                 xerr=[ci["点估计"] - ci["CI下界"], ci["CI上界"] - ci["点估计"]],   # 左右误差
                 fmt="o", capsize=4, color="#711426")                # 圆点与端帽
axes[3].set_yticks(ypos, ci["指标"], fontsize=8.4)                  # 组名标签
axes[3].set_title("客单价点估计与 95% 区间"); axes[3].set_xlabel("元")   # 标题与轴标签
plt.tight_layout()                                                 # 调整四个面板
save(fig, "fig_report_panel.png")                                  # 保存：报告主图

# ---------------------------------------------------------------- 5) 图注：自足性写法
captions = [                                                        # 每条图注都能独立读懂
    "图 1　各渠道满意度均值（数据：l4_survey_raw.csv，n=500；满意度为 1—5 分，"         # 图 1 图注（含数据来源、样本量、口径）
    "缺失 5 条按缺失处理未插补；纵轴自 2.5 起，仅用于观察渠道间差异）。",                      # 图 1 图注续行
    "图 2　受访者月收入分布（n=500；收入以千元计，右偏；12 条敏感题缺失未插补，"                  # 图 2 图注
    "图中以中位数标注中心位置）。",                                             # 图 2 图注续行
    "图 3　A/B 实验转化率对比（n=2400，随机分组各半；误差棒为 95% 置信区间；"                # 图 3 图注
    "结论为'实验组有提升提示，但区间与对照组重叠，需扩大样本再判断'）。",                         # 图 3 图注续行（写明不确定性结论）
    "图 4　转化者客单价的点估计与 95% 自助法置信区间（B=2000 次重采样；"                    # 图 4 图注
    "客单价为右偏分布，故同时报告中位数）。",                                        # 图 4 图注续行
]                                                                 # 图注列表结束
(REPORT / "captions.md").write_text("# 图注（自足性写法示例）\n\n" + "\n\n".join(captions), encoding="utf-8")
print("[data]", REPORT / "captions.md")                            # 打印产物路径
print("report pack:", sorted(p.name for p in REPORT.iterdir()))     # 列出报告素材清单
