"""l4_12_report_pipeline.py -- report pipeline: statistics tables, figures and captions produced together.

Assemble the earlier results in report order: descriptive table -> group comparison table -> figure -> captions,
all written to data/lecture04/report/ ready to paste into a report, with captions stating the data source,
sample size, definitions and uncertainty -- exactly what "self-contained captions" requires.
"""
import numpy as np                                                 # numeric computation
import pandas as pd                                                # table handling
import matplotlib.pyplot as plt                                    # report figures
from _common_l4 import DAT, save, dump                             # reuse shared helpers

REPORT = DAT / "report"                                            # report asset directory
REPORT.mkdir(parents=True, exist_ok=True)                          # create it when missing

# ---------------------------------------------------------------- 1) read the samples generated earlier
survey = pd.read_csv(DAT / "l4_survey_raw.csv")                    # survey sample (l4_01)
ab = pd.read_csv(DAT / "l4_ab_test.csv")                           # experiment data (l4_09)
unc = pd.read_csv(DAT / "l4_uncertainty_summary.csv")              # uncertainty summary (l4_10)

# ---------------------------------------------------------------- 2) table 1: descriptive statistics
desc = survey[["age", "income_k", "satisfaction", "spend_k"]].describe().T.round(3)   # common descriptive statistics
desc["缺失率"] = survey[["age", "income_k", "satisfaction", "spend_k"]].isna().mean().round(4)   # add the missing rate
desc = desc.reset_index(names="变量")                               # variable name becomes a column
desc.to_csv(REPORT / "table1_descriptive.csv", index=False, encoding="utf-8-sig")   # write table 1
print("[data]", REPORT / "table1_descriptive.csv")                 # print the output path

# ---------------------------------------------------------------- 3) table 2: group comparison (channel x satisfaction)
by_channel = (survey.groupby("channel")                              # group by channel
              .agg(样本量=("respondent_id", "count"),                 # sample size per group
                   满意度均值=("satisfaction", "mean"),                # mean satisfaction
                   收入中位数=("income_k", "median"),                  # median income (robust)
                   复购率=("revisit", "mean"))                         # repurchase rate
              .round(3).reset_index())                               # keep three decimals
by_channel.to_csv(REPORT / "table2_channel.csv", index=False, encoding="utf-8-sig")   # write table 2
print("[data]", REPORT / "table2_channel.csv")                     # print the output path

# ---------------------------------------------------------------- 4) figure: one-page report figure (four panels)
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.6))                # four panels for the report
axes[0].bar(by_channel["channel"], by_channel["满意度均值"],        # (1) satisfaction by channel
            color="#8A0C3C", alpha=0.88)                            # single color, avoids color noise
axes[0].set_ylim(2.5, 5.0)                                          # focus on the difference range and annotate the start
axes[0].set_title("各渠道满意度均值"); axes[0].set_ylabel("满意度（1—5）")   # title and axis label
axes[0].text(0.02, 4.9, "纵轴从 2.5 起，仅用于看差异", fontsize=7.8, color="#66717D", va="top")   # state the truncation explicitly
axes[1].hist(survey["income_k"].dropna(), bins=30, color="#2E6F8E", alpha=0.85)   # (2) income distribution
axes[1].axvline(survey["income_k"].median(), color="#C9A227", lw=1.6, label="中位数")   # median line
axes[1].set_title("收入分布（缺失 12 条，未插补）")                   # the title states the missing-value handling
axes[1].set_xlabel("月收入（千元）"); axes[1].legend(fontsize=8)     # axis label and legend
rate = ab.groupby("group")["converted"].mean() * 100                # (3) experiment conversion rate
se = np.sqrt((rate / 100) * (1 - rate / 100) / ab.groupby("group").size()) * 100 * 1.96   # 95% error
axes[2].bar(rate.index, rate.values, yerr=se.values, capsize=5,      # bars with error bars
            color=["#66717D", "#8A0C3C"], alpha=0.9)                # two groups, two colors
axes[2].set_ylim(0, 20); axes[2].set_title("A/B 转化率（含 95% 误差棒）")   # title
axes[2].set_ylabel("转化率 %")                                       # axis label
axes[2].text(0.02, 19, f"n={len(ab):,}", fontsize=8, color="#66717D", va="top")   # annotate the sample size
ci = unc[unc["指标"].str.contains("客单价")]                          # (4) order value interval comparison
ypos = np.arange(len(ci))                                           # y coordinates
axes[3].errorbar(ci["点估计"], ypos,                               # point estimate plus interval
                 xerr=[ci["点估计"] - ci["CI下界"], ci["CI上界"] - ci["点估计"]],   # left and right errors
                 fmt="o", capsize=4, color="#711426")                # marker and caps
axes[3].set_yticks(ypos, ci["指标"], fontsize=8.4)                  # group name labels
axes[3].set_title("客单价点估计与 95% 区间"); axes[3].set_xlabel("元")   # title and axis label
plt.tight_layout()                                                 # adjust the four panels
save(fig, "fig_report_panel.png")                                  # save: report main figure

# ---------------------------------------------------------------- 5) captions: self-contained wording
captions = [                                                        # every caption must stand alone
    "图 1　各渠道满意度均值（数据：l4_survey_raw.csv，n=500；满意度为 1—5 分，"         # figure 1 caption (source, sample size, definitions)
    "缺失 5 条按缺失处理未插补；纵轴自 2.5 起，仅用于观察渠道间差异）。",                      # figure 1 caption, continued
    "图 2　受访者月收入分布（n=500；收入以千元计，右偏；12 条敏感题缺失未插补，"                  # figure 2 caption
    "图中以中位数标注中心位置）。",                                             # figure 2 caption, continued
    "图 3　A/B 实验转化率对比（n=2400，随机分组各半；误差棒为 95% 置信区间；"                # figure 3 caption
    "结论为'实验组有提升提示，但区间与对照组重叠，需扩大样本再判断'）。",                         # figure 3 caption, continued (states the uncertainty conclusion)
    "图 4　转化者客单价的点估计与 95% 自助法置信区间（B=2000 次重采样；"                    # figure 4 caption
    "客单价为右偏分布，故同时报告中位数）。",                                        # figure 4 caption, continued
]                                                                 # end of the caption list
(REPORT / "captions.md").write_text("# 图注（自足性写法示例）\n\n" + "\n\n".join(captions), encoding="utf-8")
print("[data]", REPORT / "captions.md")                            # print the output path
print("report pack:", sorted(p.name for p in REPORT.iterdir()))     # list the report assets
