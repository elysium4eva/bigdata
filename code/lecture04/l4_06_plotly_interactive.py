"""l4_06_plotly_interactive.py -- interactive charts: interact to explore, go static to report.

The script builds three browser-openable plotly charts (line, scatter, Sankey)
and exports them as single-file HTML; matplotlib then draws a "static vs interactive" comparison
that shows which one fits which job (exploration vs final reporting).
"""
import numpy as np                                                 # numeric computation
import pandas as pd                                                # data table handling
import matplotlib.pyplot as plt                                    # static comparison figure
import plotly.express as px                                       # plotly quick plotting API
import plotly.graph_objects as go                                  # low-level API: Sankey and other special charts
from _common_l4 import RNG, INT, DAT, save, dump                   # reuse shared helpers, the interactive dir and the data dir

# ---------------------------------------------------------------- 1) synthetic monthly energy data
months = pd.date_range("2015-01-01", periods=132, freq="MS")        # 132 months starting 2015-01
SOURCES = ["煤炭", "天然气", "水电", "风电光伏"]                     # four energy sources
rows = []                                                          # one row per month and source
for i, m in enumerate(months):                                     # loop over every month
    season = 1 + 0.12 * np.sin(2 * np.pi * (m.month - 1) / 12)      # seasonal swing (more power in winter and summer)
    trend = 1 + 0.0025 * i                                         # slow upward long-run trend
    share = {"煤炭": 0.58 - 0.0016 * i, "天然气": 0.16 + 0.0006 * i,   # coal share falls year by year
             "水电": 0.14, "风电光伏": 0.12 + 0.0012 * i}            # clean energy share rises
    for s in SOURCES:                                              # loop over the four sources
        base = 320 * share[s]                                      # baseline generation for that source
        value = base * season * trend * np.exp(RNG.normal(0, 0.04))   # add season, trend and noise
        rows.append({"month": m, "source": s, "generation_twh": round(float(value), 2)})   # record one row
energy = pd.DataFrame(rows)                                        # collect into a long table
dump(energy, "l4_energy.csv")                                      # write the sample

# ---------------------------------------------------------------- 2) three interactive charts (exported as HTML)
line_fig = px.line(energy, x="month", y="generation_twh", color="source",   # line chart: colored by source
                   title="月度发电量（可悬浮查看数值、可框选缩放）",         # the title states the interactive features
                   labels={"month": "月份", "generation_twh": "发电量（TWh）", "source": "能源"})   # Chinese axis names
line_fig.write_html(INT / "energy_trend.html", include_plotlyjs="cdn")   # single-file HTML (CDN plotly)
print("[interactive]", INT / "energy_trend.html")                  # print the output path
survey = pd.read_csv(DAT / "l4_survey_raw.csv")                    # read the survey sample (from l4_01)
plot_df = survey.dropna(subset=["satisfaction", "income_k", "spend_k"])   # plotly rejects missing values: drop those rows first
print("plotly 用样本:", len(plot_df), "（原样本", len(survey), "，缺失行不参与交互图）")   # print the difference in sample size
scatter_fig = px.scatter(plot_df, x="income_k", y="spend_k", color="channel",   # scatter: colored by channel
                         size="satisfaction", hover_data=["city", "education"],   # hover shows the details
                         title="收入与消费（点大小=满意度，可缩放与筛选）",       # title
                         labels={"income_k": "月收入（千元）", "spend_k": "月消费（千元）", "channel": "渠道"})   # Chinese axis names
scatter_fig.write_html(INT / "income_spend.html", include_plotlyjs="cdn")   # export the interactive scatter
print("[interactive]", INT / "income_spend.html")                   # print the output path
flow = energy.groupby("source", as_index=False)["generation_twh"].sum()   # aggregate by source
flow["sector"] = ["工业", "居民", "工业", "交通"]                    # simplified destinations (teaching illustration)
labels = flow["source"].tolist() + flow["sector"].tolist()          # Sankey node labels (sources + sectors)
sankey_fig = go.Figure(data=[go.Sankey(                            # build the Sankey with graph_objects
    node=dict(label=labels, pad=14, thickness=16,                   # node style
              color=["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"] * 2),   # node colors
    link=dict(source=list(range(len(flow))),                        # link source indices
              target=[len(flow) + i for i in range(len(flow))],      # link target indices
              value=flow["generation_twh"].round(1).tolist()))])     # link width equals the value
sankey_fig.update_layout(title_text="能源流向示意（桑基图：适合表达构成与流向）",   # title
                         font=dict(size=12))                        # font size
sankey_fig.write_html(INT / "energy_sankey.html", include_plotlyjs="cdn")   # export the Sankey
print("[interactive]", INT / "energy_sankey.html")                  # print the output path

# ---------------------------------------------------------------- 3) static figure: for the final report
pivot = energy.pivot(index="month", columns="source", values="generation_twh")   # wide table is convenient for stacking
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))                 # static on the left, interactive sketch on the right
for s, c in zip(SOURCES, ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # one color per energy source
    axes[0].plot(pivot.index, pivot[s].rolling(3).mean(), lw=1.6,   # draw the 3-month moving average (suppresses noise)
                 color=c, label=s)                                 # color and legend
axes[0].set_title("静态图：定稿汇报用（信息固定、可打印）")            # subplot title
axes[0].set_xlabel("月份"); axes[0].set_ylabel("发电量（TWh）")        # axis labels
axes[0].legend(fontsize=8.4, ncol=2)                               # two-column legend
axes[1].plot(pivot.index, pivot["风电光伏"], lw=1.6, color="#2E6F8E")   # the series "selected" in the interactive chart
axes[1].fill_between(pivot.index, pivot["风电光伏"].min() * 0.9,     # highlight a range to mimic box selection
                     pivot["风电光伏"].max() * 1.05, alpha=0.12, color="#C9A227")   # light highlight
axes[1].annotate("悬浮显示数值\n框选缩放区间\n图例点击筛系列",          # annotate the interactive features
                 xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9.2,   # text position
                 va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3",   # text box
                                                fc="#F7F3F2", ec="#D6CDD0"))   # box colors
axes[1].set_title("交互图：探索用（可缩放、筛选、悬浮）")               # subplot title
axes[1].set_xlabel("月份"); axes[1].set_ylabel("发电量（TWh）")        # axis labels
plt.tight_layout()                                                 # adjust the two panels
save(fig, "fig_interactive_vs_static.png")                         # save: interactive vs static comparison
print("energy rows:", len(energy), "| interactive files ->", INT)   # closing log
