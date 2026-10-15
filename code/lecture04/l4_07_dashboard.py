"""l4_07_dashboard.py -- build a dashboard in Python (instead of Power BI / Tableau).

This lecture uses no commercial BI tool: the dashboard is assembled from plotly subplots and KPI cards into one HTML file
that is interactive when opened (hover, zoom, legend filtering); matplotlib draws a static version
for slides and printing. Both outputs use the same data and the same definitions.
"""
import numpy as np                                                 # numeric computation
import pandas as pd                                                # data aggregation
import matplotlib.pyplot as plt                                    # static dashboard
import plotly.graph_objects as go                                  # low-level API: assemble subplots precisely
from plotly.subplots import make_subplots                          # dashboard layout
from _common_l4 import INT, DAT, save, dump                        # reuse shared helpers and the interactive dir

# ---------------------------------------------------------------- 1) read the samples generated earlier
survey = pd.read_csv(DAT / "l4_survey_raw.csv")                    # survey sample (from l4_01)
cities = pd.read_csv(DAT / "l4_city_gdp.csv")                      # city sample (from l4_05)
energy = pd.read_csv(DAT / "l4_energy.csv", parse_dates=["month"])   # energy sample (from l4_06)

# ---------------------------------------------------------------- 2) KPI metrics
kpi = {                                                            # the four key metrics at the top of the dashboard
    "样本量": f"{len(survey):,}",                                   # number of respondents
    "平均满意度": f"{survey['satisfaction'].mean():.2f} / 5",        # mean satisfaction
    "复购率": f"{survey['revisit'].mean():.1%}",                    # repurchase share
    "人均 GDP 中位数": f"{cities['gdp_per_cap'].median() / 1e4:.1f} 万元",   # median city GDP per capita
}                                                                 # end of the KPI dict
print("KPI:", kpi)                                                 # print the key metrics
dump(pd.DataFrame([{"指标": k, "取值": v} for k, v in kpi.items()]), "l4_dashboard_kpi.csv")   # write the KPI table

# ---------------------------------------------------------------- 3) interactive plotly dashboard
fig = make_subplots(rows=2, cols=2,                                # 2x2 panels
                    subplot_titles=("各渠道满意度分布", "收入与消费",   # four subplot titles
                                    "月度发电量（按能源）", "城市 GDP 与人口"))   # more titles
for ch, c in zip(survey["channel"].unique(), ["#8A0C3C", "#C9A227", "#2E6F8E"]):   # box plot per channel
    sub = survey.loc[survey["channel"] == ch, "satisfaction"].dropna()   # satisfaction for that channel
    fig.add_trace(go.Box(y=sub, name=ch, marker_color=c, boxmean=True),   # box plot (with mean marker)
                  row=1, col=1)                                    # place it top left
fig.add_trace(go.Scatter(x=survey["income_k"], y=survey["spend_k"],   # income vs spending scatter
                         mode="markers", marker=dict(size=5, opacity=0.5, color="#711426"),   # semi-transparent points
                         name="受访者"), row=1, col=2)             # place it top right
for src, c in zip(energy["source"].unique(), ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # trend line per energy source
    sub = energy[energy["source"] == src]                          # the series for that source
    fig.add_trace(go.Scatter(x=sub["month"], y=sub["generation_twh"],   # line
                             mode="lines", name=src, line=dict(color=c)),   # color and legend
                  row=2, col=1)                                    # place it bottom left
fig.add_trace(go.Scatter(x=cities["pop_wan"], y=cities["gdp_yi"],   # city population vs GDP
                         mode="markers", text=cities["city"],      # hover shows the city name
                         marker=dict(size=8, color=cities["growth_pct"],   # point size and color encoding
                                     colorscale="RdPu", showscale=True),   # continuous palette for the growth rate
                         name="城市"), row=2, col=2)               # place it bottom right
fig.update_layout(title_text="消费者调研与能源/城市指标仪表板（Python + plotly）",   # overall title
                  height=760, showlegend=True, template="plotly_white",   # height and theme
                  title_x=0.02)                                    # left-align the title
fig.write_html(INT / "dashboard.html", include_plotlyjs="cdn")      # export the single-file HTML
print("[interactive]", INT / "dashboard.html")                      # print the output path

# ---------------------------------------------------------------- 4) static matplotlib dashboard
fig2 = plt.figure(figsize=(13.2, 6.4))                             # canvas: KPI row on top plus 2x2 panels
gs = fig2.add_gridspec(2, 4, height_ratios=[0.28, 1.0], hspace=0.42, wspace=0.35)   # custom grid
for i, (k, v) in enumerate(kpi.items()):                           # the four KPI cards
    ax_card = fig2.add_subplot(gs[0, i])                           # one cell per card
    ax_card.axis("off")                                            # the card draws no axes
    ax_card.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_card.transAxes,   # card background
                                    fc="#F7F3F2", ec="#D6CDD0", lw=0.8))   # light fill and thin border
    ax_card.text(0.5, 0.62, v, ha="center", va="center", fontsize=15, color="#8A0C3C", weight="bold")   # the value
    ax_card.text(0.5, 0.22, k, ha="center", va="center", fontsize=9.5, color="#66717D")   # the metric name
ax1 = fig2.add_subplot(gs[1, 0])                                   # top left: satisfaction by channel
data = [survey.loc[survey["channel"] == ch, "satisfaction"].dropna() for ch in survey["channel"].unique()]   # satisfaction series per channel (missing values dropped)
bp = ax1.boxplot(data, patch_artist=True)                          # box plot
ax1.set_xticklabels(list(survey["channel"].unique()))              # group labels (set_xticklabels since 3.9)
for patch, c in zip(bp["boxes"], ["#8A0C3C", "#C9A227", "#2E6F8E"]):   # color them
    patch.set_facecolor(c); patch.set_alpha(0.6)                   # fill and alpha
ax1.set_title("各渠道满意度分布"); ax1.set_ylabel("满意度（1—5）")    # title and axis label
ax2 = fig2.add_subplot(gs[1, 1])                                   # top right: income vs spending
ax2.scatter(survey["income_k"], survey["spend_k"], s=8, color="#711426", alpha=0.45)   # scatter
ax2.set_title("收入与消费"); ax2.set_xlabel("月收入（千元）"); ax2.set_ylabel("月消费（千元）")   # title and axes
ax3 = fig2.add_subplot(gs[1, 2])                                   # bottom left: energy trend
pivot = energy.pivot(index="month", columns="source", values="generation_twh")   # wide table
for src, c in zip(pivot.columns, ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # one line per source
    ax3.plot(pivot.index, pivot[src].rolling(3).mean(), lw=1.3, color=c, label=src)   # 3-month smoothing
ax3.set_title("月度发电量（按能源）"); ax3.set_ylabel("TWh"); ax3.legend(fontsize=7.6, ncol=2)   # title and legend
ax4 = fig2.add_subplot(gs[1, 3])                                   # bottom right: city population vs GDP
sc = ax4.scatter(cities["pop_wan"], cities["gdp_yi"], s=26,        # point size and color encode the growth rate
                 c=cities["growth_pct"], cmap="RdPu")              # continuous palette
ax4.set_title("城市 GDP 与人口"); ax4.set_xlabel("人口（万人）"); ax4.set_ylabel("GDP（亿元）")   # title and axes
plt.colorbar(sc, ax=ax4, fraction=0.046, label="增速 %")            # color bar
save(fig2, "fig_dashboard.png")                                    # save the static dashboard
print("dashboard KPI cards:", len(kpi), "| panels: 4")              # closing log
