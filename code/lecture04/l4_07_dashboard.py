"""l4_07_dashboard.py —— 用 Python 搭一个仪表板（替代 Power BI / Tableau）。

本讲不使用 BI 商业软件：仪表板用 plotly 子图 + KPI 卡拼装，导出一个单文件 HTML，
打开即可交互（悬浮、缩放、图例筛选）；同时用 matplotlib 画一张静态版，
便于放进课件与打印。两份产物用同一份数据、同一套口径。
"""
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 数据聚合
import matplotlib.pyplot as plt                                    # 静态仪表板
import plotly.graph_objects as go                                  # 低级接口：精确拼装子图
from plotly.subplots import make_subplots                          # 仪表板布局
from _common_l4 import INT, DAT, save, dump                        # 复用公共工具与交互目录

# ---------------------------------------------------------------- 1) 读入前面生成的样本
survey = pd.read_csv(DAT / "l4_survey_raw.csv")                    # 调研样本（来自 l4_01）
cities = pd.read_csv(DAT / "l4_city_gdp.csv")                      # 城市样本（来自 l4_05）
energy = pd.read_csv(DAT / "l4_energy.csv", parse_dates=["month"])   # 能耗样本（来自 l4_06）

# ---------------------------------------------------------------- 2) KPI 指标
kpi = {                                                            # 仪表板顶部的四个关键指标
    "样本量": f"{len(survey):,}",                                   # 受访者数量
    "平均满意度": f"{survey['satisfaction'].mean():.2f} / 5",        # 满意度均值
    "复购率": f"{survey['revisit'].mean():.1%}",                    # 复购比例
    "人均 GDP 中位数": f"{cities['gdp_per_cap'].median() / 1e4:.1f} 万元",   # 城市人均 GDP 中位数
}                                                                 # KPI 字典结束
print("KPI:", kpi)                                                 # 打印关键指标
dump(pd.DataFrame([{"指标": k, "取值": v} for k, v in kpi.items()]), "l4_dashboard_kpi.csv")   # 落盘 KPI

# ---------------------------------------------------------------- 3) plotly 交互仪表板
fig = make_subplots(rows=2, cols=2,                                # 2×2 面板
                    subplot_titles=("各渠道满意度分布", "收入与消费",   # 四个子图标题
                                    "月度发电量（按能源）", "城市 GDP 与人口"))   # 继续补标题
for ch, c in zip(survey["channel"].unique(), ["#8A0C3C", "#C9A227", "#2E6F8E"]):   # 逐渠道画箱线
    sub = survey.loc[survey["channel"] == ch, "satisfaction"].dropna()   # 取该渠道满意度
    fig.add_trace(go.Box(y=sub, name=ch, marker_color=c, boxmean=True),   # 箱线（带均值标记）
                  row=1, col=1)                                    # 放到左上格
fig.add_trace(go.Scatter(x=survey["income_k"], y=survey["spend_k"],   # 收入与消费散点
                         mode="markers", marker=dict(size=5, opacity=0.5, color="#711426"),   # 半透明点
                         name="受访者"), row=1, col=2)             # 放到右上格
for src, c in zip(energy["source"].unique(), ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # 逐能源画趋势
    sub = energy[energy["source"] == src]                          # 取该能源序列
    fig.add_trace(go.Scatter(x=sub["month"], y=sub["generation_twh"],   # 折线
                             mode="lines", name=src, line=dict(color=c)),   # 颜色与图例
                  row=2, col=1)                                    # 放到左下格
fig.add_trace(go.Scatter(x=cities["pop_wan"], y=cities["gdp_yi"],   # 城市人口与 GDP
                         mode="markers", text=cities["city"],      # 悬浮显示城市名
                         marker=dict(size=8, color=cities["growth_pct"],   # 点大小与颜色编码
                                     colorscale="RdPu", showscale=True),   # 用连续色板表示增速
                         name="城市"), row=2, col=2)               # 放到右下格
fig.update_layout(title_text="消费者调研与能源/城市指标仪表板（Python + plotly）",   # 总标题
                  height=760, showlegend=True, template="plotly_white",   # 高度与主题
                  title_x=0.02)                                    # 标题左对齐
fig.write_html(INT / "dashboard.html", include_plotlyjs="cdn")      # 导出单文件 HTML
print("[interactive]", INT / "dashboard.html")                      # 打印产物路径

# ---------------------------------------------------------------- 4) matplotlib 静态仪表板
fig2 = plt.figure(figsize=(13.2, 6.4))                             # 画布：顶部 KPI + 2×2 面板
gs = fig2.add_gridspec(2, 4, height_ratios=[0.28, 1.0], hspace=0.42, wspace=0.35)   # 自定义网格
for i, (k, v) in enumerate(kpi.items()):                           # 四个 KPI 卡
    ax_card = fig2.add_subplot(gs[0, i])                           # 每卡一格
    ax_card.axis("off")                                            # 卡片不画坐标轴
    ax_card.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_card.transAxes,   # 卡片底框
                                    fc="#F7F3F2", ec="#D6CDD0", lw=0.8))   # 浅底细边
    ax_card.text(0.5, 0.62, v, ha="center", va="center", fontsize=15, color="#8A0C3C", weight="bold")   # 数值
    ax_card.text(0.5, 0.22, k, ha="center", va="center", fontsize=9.5, color="#66717D")   # 指标名
ax1 = fig2.add_subplot(gs[1, 0])                                   # 左上：渠道满意度
data = [survey.loc[survey["channel"] == ch, "satisfaction"].dropna() for ch in survey["channel"].unique()]   # 各渠道满意度序列（缺失已剔除）
bp = ax1.boxplot(data, patch_artist=True)                          # 箱线图
ax1.set_xticklabels(list(survey["channel"].unique()))              # 分组标签（3.9 起用 set_xticklabels）
for patch, c in zip(bp["boxes"], ["#8A0C3C", "#C9A227", "#2E6F8E"]):   # 上色
    patch.set_facecolor(c); patch.set_alpha(0.6)                   # 填充与透明度
ax1.set_title("各渠道满意度分布"); ax1.set_ylabel("满意度（1—5）")    # 标题与轴标签
ax2 = fig2.add_subplot(gs[1, 1])                                   # 右上：收入与消费
ax2.scatter(survey["income_k"], survey["spend_k"], s=8, color="#711426", alpha=0.45)   # 散点
ax2.set_title("收入与消费"); ax2.set_xlabel("月收入（千元）"); ax2.set_ylabel("月消费（千元）")   # 标题与轴
ax3 = fig2.add_subplot(gs[1, 2])                                   # 左下：能源趋势
pivot = energy.pivot(index="month", columns="source", values="generation_twh")   # 宽表
for src, c in zip(pivot.columns, ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # 逐能源画线
    ax3.plot(pivot.index, pivot[src].rolling(3).mean(), lw=1.3, color=c, label=src)   # 3 月平滑
ax3.set_title("月度发电量（按能源）"); ax3.set_ylabel("TWh"); ax3.legend(fontsize=7.6, ncol=2)   # 标题与图例
ax4 = fig2.add_subplot(gs[1, 3])                                   # 右下：城市人口与 GDP
sc = ax4.scatter(cities["pop_wan"], cities["gdp_yi"], s=26,        # 点大小与颜色编码增速
                 c=cities["growth_pct"], cmap="RdPu")              # 连续色板
ax4.set_title("城市 GDP 与人口"); ax4.set_xlabel("人口（万人）"); ax4.set_ylabel("GDP（亿元）")   # 标题与轴
plt.colorbar(sc, ax=ax4, fraction=0.046, label="增速 %")            # 颜色条
save(fig2, "fig_dashboard.png")                                    # 保存静态仪表板
print("dashboard KPI cards:", len(kpi), "| panels: 4")              # 收尾日志
