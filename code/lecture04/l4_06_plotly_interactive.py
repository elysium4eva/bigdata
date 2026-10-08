"""l4_06_plotly_interactive.py —— 交互图表：探索用交互，汇报用静态。

脚本用 plotly 生成三份可在浏览器打开的交互图（折线、散点、桑基图），
导出为单文件 HTML 放进仓库；同时用 matplotlib 画一张"静态 vs 交互"的对照图，
说明两者各自适合的场景（探索 vs 定稿汇报）。
"""
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 数据表处理
import matplotlib.pyplot as plt                                    # 静态对照图
import plotly.express as px                                       # plotly 快速绘图接口
import plotly.graph_objects as go                                  # 低级接口：桑基图等专用图型
from _common_l4 import RNG, INT, DAT, save, dump                   # 复用公共工具、交互目录与数据目录

# ---------------------------------------------------------------- 1) 合成月度能耗数据
months = pd.date_range("2015-01-01", periods=132, freq="MS")        # 2015-01 起共 132 个月
SOURCES = ["煤炭", "天然气", "水电", "风电光伏"]                     # 四类能源
rows = []                                                          # 逐月逐能源记录
for i, m in enumerate(months):                                     # 遍历每个月
    season = 1 + 0.12 * np.sin(2 * np.pi * (m.month - 1) / 12)      # 季节波动（冬夏用电高）
    trend = 1 + 0.0025 * i                                         # 缓慢上升的长期趋势
    share = {"煤炭": 0.58 - 0.0016 * i, "天然气": 0.16 + 0.0006 * i,   # 煤炭占比逐年下降
             "水电": 0.14, "风电光伏": 0.12 + 0.0012 * i}            # 清洁能源占比上升
    for s in SOURCES:                                              # 遍历四类能源
        base = 320 * share[s]                                      # 该类能源的基准发电量
        value = base * season * trend * np.exp(RNG.normal(0, 0.04))   # 叠加季节、趋势与噪声
        rows.append({"month": m, "source": s, "generation_twh": round(float(value), 2)})   # 记录一行
energy = pd.DataFrame(rows)                                        # 汇总为长表
dump(energy, "l4_energy.csv")                                      # 落盘样本

# ---------------------------------------------------------------- 2) 三份交互图（导出 HTML）
line_fig = px.line(energy, x="month", y="generation_twh", color="source",   # 折线：按能源着色
                   title="月度发电量（可悬浮查看数值、可框选缩放）",         # 标题写明交互能力
                   labels={"month": "月份", "generation_twh": "发电量（TWh）", "source": "能源"})   # 中文轴名
line_fig.write_html(INT / "energy_trend.html", include_plotlyjs="cdn")   # 单文件 HTML（引用 CDN）
print("[interactive]", INT / "energy_trend.html")                  # 打印产物路径
survey = pd.read_csv(DAT / "l4_survey_raw.csv")                    # 读取调研样本（来自 l4_01）
plot_df = survey.dropna(subset=["satisfaction", "income_k", "spend_k"])   # plotly 不接受缺失：先筛掉缺失行
print("plotly 用样本:", len(plot_df), "（原样本", len(survey), "，缺失行不参与交互图）")   # 打印样本量差异
scatter_fig = px.scatter(plot_df, x="income_k", y="spend_k", color="channel",   # 散点：按渠道着色
                         size="satisfaction", hover_data=["city", "education"],   # 悬浮显示明细
                         title="收入与消费（点大小=满意度，可缩放与筛选）",       # 标题
                         labels={"income_k": "月收入（千元）", "spend_k": "月消费（千元）", "channel": "渠道"})   # 中文轴名
scatter_fig.write_html(INT / "income_spend.html", include_plotlyjs="cdn")   # 导出交互散点
print("[interactive]", INT / "income_spend.html")                   # 打印产物路径
flow = energy.groupby("source", as_index=False)["generation_twh"].sum()   # 按能源汇总
flow["sector"] = ["工业", "居民", "工业", "交通"]                    # 简化的去向（教学示意）
labels = flow["source"].tolist() + flow["sector"].tolist()          # 桑基图节点标签（能源 + 部门）
sankey_fig = go.Figure(data=[go.Sankey(                            # 用 graph_objects 构造桑基图
    node=dict(label=labels, pad=14, thickness=16,                   # 节点样式
              color=["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"] * 2),   # 节点配色
    link=dict(source=list(range(len(flow))),                        # 连线起点索引
              target=[len(flow) + i for i in range(len(flow))],      # 连线终点索引
              value=flow["generation_twh"].round(1).tolist()))])     # 连线粗细=数值
sankey_fig.update_layout(title_text="能源流向示意（桑基图：适合表达构成与流向）",   # 标题
                         font=dict(size=12))                        # 字号
sankey_fig.write_html(INT / "energy_sankey.html", include_plotlyjs="cdn")   # 导出桑基图
print("[interactive]", INT / "energy_sankey.html")                  # 打印产物路径

# ---------------------------------------------------------------- 3) 静态图：定稿汇报用
pivot = energy.pivot(index="month", columns="source", values="generation_twh")   # 宽表便于画堆叠
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))                 # 左静态、右交互示意
for s, c in zip(SOURCES, ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # 四类能源各一色
    axes[0].plot(pivot.index, pivot[s].rolling(3).mean(), lw=1.6,   # 画 3 月移动平均（抑制噪声）
                 color=c, label=s)                                 # 颜色与图例
axes[0].set_title("静态图：定稿汇报用（信息固定、可打印）")            # 子图标题
axes[0].set_xlabel("月份"); axes[0].set_ylabel("发电量（TWh）")        # 轴标签
axes[0].legend(fontsize=8.4, ncol=2)                               # 两列图例
axes[1].plot(pivot.index, pivot["风电光伏"], lw=1.6, color="#2E6F8E")   # 交互图中被"选中"的序列
axes[1].fill_between(pivot.index, pivot["风电光伏"].min() * 0.9,     # 高亮一个区间模拟框选
                     pivot["风电光伏"].max() * 1.05, alpha=0.12, color="#C9A227")   # 浅色高亮
axes[1].annotate("悬浮显示数值\n框选缩放区间\n图例点击筛系列",          # 标注交互能力
                 xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9.2,   # 文字位置
                 va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3",   # 文字底框
                                                fc="#F7F3F2", ec="#D6CDD0"))   # 框色
axes[1].set_title("交互图：探索用（可缩放、筛选、悬浮）")               # 子图标题
axes[1].set_xlabel("月份"); axes[1].set_ylabel("发电量（TWh）")        # 轴标签
plt.tight_layout()                                                 # 调整两格
save(fig, "fig_interactive_vs_static.png")                         # 保存：交互与静态对照
print("energy rows:", len(energy), "| interactive files ->", INT)   # 收尾日志
