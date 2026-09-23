"""09_datastructure_cross_time_panel.py —— 数据结构：截面 / 时间序列 / 面板数据。"""  # 模块注释
import numpy as np                                                    # 数值计算
import pandas as pd                                                   # 表格处理
import matplotlib.pyplot as plt                                       # 绘图
from _common import RNG, save, dump_csv                               # 公共工具

# ---------------------------------------------------------------- 1) 截面数据：30 个城市某一年的人均 GDP
cities = ["深圳", "上海", "北京", "广州", "杭州", "苏州", "南京", "宁波", "无锡", "武汉",  # 城市名（截面个体）
          "成都", "长沙", "青岛", "天津", "郑州", "合肥", "福州", "济南", "西安", "重庆",  # 继续列城市
          "东莞", "佛山", "常州", "南通", "大连", "沈阳", "昆明", "南昌", "贵阳", "哈尔滨"]  # 共 30 个个体
gdp = RNG.normal(150000, 32000, len(cities)).clip(60000, 260000)        # 人均 GDP（元）：截面一次性观测
cross = pd.DataFrame({"city": cities, "year": 2025,                     # 个体标识 + 观测年份（同一年）
                      "gdp_per_capita": gdp.round(0)})                  # 被观测变量
dump_csv(cross, "structure_cross_section.csv")                          # 落盘截面数据

# ---------------------------------------------------------------- 2) 时间序列：120 个月的消费价格指数
months = pd.period_range("2016-01", periods=120, freq="M").astype(str)  # 生成 2016-01 起的 120 个月标签
trend = np.linspace(100.0, 118.0, 120)                                  # 缓慢上升的长期趋势
season = 0.7 * np.sin(2 * np.pi * (np.arange(120) % 12) / 12)           # 年度季节性波动（周期 12）
cpi = trend + season + RNG.normal(0, 0.35, 120)                         # 趋势 + 季节 + 噪声 = 观测值
ts = pd.DataFrame({"month": months, "cpi": cpi.round(2),                # 时间标签 + 观测值
                   "yoy_pct": (pd.Series(cpi).pct_change(12) * 100).round(2)})  # 同比增速（滞后 12 期）
dump_csv(ts, "structure_time_series.csv")                               # 落盘时间序列

# ---------------------------------------------------------------- 3) 面板数据：8 家公司 × 10 年
firms = ["F%02d" % i for i in range(1, 9)]                              # 8 个截面个体（公司）
years = list(range(2016, 2026))                                         # 10 个时间点（年份）
rec = []                                                                # 初始化记录列表
for fi, f in enumerate(firms):                                          # 遍历每家公司
    base = 50 + 12 * fi                                                 # 每家公司的初始营收规模不同
    growth = RNG.normal(0.08, 0.05)                                     # 每家公司的平均增速不同（个体效应）
    for y in years:                                                     # 遍历每个年份
        rec.append({"firm": f, "year": y,                          # 面板的两维标识：个体 + 时间
                    "revenue": round(base * (1 + growth) ** (y - 2016)  # 营收按公司自身增速复利增长
                                     * np.exp(RNG.normal(0, 0.06)), 2),  # 叠加年度冲击（噪声）
                    "roa": round(RNG.normal(0.06 + 0.004 * fi, 0.015), 4)})  # 另一个被解释变量：资产收益率
panel = pd.DataFrame(rec)                                               # 组装长表型面板数据
dump_csv(panel, "structure_panel.csv")                                  # 落盘面板数据

# ---------------------------------------------------------------- 4) 可视化
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9))                     # 一行三列：截面 / 时序 / 面板
top = cross.sort_values("gdp_per_capita", ascending=False).head(12)      # 截面数据按大小排序取前 12
axes[0].barh(top["city"][::-1], top["gdp_per_capita"][::-1],            # 左图：截面个体横向条形图（排序后）
            color="#8A0C3C", alpha=0.85)                                # 统一配色
axes[0].set_title("截面数据：2025 年人均 GDP（前 12 城）")               # 左图标题
axes[0].set_xlabel("元 / 人")                                           # 左图横轴含义
axes[0].tick_params(axis="y", labelsize=8)                              # 缩小 y 轴标签字号

axes[1].plot(range(120), ts["cpi"], color="#8A0C3C", lw=1.2,           # 中图：时间序列折线（长期趋势）
             label="CPI 指数")                                          # 图例：水平序列
axes[1].plot(range(120), ts["cpi"].rolling(12).mean(),                 # 叠加 12 期移动平均
             color="#C9A227", lw=2.0, label="12 期移动平均")             # 平滑后更能看清趋势
axes[1].set_title("时间序列：120 个月 CPI（趋势+季节）")                 # 中图标题
axes[1].set_xlabel("第 n 个月（2016-01 起）")                           # 中图横轴含义
axes[1].legend(fontsize=8)                                              # 显示图例

for f in firms:                                                         # 右图：逐家公司画一条线
    sub = panel[panel["firm"] == f]                                     # 取出该公司的时间序列
    axes[2].plot(sub["year"], sub["revenue"], "-o", ms=2.5, lw=1.0)     # 面板 = 多条时间序列拼在一起的“小多图”
axes[2].set_title("面板数据：8 家公司 × 10 年营收")                      # 右图标题
axes[2].set_xlabel("年份")                                              # 右图横轴含义
axes[2].set_ylabel("营业收入 / 亿元")                                    # 右图纵轴含义
plt.tight_layout()                                                      # 调整布局
save(fig, "fig_structures.png")                                         # 保存图片
