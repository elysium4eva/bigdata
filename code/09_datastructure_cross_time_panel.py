"""09_datastructure_cross_time_panel.py -- data structures: cross section, time series and panel."""  # module comment
import numpy as np                                                    # numeric computation
import pandas as pd                                                   # table handling
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, save, dump_csv                               # shared helpers

# ---------------------------------------------------------------- 1) cross section: GDP per capita of 30 cities in one year
cities = ["深圳", "上海", "北京", "广州", "杭州", "苏州", "南京", "宁波", "无锡", "武汉",  # city names, the cross-sectional units
          "成都", "长沙", "青岛", "天津", "郑州", "合肥", "福州", "济南", "西安", "重庆",  # continue the city list
          "东莞", "佛山", "常州", "南通", "大连", "沈阳", "昆明", "南昌", "贵阳", "哈尔滨"]  # 30 units in total
gdp = RNG.normal(150000, 32000, len(cities)).clip(60000, 260000)        # GDP per capita in yuan, a single cross-section
cross = pd.DataFrame({"city": cities, "year": 2025,                     # unit id plus the observation year, the same year
                      "gdp_per_capita": gdp.round(0)})                  # the observed variable
dump_csv(cross, "structure_cross_section.csv")                          # dump the cross-sectional data

# ---------------------------------------------------------------- 2) time series: 120 months of consumer price index
months = pd.period_range("2016-01", periods=120, freq="M").astype(str)  # build 120 monthly labels starting at 2016-01
trend = np.linspace(100.0, 118.0, 120)                                  # slowly rising long-run trend
season = 0.7 * np.sin(2 * np.pi * (np.arange(120) % 12) / 12)           # yearly seasonal fluctuation, period 12
cpi = trend + season + RNG.normal(0, 0.35, 120)                         # trend plus season plus noise gives the observation
ts = pd.DataFrame({"month": months, "cpi": cpi.round(2),                # time label plus observed value
                   "yoy_pct": (pd.Series(cpi).pct_change(12) * 100).round(2)})  # year-on-year growth, lagged 12 periods
dump_csv(ts, "structure_time_series.csv")                               # dump the time series

# ---------------------------------------------------------------- 3) panel data: 8 firms over 10 years
firms = ["F%02d" % i for i in range(1, 9)]                              # 8 cross-sectional units, the firms
years = list(range(2016, 2026))                                         # 10 time points, the years
rec = []                                                                # initialise the record list
for fi, f in enumerate(firms):                                          # loop over the firms
    base = 50 + 12 * fi                                                 # each firm starts at a different revenue level
    growth = RNG.normal(0.08, 0.05)                                     # each firm has its own growth rate, a firm effect
    for y in years:                                                     # loop over the years
        rec.append({"firm": f, "year": y,                          # the two panel dimensions: firm and year
                    "revenue": round(base * (1 + growth) ** (y - 2016)  # revenue compounds at the firm growth rate
                                     * np.exp(RNG.normal(0, 0.06)), 2),  # add a yearly shock, that is noise
                    "roa": round(RNG.normal(0.06 + 0.004 * fi, 0.015), 4)})  # another dependent variable: return on assets
panel = pd.DataFrame(rec)                                               # assemble the panel in long format
dump_csv(panel, "structure_panel.csv")                                  # dump the panel data

# ---------------------------------------------------------------- 4) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9))                     # one row, three columns: cross section / series / panel
top = cross.sort_values("gdp_per_capita", ascending=False).head(12)      # sort the cross section and take the top 12
axes[0].barh(top["city"][::-1], top["gdp_per_capita"][::-1],            # left: horizontal bars of the sorted cross section
            color="#8A0C3C", alpha=0.85)                                # one consistent colour
axes[0].set_title("截面数据：2025 年人均 GDP（前 12 城）")               # left panel title
axes[0].set_xlabel("元 / 人")                                           # left panel x label
axes[0].tick_params(axis="y", labelsize=8)                              # shrink the y tick label size

axes[1].plot(range(120), ts["cpi"], color="#8A0C3C", lw=1.2,           # middle: time series line showing the trend
             label="CPI 指数")                                          # legend: the level series
axes[1].plot(range(120), ts["cpi"].rolling(12).mean(),                 # overlay a 12 period moving average
             color="#C9A227", lw=2.0, label="12 期移动平均")             # smoothing makes the trend easier to read
axes[1].set_title("时间序列：120 个月 CPI（趋势+季节）")                 # middle panel title
axes[1].set_xlabel("第 n 个月（2016-01 起）")                           # middle panel x label
axes[1].legend(fontsize=8)                                              # show the legend

for f in firms:                                                         # right: one line per firm
    sub = panel[panel["firm"] == f]                                     # take the time series of this firm
    axes[2].plot(sub["year"], sub["revenue"], "-o", ms=2.5, lw=1.0)     # a panel is many time series drawn side by side
axes[2].set_title("面板数据：8 家公司 × 10 年营收")                      # right panel title
axes[2].set_xlabel("年份")                                              # right panel x label
axes[2].set_ylabel("营业收入 / 亿元")                                    # right panel y label
plt.tight_layout()                                                      # adjust the layout
save(fig, "fig_structures.png")                                         # save the figure
