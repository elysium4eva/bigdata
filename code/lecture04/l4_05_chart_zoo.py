"""l4_05_chart_zoo.py -- chart zoo: five univariate charts plus an overview of seven common chart types.

The first half asks "what do we see when one column is drawn differently"; the second half turns the slide list
(bar, distribution, trend, scatter, correlation matrix, heatmap, flow/network charts)
into one reference overview, with a "pick a chart by question" panel in the last cell.
"""
import numpy as np                                                 # numeric computation
import pandas as pd                                                # table handling
import matplotlib.pyplot as plt                                    # every chart type is drawn with matplotlib
from _common_l4 import RNG, OKABE, save, dump                      # reuse shared helpers and the colorblind-friendly palette

# ---------------------------------------------------------------- sample 1: city economic data
CITY_NAMES = ["深圳", "上海", "北京", "广州", "重庆", "苏州", "成都", "杭州", "武汉", "南京",   # first line of the city name list
              "宁波", "青岛", "无锡", "长沙", "郑州", "佛山", "合肥", "福州", "济南", "西安",   # second line of the city name list
              "东莞", "泉州", "南通", "常州", "大连", "沈阳", "昆明", "南昌", "贵阳", "哈尔滨",   # third line of the city name list
              "厦门", "石家庄", "太原", "南宁", "徐州", "温州", "烟台", "珠海", "惠州", "中山"]   # 40 cities
M = len(CITY_NAMES)                                               # number of cities
base_gdp = RNG.lognormal(mean=9.3, sigma=0.45, size=M)             # city GDP (100M CNY, right-skewed)
pop = RNG.lognormal(mean=6.1, sigma=0.4, size=M)                   # resident population (10k people)
cities = pd.DataFrame({                                            # assemble the city table
    "city": CITY_NAMES,                                            # city name
    "province": RNG.choice(["广东", "江苏", "浙江", "山东", "四川", "湖北", "福建", "其他"], M),   # province (abbreviated)
    "gdp_yi": np.round(base_gdp, 1),                               # GDP (100M CNY)
    "pop_wan": np.round(pop, 1),                                   # population (10k people)
    "lat": np.round(RNG.uniform(22.5, 45.5, M), 3),                # latitude (for maps)
    "lon": np.round(RNG.uniform(103.0, 126.0, M), 3),              # longitude (for maps)
    "growth_pct": np.round(RNG.normal(5.2, 1.6, M), 2),            # growth rate (%)
})                                                                # end of the city table
cities["gdp_per_cap"] = (cities["gdp_yi"] * 1e8 / (cities["pop_wan"] * 1e4)).round(0)   # GDP per capita (CNY)
dump(cities, "l4_city_gdp.csv")                                    # write the city sample
counts = pd.DataFrame({                                            # sample 2: category counts
    "category": ["电子", "医药", "机械", "消费", "金融", "能源", "物流"],   # seven industries
    "count": [128, 64, 92, 110, 47, 38, 55]})                      # firm count per industry
dump(counts, "l4_category_counts.csv")                             # write the category counts

# ---------------------------------------------------------------- figure A: five ways to draw one variable
gdp = cities["gdp_yi"].values                                      # single variable: city GDP
fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.5))                # one row of five panels
axes[0].hist(gdp, bins=20, color="#8A0C3C", alpha=0.85)            # (1) histogram: distribution shape
axes[0].set_title("① 直方图")                                      # subplot title
axes[0].set_xlabel("GDP（亿元）"); axes[0].set_ylabel("城市数")     # axis labels
axes[1].boxplot(gdp, patch_artist=True,                            # (2) box plot: quartiles and outliers
                boxprops=dict(facecolor="#C9A227", alpha=0.6))     # box fill
axes[1].set_title("② 箱线图")                                      # subplot title
axes[1].set_ylabel("GDP（亿元）")                                   # axis label
axes[2].violinplot(gdp, showmeans=True)                            # (3) violin plot: density and shape
axes[2].set_title("③ 小提琴图")                                    # subplot title
axes[2].set_ylabel("GDP（亿元）")                                   # axis label
sorted_gdp = np.sort(gdp)                                          # sorted ascending for the ECDF
ecdf = np.arange(1, len(sorted_gdp) + 1) / len(sorted_gdp)         # cumulative probability
axes[3].step(sorted_gdp, ecdf, where="post", color="#2E6F8E", lw=1.8)   # (4) empirical distribution function
axes[3].set_title("④ 经验分布 ECDF")                               # subplot title
axes[3].set_xlabel("GDP（亿元）"); axes[3].set_ylabel("累计占比")   # axis labels
axes[4].axis("off")                                                # (5) stem-and-leaf is rendered as text
stems = {}                                                         # group by thousands
for v in np.round(sorted_gdp, 0).astype(int):                      # loop over every observation
    stems.setdefault(v // 1000, []).append(v % 1000 // 100)         # thousands digit as stem, hundreds digit as leaf
lines = [f"{s} | " + " ".join(str(x) for x in vals) for s, vals in sorted(stems.items())[:9]]   # join into text lines
axes[4].text(0.02, 0.95, "⑤ 茎叶图（千位 | 百位）\n\n" + "\n".join(lines),   # top-left text
             family=["Consolas", "Microsoft YaHei"], fontsize=7.6,   # monospace font plus Chinese fallback
             va="top", ha="left")                                  # top aligned
plt.tight_layout()                                                 # adjust the five panels
save(fig, "fig_univariate_zoo.png")                                # save: the univariate set of five

# ---------------------------------------------------------------- figure B: overview of seven common chart types
fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.1))                # two rows of four, eight panels in total
top = cities.nlargest(8, "gdp_yi")                                 # the eight largest cities by GDP
axes[0, 0].bar(top["city"], top["gdp_yi"], color="#8A0C3C", alpha=0.9)   # (1) bar chart: compare magnitudes
axes[0, 0].set_title("① 柱状图：比较")                              # subplot title
axes[0, 0].tick_params(axis="x", rotation=45, labelsize=7.5)        # rotate the x labels
axes[0, 1].hist(cities["gdp_per_cap"], bins=18, color="#C9A227", alpha=0.9)   # (2) distribution chart
axes[0, 1].set_title("② 分布图：看形状")                            # subplot title
axes[0, 1].set_xlabel("人均 GDP（元）", fontsize=9)                 # axis label
trend = cities.sort_values("gdp_yi")["gdp_yi"].values               # (3) trend chart: use a sorted series as a demo
axes[0, 2].plot(range(len(trend)), trend, "-", color="#2E6F8E", lw=1.8)   # line
axes[0, 2].set_title("③ 趋势图：看变化")                            # subplot title
axes[0, 2].set_xlabel("排名（按 GDP 排序）", fontsize=9)             # axis label
axes[0, 3].scatter(cities["pop_wan"], cities["gdp_yi"], s=22,       # (4) scatter plot: relationships
                   color="#711426", alpha=0.7)                      # point color and alpha
axes[0, 3].set_title("④ 散点图：看关系")                            # subplot title
axes[0, 3].set_xlabel("人口（万人）", fontsize=9); axes[0, 3].set_ylabel("GDP（亿元）", fontsize=9)   # axis labels
corr_cols = ["gdp_yi", "pop_wan", "gdp_per_cap", "growth_pct"]      # (5) correlation matrix: four numeric columns
corr = cities[corr_cols].corr()                                    # compute the correlation matrix
im = axes[1, 0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)   # draw the heatmap
axes[1, 0].set_xticks(range(len(corr_cols)), ["GDP", "人口", "人均", "增速"], fontsize=8)   # x axis
axes[1, 0].set_yticks(range(len(corr_cols)), ["GDP", "人口", "人均", "增速"], fontsize=8)   # y axis
axes[1, 0].grid(False)                                             # no grid needed on a heatmap
axes[1, 0].set_title("⑤ 相关矩阵：看两两关系")                      # subplot title
axes[1, 1].barh(counts["category"], counts["count"], color=OKABE[:len(counts)], alpha=0.9)   # (6) category bar chart
axes[1, 1].set_title("⑥ 条形图：看构成与排序")                      # subplot title
axes[1, 1].tick_params(axis="y", labelsize=8)                       # y label font size
nodes = {"上游": (0.2, 0.7), "中游": (0.5, 0.75), "下游": (0.8, 0.7),   # (7) network chart: three illustrative nodes
         "海外": (0.5, 0.25), "内需": (0.2, 0.3), "替代": (0.8, 0.3)}   # coordinates of the six nodes
edges = [("上游", "中游"), ("中游", "下游"), ("上游", "内需"),         # supply chain relations
         ("中游", "海外"), ("下游", "替代")]                          # more edges
for a, b in edges:                                                 # draw every edge
    axes[1, 2].annotate("", xy=nodes[b], xytext=nodes[a],           # arrow from start to end
                        arrowprops=dict(arrowstyle="->", color="#66717D", lw=1.4))   # grey arrow
for name, (x, y) in nodes.items():                                 # draw every node
    axes[1, 2].scatter([x], [y], s=420, color="#8A0C3C", alpha=0.85, zorder=3)   # node dot
    axes[1, 2].text(x, y, name, color="white", ha="center", va="center", fontsize=8.6, zorder=4)   # node label
axes[1, 2].set_xlim(0, 1); axes[1, 2].set_ylim(0, 1)                # fixed coordinate range
axes[1, 2].axis("off")                                             # hide the axes
axes[1, 2].set_title("⑦ 网络图：看结构与流向")                      # subplot title
axes[1, 3].axis("off")                                             # (8) chart-choice hints rendered as text
axes[1, 3].text(0.0, 0.95,                                                 # start at the top left
                "⑧ 按问题选图\n"                                       # (8) chart choice hints, one line per question type
                "· 比较多少 → 柱状图/条形图（排序！）\n"                         # comparison questions -> bar chart
                "· 看分布 → 直方图/箱线图/ECDF\n"                          # distribution questions -> histogram/box plot
                "· 看变化 → 折线图（注意基线与平滑）\n"                          # change over time -> line chart
                "· 看关系 → 散点图/相关矩阵（防过绘）\n"                         # relationships -> scatter/correlation matrix
                "· 看构成 → 堆叠条/桑基图（慎用饼图）\n"                         # composition -> stacked bar/Sankey (avoid pie)
                "· 看空间 → choropleth/符号地图\n"                       # space -> choropleth/symbol map
                "· 看结构 → 网络图（少节点、少颜色）",                           # structure -> network chart
                fontsize=8.6, va="top", ha="left", linespacing=1.5)   # slightly larger line spacing for readability
plt.tight_layout()                                                 # adjust the eight panels
save(fig, "fig_chart_zoo.png")                                     # save: chart type overview
print("cities:", cities.shape, "| top city:", top["city"].iloc[0])   # closing log
