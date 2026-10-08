"""l4_05_chart_zoo.py —— 图型动物园：五种单变量图 + 七类常用图型总览。

上半部分回答"同一列数据换一种画法会看到什么"，下半部分把原课件的图型清单
（柱状图、分布图、趋势图、散点图、相关矩阵、热力图、流程图/网络图）
做成一张可对照的总览图，并在最后一格放上"按问题选图"的提示。
"""
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 表格处理
import matplotlib.pyplot as plt                                    # 全部图型都用 matplotlib 绘制
from _common_l4 import RNG, OKABE, save, dump                      # 复用公共工具与色盲友好色板

# ---------------------------------------------------------------- 样本一：城市经济数据
CITY_NAMES = ["深圳", "上海", "北京", "广州", "重庆", "苏州", "成都", "杭州", "武汉", "南京",   # 城市名清单第一行
              "宁波", "青岛", "无锡", "长沙", "郑州", "佛山", "合肥", "福州", "济南", "西安",   # 城市名清单第二行
              "东莞", "泉州", "南通", "常州", "大连", "沈阳", "昆明", "南昌", "贵阳", "哈尔滨",   # 城市名清单第三行
              "厦门", "石家庄", "太原", "南宁", "徐州", "温州", "烟台", "珠海", "惠州", "中山"]   # 40 个城市
M = len(CITY_NAMES)                                               # 城市个数
base_gdp = RNG.lognormal(mean=9.3, sigma=0.45, size=M)             # 城市 GDP（亿元，右偏）
pop = RNG.lognormal(mean=6.1, sigma=0.4, size=M)                   # 常住人口（万人）
cities = pd.DataFrame({                                            # 组装城市表
    "city": CITY_NAMES,                                            # 城市名
    "province": RNG.choice(["广东", "江苏", "浙江", "山东", "四川", "湖北", "福建", "其他"], M),   # 省份（简写）
    "gdp_yi": np.round(base_gdp, 1),                               # GDP（亿元）
    "pop_wan": np.round(pop, 1),                                   # 人口（万人）
    "lat": np.round(RNG.uniform(22.5, 45.5, M), 3),                # 纬度（用于地图）
    "lon": np.round(RNG.uniform(103.0, 126.0, M), 3),              # 经度（用于地图）
    "growth_pct": np.round(RNG.normal(5.2, 1.6, M), 2),            # 增速（%）
})                                                                # 城市表定义结束
cities["gdp_per_cap"] = (cities["gdp_yi"] * 1e8 / (cities["pop_wan"] * 1e4)).round(0)   # 人均 GDP（元）
dump(cities, "l4_city_gdp.csv")                                    # 落盘城市样本
counts = pd.DataFrame({                                            # 样本二：类别计数
    "category": ["电子", "医药", "机械", "消费", "金融", "能源", "物流"],   # 七个行业
    "count": [128, 64, 92, 110, 47, 38, 55]})                      # 各行业企业数
dump(counts, "l4_category_counts.csv")                             # 落盘类别计数

# ---------------------------------------------------------------- 图 A：同一变量的五种画法
gdp = cities["gdp_yi"].values                                      # 单变量：城市 GDP
fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.5))                # 一行五格
axes[0].hist(gdp, bins=20, color="#8A0C3C", alpha=0.85)            # ① 直方图：看分布形状
axes[0].set_title("① 直方图")                                      # 子图标题
axes[0].set_xlabel("GDP（亿元）"); axes[0].set_ylabel("城市数")     # 轴标签
axes[1].boxplot(gdp, patch_artist=True,                            # ② 箱线图：看四分位与离群
                boxprops=dict(facecolor="#C9A227", alpha=0.6))     # 箱体填充
axes[1].set_title("② 箱线图")                                      # 子图标题
axes[1].set_ylabel("GDP（亿元）")                                   # 轴标签
axes[2].violinplot(gdp, showmeans=True)                            # ③ 小提琴图：看密度与形状
axes[2].set_title("③ 小提琴图")                                    # 子图标题
axes[2].set_ylabel("GDP（亿元）")                                   # 轴标签
sorted_gdp = np.sort(gdp)                                          # 升序排列用于 ECDF
ecdf = np.arange(1, len(sorted_gdp) + 1) / len(sorted_gdp)         # 累计概率
axes[3].step(sorted_gdp, ecdf, where="post", color="#2E6F8E", lw=1.8)   # ④ 经验分布函数
axes[3].set_title("④ 经验分布 ECDF")                               # 子图标题
axes[3].set_xlabel("GDP（亿元）"); axes[3].set_ylabel("累计占比")   # 轴标签
axes[4].axis("off")                                                # ⑤ 茎叶图用文字呈现
stems = {}                                                         # 按十位分组
for v in np.round(sorted_gdp, 0).astype(int):                      # 遍历每个观测
    stems.setdefault(v // 1000, []).append(v % 1000 // 100)         # 千位为茎，百位为叶
lines = [f"{s} | " + " ".join(str(x) for x in vals) for s, vals in sorted(stems.items())[:9]]   # 拼成文本行
axes[4].text(0.02, 0.95, "⑤ 茎叶图（千位 | 百位）\n\n" + "\n".join(lines),   # 左上角文本
             family=["Consolas", "Microsoft YaHei"], fontsize=7.6,   # 等宽字体 + 中文字体逐字回退
             va="top", ha="left")                                  # 顶端对齐
plt.tight_layout()                                                 # 调整五格
save(fig, "fig_univariate_zoo.png")                                # 保存：单变量五件套

# ---------------------------------------------------------------- 图 B：七类常用图型总览
fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.1))                # 两行四列共八格
top = cities.nlargest(8, "gdp_yi")                                 # 取 GDP 前八城市
axes[0, 0].bar(top["city"], top["gdp_yi"], color="#8A0C3C", alpha=0.9)   # ① 柱状图：比较大小
axes[0, 0].set_title("① 柱状图：比较")                              # 子图标题
axes[0, 0].tick_params(axis="x", rotation=45, labelsize=7.5)        # 横轴标签旋转
axes[0, 1].hist(cities["gdp_per_cap"], bins=18, color="#C9A227", alpha=0.9)   # ② 分布图
axes[0, 1].set_title("② 分布图：看形状")                            # 子图标题
axes[0, 1].set_xlabel("人均 GDP（元）", fontsize=9)                 # 轴标签
trend = cities.sort_values("gdp_yi")["gdp_yi"].values               # ③ 趋势图：用排序后的序列演示
axes[0, 2].plot(range(len(trend)), trend, "-", color="#2E6F8E", lw=1.8)   # 折线
axes[0, 2].set_title("③ 趋势图：看变化")                            # 子图标题
axes[0, 2].set_xlabel("排名（按 GDP 排序）", fontsize=9)             # 轴标签
axes[0, 3].scatter(cities["pop_wan"], cities["gdp_yi"], s=22,       # ④ 散点图：看关系
                   color="#711426", alpha=0.7)                      # 点颜色与透明度
axes[0, 3].set_title("④ 散点图：看关系")                            # 子图标题
axes[0, 3].set_xlabel("人口（万人）", fontsize=9); axes[0, 3].set_ylabel("GDP（亿元）", fontsize=9)   # 轴标签
corr_cols = ["gdp_yi", "pop_wan", "gdp_per_cap", "growth_pct"]      # ⑤ 相关矩阵：四个数值列
corr = cities[corr_cols].corr()                                    # 计算相关矩阵
im = axes[1, 0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)   # 热力图绘制
axes[1, 0].set_xticks(range(len(corr_cols)), ["GDP", "人口", "人均", "增速"], fontsize=8)   # 横轴
axes[1, 0].set_yticks(range(len(corr_cols)), ["GDP", "人口", "人均", "增速"], fontsize=8)   # 纵轴
axes[1, 0].grid(False)                                             # 热力图不需要网格
axes[1, 0].set_title("⑤ 相关矩阵：看两两关系")                      # 子图标题
axes[1, 1].barh(counts["category"], counts["count"], color=OKABE[:len(counts)], alpha=0.9)   # ⑥ 类别条形图
axes[1, 1].set_title("⑥ 条形图：看构成与排序")                      # 子图标题
axes[1, 1].tick_params(axis="y", labelsize=8)                       # 纵轴标签字号
nodes = {"上游": (0.2, 0.7), "中游": (0.5, 0.75), "下游": (0.8, 0.7),   # ⑦ 网络图：用三个节点示意
         "海外": (0.5, 0.25), "内需": (0.2, 0.3), "替代": (0.8, 0.3)}   # 六个节点坐标
edges = [("上游", "中游"), ("中游", "下游"), ("上游", "内需"),         # 供应链关系
         ("中游", "海外"), ("下游", "替代")]                          # 继续补边
for a, b in edges:                                                 # 逐条画边
    axes[1, 2].annotate("", xy=nodes[b], xytext=nodes[a],           # 从起点指向终点
                        arrowprops=dict(arrowstyle="->", color="#66717D", lw=1.4))   # 灰色箭头
for name, (x, y) in nodes.items():                                 # 逐点画节点
    axes[1, 2].scatter([x], [y], s=420, color="#8A0C3C", alpha=0.85, zorder=3)   # 节点圆点
    axes[1, 2].text(x, y, name, color="white", ha="center", va="center", fontsize=8.6, zorder=4)   # 节点文字
axes[1, 2].set_xlim(0, 1); axes[1, 2].set_ylim(0, 1)                # 固定坐标范围
axes[1, 2].axis("off")                                             # 关闭坐标轴
axes[1, 2].set_title("⑦ 网络图：看结构与流向")                      # 子图标题
axes[1, 3].axis("off")                                             # ⑧ 选图提示用文字呈现
axes[1, 3].text(0.0, 0.95,                                                 # 左上角开始
                "⑧ 按问题选图\n"                                       # ⑧ 选图提示（按问题类型逐条给出）
                "· 比较多少 → 柱状图/条形图（排序！）\n"                         # 比较类问题 → 柱状图
                "· 看分布 → 直方图/箱线图/ECDF\n"                          # 分布类问题 → 直方图/箱线图
                "· 看变化 → 折线图（注意基线与平滑）\n"                          # 时间类问题 → 折线图
                "· 看关系 → 散点图/相关矩阵（防过绘）\n"                         # 关系类问题 → 散点图
                "· 看构成 → 堆叠条/桑基图（慎用饼图）\n"                         # 构成类问题 → 堆叠条/桑基图
                "· 看空间 → choropleth/符号地图\n"                       # 空间类问题 → 地图
                "· 看结构 → 网络图（少节点、少颜色）",                           # 结构类问题 → 网络图
                fontsize=8.6, va="top", ha="left", linespacing=1.5)   # 行距稍大便于阅读
plt.tight_layout()                                                 # 调整八格
save(fig, "fig_chart_zoo.png")                                     # 保存：图型总览
print("cities:", cities.shape, "| top city:", top["city"].iloc[0])   # 收尾日志
