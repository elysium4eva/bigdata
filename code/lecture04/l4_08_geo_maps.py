"""l4_08_geo_maps.py —— 地理信息可视化：choropleth、符号地图、流向图与空间自相关。

脚本先把省级与国家级指标合成出来，再演示四种地理图：
① 分级统计图（choropleth，区块填色）；② 符号地图（气泡大小/颜色编码）；
③ 贸易流向图（起点—终点连线）；④ Moran 散点图（空间自相关）。
中国省级边界文件在构建时尝试下载并缓存；若不可用，自动退化为符号地图，流程仍可跑通。
"""
import json                                                        # 解析 GeoJSON
import urllib.request                                              # 下载边界文件（可选步骤）
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 数据处理
import matplotlib.pyplot as plt                                    # 静态地图
from matplotlib.collections import PatchCollection                  # 批量绘制多边形
from matplotlib.patches import Polygon                              # 单个多边形
import plotly.express as px                                        # 世界地图（交互）
import plotly.graph_objects as go                                  # 低级接口：流向连线
from _common_l4 import RNG, DAT, INT, SEQ, save, dump              # 复用公共工具

# ---------------------------------------------------------------- 1) 合成省级与国家级指标
PROVINCES = {                                                      # 31 个省级单位的简称、经度、纬度、代码
    "北京": (116.4, 39.9, "110000"), "天津": (117.2, 39.1, "120000"), "河北": (114.5, 38.0, "130000"),   # 华北与东北各省（经度、纬度、行政区划代码）
    "山西": (112.5, 37.9, "140000"), "内蒙古": (111.7, 40.8, "150000"), "辽宁": (123.4, 41.8, "210000"),   # 华北与东北各省（续）
    "吉林": (125.3, 43.9, "220000"), "黑龙江": (126.6, 45.8, "230000"), "上海": (121.5, 31.2, "310000"),   # 东北与华东各省
    "江苏": (118.8, 32.1, "320000"), "浙江": (120.2, 30.3, "330000"), "安徽": (117.3, 31.9, "340000"),   # 华东各省
    "福建": (119.3, 26.1, "350000"), "江西": (115.9, 28.7, "360000"), "山东": (117.0, 36.7, "370000"),   # 华东与华中各省
    "河南": (113.6, 34.8, "410000"), "湖北": (114.3, 30.6, "420000"), "湖南": (113.0, 28.2, "430000"),   # 华中各省
    "广东": (113.3, 23.1, "440000"), "广西": (108.3, 22.8, "450000"), "海南": (110.3, 20.0, "460000"),   # 华南各省
    "重庆": (106.5, 29.6, "500000"), "四川": (104.1, 30.7, "510000"), "贵州": (106.7, 26.6, "520000"),   # 西南各省
    "云南": (102.7, 25.0, "530000"), "西藏": (91.1, 29.6, "540000"), "陕西": (108.9, 34.3, "610000"),   # 西南与西北各省
    "甘肃": (103.8, 36.1, "620000"), "青海": (101.8, 36.6, "630000"), "宁夏": (106.3, 38.5, "640000"),   # 西北各省
    "新疆": (87.6, 43.8, "650000"),                                # 省级单位到此结束
}                                                                 # 省级字典结束
prov = pd.DataFrame([{"province": k, "lon": v[0], "lat": v[1],      # 展开为表
                      "code": v[2]} for k, v in PROVINCES.items()])  # 名称、经纬度、代码
prov["gdp_yi"] = np.round(RNG.lognormal(mean=9.2, sigma=0.75, size=len(prov)), 1)   # 省级 GDP（亿元，右偏）
grad = np.exp(0.055 * (prov["lon"] - 100))                          # 东高西低的平滑梯度（制造空间聚集）
prov["gdp_pc"] = (55000 * grad * np.exp(RNG.normal(0, 0.10, len(prov)))).round(0)   # 人均 GDP = 梯度 × 噪声
prov["pop_wan"] = (prov["gdp_yi"] * 1e8 / (prov["gdp_pc"] * 1e4)).round(1)   # 由人均反推人口，保持口径一致
dump(prov, "l4_province_gdp.csv")                                  # 落盘省级样本

COUNTRIES = {                                                      # 24 个经济体的 ISO3 与坐标
    "CHN": ("中国", 35.0, 103.0), "USA": ("美国", 39.0, -98.0), "JPN": ("日本", 36.2, 138.2),   # 主要经济体（ISO3 → 名称、纬度、经度）
    "KOR": ("韩国", 36.5, 127.9), "DEU": ("德国", 51.2, 10.4), "FRA": ("法国", 46.6, 2.4),   # 欧洲主要经济体
    "GBR": ("英国", 54.0, -2.0), "ITA": ("意大利", 42.8, 12.8), "ESP": ("西班牙", 40.2, -3.7),   # 南欧经济体
    "NLD": ("荷兰", 52.2, 5.3), "RUS": ("俄罗斯", 60.0, 90.0), "BRA": ("巴西", -10.0, -52.0),   # 西欧与新兴经济体
    "IND": ("印度", 22.0, 79.0), "IDN": ("印度尼西亚", -2.0, 118.0), "VNM": ("越南", 14.1, 108.3),   # 亚洲新兴经济体
    "THA": ("泰国", 15.9, 101.0), "MYS": ("马来西亚", 4.2, 101.9), "SGP": ("新加坡", 1.35, 103.8),   # 东南亚经济体
    "AUS": ("澳大利亚", -25.0, 134.0), "CAN": ("加拿大", 56.0, -106.0), "MEX": ("墨西哥", 23.6, -102.5),   # 大洋洲与北美经济体
    "ZAF": ("南非", -29.0, 24.0), "SAU": ("沙特阿拉伯", 24.0, 45.0), "TUR": ("土耳其", 39.0, 35.0),   # 非洲与中东经济体
}                                                                 # 国家字典结束
cty = pd.DataFrame([{"iso3": k, "country": v[0], "lat": v[1], "lon": v[2]}   # 展开国家表
                    for k, v in COUNTRIES.items()])                # 遍历字典构造
cty["gdp_pc"] = np.round(RNG.lognormal(mean=9.6, sigma=0.85, size=len(cty)), 0)   # 人均 GDP（美元）
cty["trade_share"] = np.round(RNG.uniform(0.5, 12.0, len(cty)), 2)   # 贸易占 GDP 比重（%）
dump(cty, "l4_country_indicators.csv")                             # 落盘国家样本

# 尝试下载中国省级边界（公开 GeoJSON）；失败则退化为符号地图
GEO = DAT / "china_provinces.geojson"                              # 本地缓存路径
if not GEO.exists():                                               # 只尝试一次
    url = ("https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json")   # 公开省级边界
    try:                                                           # 网络不可用时不应中断流程
        with urllib.request.urlopen(url, timeout=20) as resp:       # 打开远程文件
            GEO.write_bytes(resp.read())                           # 写入本地缓存
        print("[geojson] downloaded:", GEO)                        # 提示下载成功
    except Exception as err:                                       # 捕获网络异常
        print("[geojson] download failed, will use a symbol map:", err)   # 提示将退化
geojson = json.loads(GEO.read_text(encoding="utf-8")) if GEO.exists() else None   # 读取缓存（若存在）
print("geojson features:", len(geojson["features"]) if geojson else 0)   # 打印要素数量

# ---------------------------------------------------------------- 2) 静态对照图：choropleth vs 符号地图
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))                # 三格对照
value = prov.set_index("province")["gdp_pc"]                       # 用省级简称索引的人均 GDP
vmin, vmax = float(value.min()), float(value.max())                # 色标范围
cmap = plt.get_cmap(SEQ)                                           # 连续色板
if geojson:                                                        # 有边界文件：画真正的 choropleth
    patches, colors = [], []                                       # 收集多边形与颜色
    for feat in geojson["features"]:                               # 遍历每个省级要素
        name = feat["properties"].get("name", "")                   # 要素名称（如"广东省"）
        short = name.replace("省", "").replace("市", "").replace("自治区", "").replace("壮族", "")   # 去掉后缀
        short = short.replace("回族", "").replace("维吾尔", "").replace("特别行政区", "")   # 继续清洗
        if short not in value.index:                               # 数据里没有该名称则跳过
            continue                                               # 进入下一个要素
        geom = feat["geometry"]                                    # 几何对象
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]   # 统一为列表
        for poly in polys:                                         # 每个多边形（含内环）
            ring = np.array(poly[0])                               # 取外环坐标
            if ring.shape[0] < 3:                                  # 少于三个点无法成面
                continue                                           # 跳过
            patches.append(Polygon(ring, closed=True))             # 构造多边形
            val = float(value[short])                              # 该省的人均 GDP
            colors.append(cmap((val - vmin) / (vmax - vmin + 1e-9)))   # 归一化后取色
    coll = PatchCollection(patches, facecolor=colors, edgecolor="white", lw=0.4)   # 批量绘制
    axes[0].add_collection(coll)                                   # 加入坐标轴
    axes[0].set_xlim(73, 136); axes[0].set_ylim(17, 54)            # 中国范围
    axes[0].set_aspect(1.2)                                        # 近似等比例（按纬度补偿）
    axes[0].set_title("① choropleth：省级人均 GDP（区块填色）")      # 子图标题
else:                                                              # 没有边界文件：退化为符号地图
    sc = axes[0].scatter(prov["lon"], prov["lat"], s=prov["gdp_pc"] / 900,   # 气泡大小编码数值
                         c=prov["gdp_pc"], cmap=SEQ, alpha=0.85, edgecolors="white", lw=0.5)   # 气泡颜色同编码
    axes[0].set_title("① 未获取边界文件：退化为符号地图")             # 子图标题
    plt.colorbar(sc, ax=axes[0], fraction=0.046, label="人均 GDP（元）")   # 颜色条
axes[0].grid(False)                                                # 地图不需要网格
axes[0].tick_params(labelsize=7.5)                                 # 刻度字号
sm = axes[1].scatter(prov["lon"], prov["lat"], s=prov["gdp_yi"] * 3.2,   # 符号地图：大小=GDP 总量
                     c=prov["gdp_pc"], cmap=SEQ, alpha=0.8, edgecolors="#1E2630", lw=0.4)   # 颜色=人均
axes[1].set_title("② 符号地图：大小=总量，颜色=人均")                # 子图标题
axes[1].set_xlim(73, 136); axes[1].set_ylim(17, 54)                # 与中国地图同范围便于对照
axes[1].set_aspect(1.2)                                            # 等比例
axes[1].grid(False)                                                # 地图不需要网格
plt.colorbar(sm, ax=axes[1], fraction=0.046, label="人均 GDP（元）")   # 颜色条
axes[2].axis("off")                                                # 第三格写选图建议
axes[2].text(0.0, 0.95,                                            # 说明文字
             "何时用哪种？\n\n"                                         # 地图选型说明（何时用 choropleth、何时用符号地图）
             "· 表达比率/强度（人均、占比、密度）→ choropleth\n"                  # 比率/强度类指标 → choropleth
             "   但大片区域会抢眼（面积=视觉权重）\n\n"                           # 提醒：大面积区域视觉权重更大
             "· 表达总量（GDP、人口、贸易额）→ 符号地图\n"                         # 总量类指标 → 符号地图
             "   避免让大省小人口扭曲结论\n\n"                                # 避免面积扭曲结论
             "· 表达联系（贸易、迁移）→ 流向图 / 网络图\n\n"                       # 联系类指标 → 流向图/网络图
             "· 分级方法：等间距 vs 分位数 vs 自然断点\n"                        # 分级方法会影响颜色分布
             "   换一种分级，颜色分布就变——必须写进图注",                           # 分级方法必须写进图注
             fontsize=9.2, va="top", ha="left", linespacing=1.5)    # 行距稍大
plt.tight_layout()                                                 # 调整三格
save(fig, "fig_map_choropleth_vs_symbol.png")                      # 保存：地图对照

# ---------------------------------------------------------------- 3) 空间自相关：Moran's I 与 Moran 散点
coords = prov[["lon", "lat"]].values                               # 省级中心点坐标
z = (value - value.mean()) / value.std(ddof=0)                     # 标准化后的人均 GDP
z = z.reindex(prov["province"]).values                             # 按表的顺序对齐
K = 4                                                              # 用 4 近邻定义空间邻接
d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)      # 两两距离平方
np.fill_diagonal(d2, np.inf)                                       # 自身不参与近邻
W = np.zeros_like(d2)                                              # 初始化权重矩阵
idx = np.argsort(d2, axis=1)[:, :K]                                # 每家取最近的 K 个邻居
for i, nbrs in enumerate(idx):                                     # 逐行填权重
    W[i, nbrs] = 1.0                                               # 邻居权重为 1
W = W / W.sum(axis=1, keepdims=True)                               # 行标准化（每行权重和为 1）
lag = W @ z                                                        # 空间滞后：邻居的加权平均
moran_i = float((z * lag).sum() / (z * z).sum())                   # Moran's I 公式
print(f"Moran's I（K={K} 近邻）= {moran_i:.3f}")                    # 打印空间自相关强度
quad = np.where((z > 0) & (lag > 0), "HH",                          # 高—高（聚集的高值区）
                np.where((z < 0) & (lag < 0), "LL",                # 低—低（聚集的低值区）
                         np.where((z > 0) & (lag < 0), "HL", "LH")))   # 高—低、低—高（异常点）
moran_tbl = pd.DataFrame({"province": prov["province"], "z": z.round(3),   # 记录标准化值与滞后
                          "spatial_lag": lag.round(3), "quadrant": quad})   # 记录象限
dump(moran_tbl, "l4_moran.csv")                                    # 落盘 Moran 结果
print("象限分布:", pd.Series(quad).value_counts().to_dict())        # 打印象限计数

fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))                # Moran 散点 + 象限地图
qcolor = {"HH": "#8A0C3C", "LL": "#2E6F8E", "HL": "#C9A227", "LH": "#66717D"}   # 四象限配色
for q, c in qcolor.items():                                        # 逐象限画点
    m = quad == q                                                  # 该象限的掩码
    axes[0].scatter(z[m], lag[m], s=34, color=c, alpha=0.85, label=f"{q}（{m.sum()}）")   # 散点
axes[0].axhline(0, color="#1E2630", lw=1.0); axes[0].axvline(0, color="#1E2630", lw=1.0)   # 象限分界
b, a = np.polyfit(z, lag, 1)                                       # 拟合斜率（即 Moran's I 的方向）
xs = np.linspace(z.min(), z.max(), 20)                             # 拟合线横坐标
axes[0].plot(xs, a + b * xs, color="#711426", lw=1.8, ls="--")      # 画拟合线
axes[0].set_title("Moran 散点图：邻居与自身是否同向")                 # 子图标题
axes[0].set_xlabel("标准化的人均 GDP（z）")                          # 横轴
axes[0].set_ylabel("空间滞后（邻居均值）")                            # 纵轴
axes[0].legend(fontsize=8.4)                                       # 图例（含各象限计数）
axes[0].text(0.02, 0.96, f"Moran's I = {moran_i:.3f}",              # 图内标注 Moran's I
             transform=axes[0].transAxes, va="top", fontsize=10,    # 位置与字号
             color="#8A0C3C", weight="bold")                       # 强调色
for q, c in qcolor.items():                                        # 右图：象限地图
    m = quad == q                                                  # 该象限掩码
    axes[1].scatter(prov["lon"][m], prov["lat"][m], s=95, color=c, alpha=0.9,   # 用颜色表示象限
                    edgecolors="white", lw=0.6, label=q)            # 白边更清晰
axes[1].set_xlim(73, 136); axes[1].set_ylim(17, 54)                # 中国范围
axes[1].set_aspect(1.2)                                            # 等比例
axes[1].grid(False)                                                # 地图不需要网格
axes[1].set_title("空间聚集格局：HH 与 LL 成片出现")                  # 子图标题
axes[1].legend(fontsize=8.4, ncol=2)                               # 两列图例
plt.tight_layout()                                                 # 调整两格
save(fig, "fig_moran_scatter.png")                                 # 保存：空间自相关

# ---------------------------------------------------------------- 4) 交互地图：世界 choropleth + 流向图
world = px.choropleth(cty, locations="iso3", color="gdp_pc",       # 世界人均 GDP 分级统计图
                      hover_name="country", locationmode="ISO-3",   # 用 ISO-3 代码定位
                      color_continuous_scale="RdPu",               # 与课件一致的连续色板
                      title="世界人均 GDP（choropleth，可缩放与悬浮）")   # 标题
world.write_html(INT / "map_world_choropleth.html", include_plotlyjs="cdn")   # 导出交互地图
print("[interactive]", INT / "map_world_choropleth.html")           # 打印产物路径
hub = "CHN"                                                        # 以中国为起点的贸易流向示意
hub_lat = float(cty.loc[cty["iso3"] == hub, "lat"].iloc[0])          # 起点纬度
hub_lon = float(cty.loc[cty["iso3"] == hub, "lon"].iloc[0])          # 起点经度
links = cty[cty["iso3"] != hub].copy()                             # 其余国家作为终点
links["value"] = (cty.loc[cty["iso3"] == hub, "gdp_pc"].iloc[0] / links["gdp_pc"]).round(2)   # 简化的流量
fig_flow = go.Figure()                                             # 用 graph_objects 手工拼图
fig_flow.add_trace(go.Scattergeo(                                  # 终点：气泡大小与颜色编码贸易占比
    lat=links["lat"], lon=links["lon"], text=links["country"],      # 坐标与悬浮文字
    marker=dict(size=links["trade_share"] * 2.2,                    # 气泡大小（按比例而非半径滥用）
                color=links["trade_share"], colorscale="RdPu",      # 颜色编码同一数值
                showscale=True, colorbar=dict(title="贸易占比 %")),  # 颜色条
    mode="markers", name="贸易伙伴"))                                 # 图例名称
for row in links.itertuples():                                     # 逐条画起点到终点的连线
    fig_flow.add_trace(go.Scattergeo(                              # 每条连线一个 trace
        lat=[hub_lat, row.lat], lon=[hub_lon, row.lon],             # 两端坐标
        mode="lines", line=dict(width=0.9, color="#8A0C3C"),        # 细红线
        opacity=0.45, showlegend=False))                            # 透明度与图例
fig_flow.update_layout(title_text="贸易联系示意（气泡=贸易占比，连线=流向）",   # 标题
                       geo=dict(showland=True, landcolor="#F7F3F2",   # 陆地底色
                                showcountries=True, countrycolor="#D6CDD0",   # 国界
                                projection_type="natural earth"))   # 投影选择（影响形状）
fig_flow.write_html(INT / "map_trade_flows.html", include_plotlyjs="cdn")   # 导出流向图
print("[interactive]", INT / "map_trade_flows.html")                # 打印产物路径
print("provinces:", len(prov), "| countries:", len(cty), "| geojson:", bool(geojson))   # 收尾日志
