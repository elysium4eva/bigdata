"""l4_08_geo_maps.py -- geographic visualization: choropleth, symbol map, flow map and spatial autocorrelation.

The script first synthesises province and country indicators, then demonstrates four geographic charts:
(1) choropleth (filled areas); (2) symbol map (bubble size and color encoding);
(3) trade flow map (origin-destination lines); (4) Moran scatter plot (spatial autocorrelation).
The China province boundary file is downloaded and cached when possible; otherwise it falls back to a symbol map.
"""
import json                                                        # parse GeoJSON
import urllib.request                                              # download the boundary file (optional step)
import numpy as np                                                 # numeric computation
import pandas as pd                                                # data processing
import matplotlib.pyplot as plt                                    # static maps
from matplotlib.collections import PatchCollection                  # draw polygons in batches
from matplotlib.patches import Polygon                              # a single polygon
import plotly.express as px                                        # world map (interactive)
import plotly.graph_objects as go                                  # low-level API: flow lines
from _common_l4 import RNG, DAT, INT, SEQ, save, dump              # reuse shared helpers

# ---------------------------------------------------------------- 1) synthetic province and country indicators
PROVINCES = {                                                      # 31 province-level units: abbreviation, longitude, latitude, code
    "北京": (116.4, 39.9, "110000"), "天津": (117.2, 39.1, "120000"), "河北": (114.5, 38.0, "130000"),   # provinces of north and northeast China (lon, lat, administrative code)
    "山西": (112.5, 37.9, "140000"), "内蒙古": (111.7, 40.8, "150000"), "辽宁": (123.4, 41.8, "210000"),   # north and northeast China (continued)
    "吉林": (125.3, 43.9, "220000"), "黑龙江": (126.6, 45.8, "230000"), "上海": (121.5, 31.2, "310000"),   # northeast and east China provinces
    "江苏": (118.8, 32.1, "320000"), "浙江": (120.2, 30.3, "330000"), "安徽": (117.3, 31.9, "340000"),   # east China provinces
    "福建": (119.3, 26.1, "350000"), "江西": (115.9, 28.7, "360000"), "山东": (117.0, 36.7, "370000"),   # east and central China provinces
    "河南": (113.6, 34.8, "410000"), "湖北": (114.3, 30.6, "420000"), "湖南": (113.0, 28.2, "430000"),   # central China provinces
    "广东": (113.3, 23.1, "440000"), "广西": (108.3, 22.8, "450000"), "海南": (110.3, 20.0, "460000"),   # south China provinces
    "重庆": (106.5, 29.6, "500000"), "四川": (104.1, 30.7, "510000"), "贵州": (106.7, 26.6, "520000"),   # southwest China provinces
    "云南": (102.7, 25.0, "530000"), "西藏": (91.1, 29.6, "540000"), "陕西": (108.9, 34.3, "610000"),   # southwest and northwest China provinces
    "甘肃": (103.8, 36.1, "620000"), "青海": (101.8, 36.6, "630000"), "宁夏": (106.3, 38.5, "640000"),   # northwest China provinces
    "新疆": (87.6, 43.8, "650000"),                                # end of the province-level units
}                                                                 # end of the province dict
prov = pd.DataFrame([{"province": k, "lon": v[0], "lat": v[1],      # expand into a table
                      "code": v[2]} for k, v in PROVINCES.items()])  # name, coordinates and code
prov["gdp_yi"] = np.round(RNG.lognormal(mean=9.2, sigma=0.75, size=len(prov)), 1)   # provincial GDP (100M CNY, right-skewed)
grad = np.exp(0.055 * (prov["lon"] - 100))                          # smooth east-high west-low gradient (creates spatial clustering)
prov["gdp_pc"] = (55000 * grad * np.exp(RNG.normal(0, 0.10, len(prov)))).round(0)   # GDP per capita = gradient x noise
prov["pop_wan"] = (prov["gdp_yi"] * 1e8 / (prov["gdp_pc"] * 1e4)).round(1)   # back out population to keep one definition
dump(prov, "l4_province_gdp.csv")                                  # write the province sample

COUNTRIES = {                                                      # 24 economies: ISO3 code and coordinates
    "CHN": ("中国", 35.0, 103.0), "USA": ("美国", 39.0, -98.0), "JPN": ("日本", 36.2, 138.2),   # major economies (ISO3 -> name, latitude, longitude)
    "KOR": ("韩国", 36.5, 127.9), "DEU": ("德国", 51.2, 10.4), "FRA": ("法国", 46.6, 2.4),   # major European economies
    "GBR": ("英国", 54.0, -2.0), "ITA": ("意大利", 42.8, 12.8), "ESP": ("西班牙", 40.2, -3.7),   # southern European economies
    "NLD": ("荷兰", 52.2, 5.3), "RUS": ("俄罗斯", 60.0, 90.0), "BRA": ("巴西", -10.0, -52.0),   # western European and emerging economies
    "IND": ("印度", 22.0, 79.0), "IDN": ("印度尼西亚", -2.0, 118.0), "VNM": ("越南", 14.1, 108.3),   # emerging Asian economies
    "THA": ("泰国", 15.9, 101.0), "MYS": ("马来西亚", 4.2, 101.9), "SGP": ("新加坡", 1.35, 103.8),   # southeast Asian economies
    "AUS": ("澳大利亚", -25.0, 134.0), "CAN": ("加拿大", 56.0, -106.0), "MEX": ("墨西哥", 23.6, -102.5),   # Oceania and North American economies
    "ZAF": ("南非", -29.0, 24.0), "SAU": ("沙特阿拉伯", 24.0, 45.0), "TUR": ("土耳其", 39.0, 35.0),   # African and Middle Eastern economies
}                                                                 # end of the country dict
cty = pd.DataFrame([{"iso3": k, "country": v[0], "lat": v[1], "lon": v[2]}   # expand into a country table
                    for k, v in COUNTRIES.items()])                # build by iterating the dict
cty["gdp_pc"] = np.round(RNG.lognormal(mean=9.6, sigma=0.85, size=len(cty)), 0)   # GDP per capita (USD)
cty["trade_share"] = np.round(RNG.uniform(0.5, 12.0, len(cty)), 2)   # trade as a share of GDP (%)
dump(cty, "l4_country_indicators.csv")                             # write the country sample

# try to download province boundaries; fall back to a symbol map on failure
GEO = DAT / "china_provinces.geojson"                              # local cache path
if not GEO.exists():                                               # try only once
    url = ("https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json")   # public province boundaries
    try:                                                           # a network failure must not stop the pipeline
        with urllib.request.urlopen(url, timeout=20) as resp:       # open the remote file
            GEO.write_bytes(resp.read())                           # write the local cache
        print("[geojson] downloaded:", GEO)                        # report the download
    except Exception as err:                                       # catch network errors
        print("[geojson] download failed, will use a symbol map:", err)   # report the fallback
geojson = json.loads(GEO.read_text(encoding="utf-8")) if GEO.exists() else None   # read the cache if it exists
print("geojson features:", len(geojson["features"]) if geojson else 0)   # print the feature count

# ---------------------------------------------------------------- 2) static comparison: choropleth vs symbol map
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))                # three panels for comparison
value = prov.set_index("province")["gdp_pc"]                       # GDP per capita indexed by province abbreviation
vmin, vmax = float(value.min()), float(value.max())                # color scale range
cmap = plt.get_cmap(SEQ)                                           # continuous palette
if geojson:                                                        # boundary file present: draw a real choropleth
    patches, colors = [], []                                       # collect polygons and colors
    for feat in geojson["features"]:                               # loop over every province feature
        name = feat["properties"].get("name", "")                   # feature name (for example "Guangdong Province")
        short = name.replace("省", "").replace("市", "").replace("自治区", "").replace("壮族", "")   # strip the suffix
        short = short.replace("回族", "").replace("维吾尔", "").replace("特别行政区", "")   # keep cleaning the name
        if short not in value.index:                               # skip names that are not in the data
            continue                                               # go to the next feature
        geom = feat["geometry"]                                    # geometry object
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]   # normalise to a list
        for poly in polys:                                         # every polygon (including inner rings)
            ring = np.array(poly[0])                               # take the outer ring coordinates
            if ring.shape[0] < 3:                                  # fewer than three points cannot form an area
                continue                                           # skip it
            patches.append(Polygon(ring, closed=True))             # build the polygon
            val = float(value[short])                              # GDP per capita of that province
            colors.append(cmap((val - vmin) / (vmax - vmin + 1e-9)))   # normalise and pick the color
    coll = PatchCollection(patches, facecolor=colors, edgecolor="white", lw=0.4)   # draw all polygons in one batch
    axes[0].add_collection(coll)                                   # add to the axes
    axes[0].set_xlim(73, 136); axes[0].set_ylim(17, 54)            # China extent
    axes[0].set_aspect(1.2)                                        # roughly equal scale (latitude compensation)
    axes[0].set_title("① choropleth：省级人均 GDP（区块填色）")      # subplot title
else:                                                              # no boundary file: fall back to a symbol map
    sc = axes[0].scatter(prov["lon"], prov["lat"], s=prov["gdp_pc"] / 900,   # bubble size encodes the value
                         c=prov["gdp_pc"], cmap=SEQ, alpha=0.85, edgecolors="white", lw=0.5)   # bubble color encodes the same value
    axes[0].set_title("① 未获取边界文件：退化为符号地图")             # subplot title
    plt.colorbar(sc, ax=axes[0], fraction=0.046, label="人均 GDP（元）")   # color bar
axes[0].grid(False)                                                # maps need no grid
axes[0].tick_params(labelsize=7.5)                                 # tick font size
sm = axes[1].scatter(prov["lon"], prov["lat"], s=prov["gdp_yi"] * 3.2,   # symbol map: size = total GDP
                     c=prov["gdp_pc"], cmap=SEQ, alpha=0.8, edgecolors="#1E2630", lw=0.4)   # color = per capita
axes[1].set_title("② 符号地图：大小=总量，颜色=人均")                # subplot title
axes[1].set_xlim(73, 136); axes[1].set_ylim(17, 54)                # same extent as the China map for comparison
axes[1].set_aspect(1.2)                                            # equal scale
axes[1].grid(False)                                                # maps need no grid
plt.colorbar(sm, ax=axes[1], fraction=0.046, label="人均 GDP（元）")   # color bar
axes[2].axis("off")                                                # third panel holds the chart choice advice
axes[2].text(0.0, 0.95,                                            # explanatory text
             "何时用哪种？\n\n"                                         # map choice notes (when to use choropleth, when a symbol map)
             "· 表达比率/强度（人均、占比、密度）→ choropleth\n"                  # ratios and intensity -> choropleth
             "   但大片区域会抢眼（面积=视觉权重）\n\n"                           # but large areas dominate (area = visual weight)
             "· 表达总量（GDP、人口、贸易额）→ 符号地图\n"                         # totals -> symbol map
             "   避免让大省小人口扭曲结论\n\n"                                # avoids distorting conclusions by area
             "· 表达联系（贸易、迁移）→ 流向图 / 网络图\n\n"                       # connections -> flow or network map
             "· 分级方法：等间距 vs 分位数 vs 自然断点\n"                        # the classification method drives the colors
             "   换一种分级，颜色分布就变——必须写进图注",                           # changing classes changes colors: state it in the caption
             fontsize=9.2, va="top", ha="left", linespacing=1.5)    # slightly larger line spacing
plt.tight_layout()                                                 # adjust the three panels
save(fig, "fig_map_choropleth_vs_symbol.png")                      # save: map comparison

# ---------------------------------------------------------------- 3) spatial autocorrelation: Moran's I and Moran scatter
coords = prov[["lon", "lat"]].values                               # province centroid coordinates
z = (value - value.mean()) / value.std(ddof=0)                     # standardised GDP per capita
z = z.reindex(prov["province"]).values                             # align to the table order
K = 4                                                              # define neighbours with the 4 nearest units
d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)      # pairwise squared distances
np.fill_diagonal(d2, np.inf)                                       # a unit is not its own neighbour
W = np.zeros_like(d2)                                              # initialise the weight matrix
idx = np.argsort(d2, axis=1)[:, :K]                                # take the K nearest neighbours
for i, nbrs in enumerate(idx):                                     # fill the weights row by row
    W[i, nbrs] = 1.0                                               # neighbour weight is 1
W = W / W.sum(axis=1, keepdims=True)                               # row standardise (each row sums to 1)
lag = W @ z                                                        # spatial lag: weighted average of neighbours
moran_i = float((z * lag).sum() / (z * z).sum())                   # Moran's I formula
print(f"Moran's I（K={K} 近邻）= {moran_i:.3f}")                    # print the spatial autocorrelation strength
quad = np.where((z > 0) & (lag > 0), "HH",                          # high-high (clustered high values)
                np.where((z < 0) & (lag < 0), "LL",                # low-low (clustered low values)
                         np.where((z > 0) & (lag < 0), "HL", "LH")))   # high-low and low-high (outliers)
moran_tbl = pd.DataFrame({"province": prov["province"], "z": z.round(3),   # record standardised values and lags
                          "spatial_lag": lag.round(3), "quadrant": quad})   # record the quadrants
dump(moran_tbl, "l4_moran.csv")                                    # write the Moran results
print("象限分布:", pd.Series(quad).value_counts().to_dict())        # print the quadrant counts

fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))                # Moran scatter plus quadrant map
qcolor = {"HH": "#8A0C3C", "LL": "#2E6F8E", "HL": "#C9A227", "LH": "#66717D"}   # colors for the four quadrants
for q, c in qcolor.items():                                        # draw points quadrant by quadrant
    m = quad == q                                                  # mask for that quadrant
    axes[0].scatter(z[m], lag[m], s=34, color=c, alpha=0.85, label=f"{q}（{m.sum()}）")   # scatter
axes[0].axhline(0, color="#1E2630", lw=1.0); axes[0].axvline(0, color="#1E2630", lw=1.0)   # quadrant dividers
b, a = np.polyfit(z, lag, 1)                                       # fitted slope (the direction of Moran's I)
xs = np.linspace(z.min(), z.max(), 20)                             # x coordinates of the fit line
axes[0].plot(xs, a + b * xs, color="#711426", lw=1.8, ls="--")      # draw the fit line
axes[0].set_title("Moran 散点图：邻居与自身是否同向")                 # subplot title
axes[0].set_xlabel("标准化的人均 GDP（z）")                          # x axis
axes[0].set_ylabel("空间滞后（邻居均值）")                            # y axis
axes[0].legend(fontsize=8.4)                                       # legend (with the quadrant counts)
axes[0].text(0.02, 0.96, f"Moran's I = {moran_i:.3f}",              # in-figure Moran's I label
             transform=axes[0].transAxes, va="top", fontsize=10,    # position and font size
             color="#8A0C3C", weight="bold")                       # emphasis color
for q, c in qcolor.items():                                        # right figure: quadrant map
    m = quad == q                                                  # mask for that quadrant
    axes[1].scatter(prov["lon"][m], prov["lat"][m], s=95, color=c, alpha=0.9,   # color shows the quadrant
                    edgecolors="white", lw=0.6, label=q)            # white edges look cleaner
axes[1].set_xlim(73, 136); axes[1].set_ylim(17, 54)                # China extent
axes[1].set_aspect(1.2)                                            # equal scale
axes[1].grid(False)                                                # maps need no grid
axes[1].set_title("空间聚集格局：HH 与 LL 成片出现")                  # subplot title
axes[1].legend(fontsize=8.4, ncol=2)                               # two-column legend
plt.tight_layout()                                                 # adjust the two panels
save(fig, "fig_moran_scatter.png")                                 # save: spatial autocorrelation

# ---------------------------------------------------------------- 4) interactive maps: world choropleth and flow map
world = px.choropleth(cty, locations="iso3", color="gdp_pc",       # world GDP per capita choropleth
                      hover_name="country", locationmode="ISO-3",   # locate by ISO-3 code
                      color_continuous_scale="RdPu",               # palette consistent with the slides
                      title="世界人均 GDP（choropleth，可缩放与悬浮）")   # title
world.write_html(INT / "map_world_choropleth.html", include_plotlyjs="cdn")   # export the interactive map
print("[interactive]", INT / "map_world_choropleth.html")           # print the output path
hub = "CHN"                                                        # trade flows originating from China
hub_lat = float(cty.loc[cty["iso3"] == hub, "lat"].iloc[0])          # origin latitude
hub_lon = float(cty.loc[cty["iso3"] == hub, "lon"].iloc[0])          # origin longitude
links = cty[cty["iso3"] != hub].copy()                             # the other countries are destinations
links["value"] = (cty.loc[cty["iso3"] == hub, "gdp_pc"].iloc[0] / links["gdp_pc"]).round(2)   # simplified flow value
fig_flow = go.Figure()                                             # build the figure by hand with graph_objects
fig_flow.add_trace(go.Scattergeo(                                  # destinations: bubble size and color encode trade share
    lat=links["lat"], lon=links["lon"], text=links["country"],      # coordinates and hover text
    marker=dict(size=links["trade_share"] * 2.2,                    # bubble size (scaled, not abusing the radius)
                color=links["trade_share"], colorscale="RdPu",      # color encodes the same value
                showscale=True, colorbar=dict(title="贸易占比 %")),  # color bar
    mode="markers", name="贸易伙伴"))                                 # legend name
for row in links.itertuples():                                     # draw one origin-destination line each
    fig_flow.add_trace(go.Scattergeo(                              # one trace per line
        lat=[hub_lat, row.lat], lon=[hub_lon, row.lon],             # the two end coordinates
        mode="lines", line=dict(width=0.9, color="#8A0C3C"),        # thin red line
        opacity=0.45, showlegend=False))                            # alpha and legend
fig_flow.update_layout(title_text="贸易联系示意（气泡=贸易占比，连线=流向）",   # title
                       geo=dict(showland=True, landcolor="#F7F3F2",   # land color
                                showcountries=True, countrycolor="#D6CDD0",   # country borders
                                projection_type="natural earth"))   # projection choice (affects the shape)
fig_flow.write_html(INT / "map_trade_flows.html", include_plotlyjs="cdn")   # export the flow map
print("[interactive]", INT / "map_trade_flows.html")                # print the output path
print("provinces:", len(prov), "| countries:", len(cty), "| geojson:", bool(geojson))   # closing log
