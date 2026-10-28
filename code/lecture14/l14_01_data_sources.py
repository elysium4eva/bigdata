"""l14_01_data_sources.py -- the three layers of trade data and the firm-customs match.

Every line carries an English comment. The script builds the country panel, the firm survey and   # continuation of the call
the customs records, computes the match rate between the two firm-level sources, and draws the   # continuation of the call
"three layers" table plus the matching diagnostics that the deck uses in Part 1.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l14 import (make_country_panel, make_firms,           # shared generators
                         make_customs, save, dump, style_axes,          # shared generators
                         annotated, OKABE)                              # shared plotting helpers

# ---------------------------------------------------------------- 1) the country layer
panel = make_country_panel(60, (2000, 2019))                       # 60 countries x 20 years
dump(panel, "l14_country_panel.csv")                               # persist
print("country-years:", len(panel), "| countries:", panel["iso"].nunique())   # report

# ---------------------------------------------------------------- 2) the firm layer and the customs layer
firms = make_firms(4000)                                           # 4,000 manufacturing firms
dump(firms, "l14_firms.csv")                                       # persist
customs = make_customs(firms, 6000)                                # 6,000 customs records
dump(customs, "l14_customs.csv")                                   # persist
print("firms:", len(firms), "| customs records:", len(customs))     # report

# ---------------------------------------------------------------- 3) the match rate
matched = customs[customs["firm_id"] != ""]                        # records with a firm id
match_rate = len(matched) / len(customs)                           # share of linked records
print("match rate: %.3f (%d of %d)" % (match_rate, len(matched), len(customs)))   # report

# ---------------------------------------------------------------- 4) who gets matched
customs["linked"] = (customs["firm_id"] != "").astype(int)          # a binary match flag
by_year = customs.groupby("year")["linked"].mean().round(3)         # match rate by year
by_value = customs.groupby(pd.qcut(customs["value"], 5, duplicates="drop"))["linked"].mean().round(3)   # by size quintile
match_by_year = by_year.reset_index().rename(columns={"linked": "match_rate"})   # tidy
match_by_size = pd.DataFrame({"size_quintile": [str(i) for i in by_value.index],   # size bands
                              "match_rate": by_value.values})      # the rates
dump(match_by_year, "l14_match_by_year.csv")                       # persist
dump(match_by_size, "l14_match_by_size.csv")                       # persist
print(match_by_year.to_string(index=False))                        # report
print(match_by_size.to_string(index=False))                        # report

# ---------------------------------------------------------------- 5) survivorship: matched firms are bigger
size_all = firms["employ"].describe()                               # all firms
size_matched = firms[firms["firm_id"].isin(matched["firm_id"])]["employ"].describe()   # matched only
print("median employment: all %.0f | matched %.0f" % (size_all["50%"], size_matched["50%"]))   # report
survivor = pd.DataFrame([{"group": "全部企业", "median_employ": size_all["50%"]},   # the comparison
                         {"group": "可匹配企业", "median_employ": size_matched["50%"]}])   # matched firms
dump(survivor, "l14_survivorship.csv")                             # persist

# ---------------------------------------------------------------- 6) the three-layer figure
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))                 # two panels
axes[0].axis("off")                                                # the table needs no axes
rows = [["国别层", "CEPII / Comtrade", "国家-国家-年", "引力模型、政策评估"],   # layer, source, unit, use
        ["行业层", "BACI / WIOD", "国家-行业-年", "GVC、投入产出"],            # the middle layer
        ["企业层", "工企库 + 海关库", "企业-产品-国家-月", "异质性、贸易与生产率"]]   # the finest layer
table = axes[0].table(cellText=rows, colLabels=["层级", "来源", "观测单位", "典型用途"],   # draw it
                      loc="upper center", cellLoc="center")        # centred
table.auto_set_font_size(False)                                     # fixed font size
table.set_fontsize(8.2)                                            # readable type
table.scale(1.0, 1.9)                                              # roomier rows
axes[0].set_title("三层数据的粒度与用途")                             # panel title
axes[1].barh(["企业层", "行业层", "国别层"], [3, 2, 1], color=[OKABE[5], OKABE[2], OKABE[1]], alpha=0.9)   # a schematic ladder
axes[1].set_yticks([0, 1, 2], ["企业层", "行业层", "国别层"])        # labels
axes[1].set_xticks([])                                             # no numeric axis
style_axes(axes[1], "越往下越细，也越贵越难拿", "", "")              # shared style
for i, txt in enumerate(["样本小、匹配难", "需投入产出表", "覆盖最广、最便宜"]):   # annotate the layers
    axes[1].text(0.15, i, txt, va="center", fontsize=8.4, color="#2B2B2B")   # the note
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_data_layers.png")                                   # save the layers figure

# ---------------------------------------------------------------- 7) the matching figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].plot(by_year.index, by_year.values, "-o", ms=5, color=OKABE[4])   # match rate over time
axes[0].set_ylim(0, 1)                                             # full scale
style_axes(axes[0], "Match rate by year", "year", "share matched")   # shared style
annotated(axes[0], "口径变化会压低匹配率", "lower right")              # annotate
axes[1].bar(range(len(by_value)), by_value.values, color=OKABE[2], alpha=0.9)   # match rate by size
axes[1].set_xticks(range(len(by_value)), ["Q1\n最小", "Q2", "Q3", "Q4", "Q5\n最大"], fontsize=8)   # size bands
style_axes(axes[1], "Larger firms are easier to match", "", "match rate")   # shared style
annotated(axes[1], "幸存者偏差：匹配样本偏大", "upper left")             # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_firm_match.png")                                    # save the matching figure
summary = {"match_rate": round(float(match_rate), 3),               # the run summary
           "median_employ_all": float(size_all["50%"]),             # all firms
           "median_employ_matched": float(size_matched["50%"])}     # matched firms
dump(pd.DataFrame([summary]), "l14_match_summary.csv")             # persist
print("done: three data layers and the firm-customs match")           # closing log
