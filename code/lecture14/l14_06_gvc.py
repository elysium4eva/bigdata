"""l14_06_gvc.py -- global value chain participation from a simplified WIOT.

Every line carries an English comment. The script builds a small world input-output table,   # continuation of the call
decomposes each country's exports into domestic value added and foreign value added, computes   # continuation of the call
forward and backward participation, and plots the participation-position map the deck uses to   # continuation of the call
explain the smile curve.                                                # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l14 import (make_wiot, WIO_COUNTRIES, SECTORS, save,   # shared generators
                         dump, style_axes, annotated, OKABE)            # continuation of the call

# ---------------------------------------------------------------- 1) the table
wiot = make_wiot()                                                 # a 12 x 12 WIOT (4 countries x 3 sectors)
Z, fd, x, va = wiot["Z"], wiot["fd"], wiot["x"], wiot["va"]          # the pieces
labels = wiot["labels"]                                            # industry labels
print("industries:", len(labels), "| total output: %.0f" % x.sum())   # report
print("value added share: %.3f" % (va.sum() / x.sum()))              # report the aggregate

# ---------------------------------------------------------------- 2) gross exports and their decomposition
country_of = np.array([lab.split("-")[0] for lab in labels])        # country of each industry
sector_of = np.array([lab.split("-")[1] for lab in labels])          # sector of each industry
# Gross exports of a country = its output sold to other countries (intermediate + final).
rows = []                                                          # one row per country
for c in WIO_COUNTRIES:                                            # loop over countries
    idx_c = np.where(country_of == c)[0]                            # this country's industries
    idx_other = np.where(country_of != c)[0]                        # everyone else
    intermediate = Z[np.ix_(idx_c, idx_other)].sum()                 # intermediates sold abroad
    final_goods = fd[idx_c].sum() * (len(idx_other) / (len(idx_other) + len(idx_c)))   # a share of final demand
    exports = intermediate + final_goods                             # gross exports
    # domestic value added in exports: the share of this country's VA in its own output
    va_ratio = va[idx_c].sum() / x[idx_c].sum()                      # value-added intensity
    dva = exports * va_ratio                                        # domestic value added
    fva = exports - dva                                              # foreign value added content
    rows.append({"country": c, "exports": round(float(exports), 1),   # the row
                 "dva": round(float(dva), 1), "fva": round(float(fva), 1),   # the decomposition
                 "va_ratio": round(float(va_ratio), 3)})           # intensity
dec = pd.DataFrame(rows)                                           # assemble
dec["backward"] = (dec["fva"] / dec["exports"] * 100).round(2)       # backward participation (FVA share)
dec["forward"] = (dec["dva"] / dec["exports"] * 100).round(2)        # forward participation (DVA share)
dump(dec, "l14_gvc_decomposition.csv")                             # persist
print(dec.to_string(index=False))                                  # report

# ---------------------------------------------------------------- 3) position: upstream or downstream
# A simple position index: the share of a country's output used as intermediate input abroad
rows = []                                                          # one row per country
for c in WIO_COUNTRIES:                                            # loop over countries
    idx_c = np.where(country_of == c)[0]                            # its industries
    idx_other = np.where(country_of != c)[0]                        # the rest
    used_abroad = Z[np.ix_(idx_other, idx_c)].sum()                  # its output used abroad (as input)
    position = used_abroad / max(x[idx_c].sum(), 1e-9)               # upstream if high
    rows.append({"country": c, "position": round(float(position), 3)})   # the row
pos = pd.DataFrame(rows)                                           # assemble
dec = dec.merge(pos, on="country")                                  # join
dump(dec, "l14_gvc_position.csv")                                  # persist
print(dec[["country", "backward", "forward", "position"]].to_string(index=False))   # report

# ---------------------------------------------------------------- 4) the GVC figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
xs = np.arange(len(dec))                                           # positions
axes[0].bar(xs, dec["dva"], color=OKABE[2], alpha=0.92, label="国内增加值 DVA")   # domestic value added
axes[0].bar(xs, dec["fva"], bottom=dec["dva"], color=OKABE[5], alpha=0.92, label="国外增加值 FVA")   # foreign content
axes[0].set_xticks(xs, dec["country"])                              # labels
style_axes(axes[0], "总出口的附加值分解", "", "十亿美元")               # shared style
axes[0].legend(fontsize=7.6)                                        # compact legend
sc = axes[1].scatter(dec["backward"], dec["position"], s=180, c=dec["forward"], cmap="RdPu")   # participation vs position
for _, r in dec.iterrows():                                        # annotate each country
    axes[1].annotate(r["country"], (r["backward"], r["position"]), xytext=(6, 5), textcoords="offset points", fontsize=8)   # label
axes[1].set_xlabel("后向参与度（FVA/出口，%）")                       # axis label
axes[1].set_ylabel("上游度（中间品出口占比）")                        # axis label
axes[1].set_title("参与度 × 位置：微笑曲线的两端")                     # panel title
plt.colorbar(sc, ax=axes[1], fraction=0.046, label="前向参与度 %")      # colour bar
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_gvc.png")                                           # save the GVC figure
print("done: WIOT, value-added decomposition and GVC position")       # closing log
