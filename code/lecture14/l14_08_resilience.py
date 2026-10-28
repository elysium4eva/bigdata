"""l14_08_resilience.py -- AIS-based port network, congestion and supply-chain resilience.

Every line carries an English comment. The script builds a synthetic AIS voyage table, draws the   # continuation of the call
port network and the delay distribution, computes concentration and substitutability indicators,   # continuation of the call
and runs a simple "port closure" scenario so the deck can show what a resilience index means.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import networkx as nx                                              # the port graph
from _common_l14 import (PORTS, make_ais, save, dump, style_axes,   # shared generators
                         annotated, OKABE)                              # shared plotting helpers

# ---------------------------------------------------------------- 1) the voyages
ais = make_ais(120)                                                # 120 synthetic voyages
dump(ais, "l14_ais.csv")                                           # persist
print("voyages:", len(ais), "| mean delay: %.2f days" % ais["delay_days"].mean())   # report

# ---------------------------------------------------------------- 2) the port network
G = nx.Graph()                                                     # an undirected port graph
for _, r in ais.iterrows():                                        # loop over the voyages
    w = 1.0                                                        # each voyage is one connection
    if G.has_edge(r["origin"], r["dest"]):                         # already connected
        G[r["origin"]][r["dest"]]["weight"] += w                    # add one more voyage
    else:                                                          # a new connection
        G.add_edge(r["origin"], r["dest"], weight=w)               # create it
print("ports:", G.number_of_nodes(), "| routes:", G.number_of_edges())   # report

# ---------------------------------------------------------------- 3) concentration and resilience
share = ais["dest"].value_counts(normalize=True)                    # share of arrivals by port
hhi = float((share ** 2).sum())                                    # Herfindahl index
top1 = float(share.iloc[0])                                         # the largest port's share
substitutable = float(1 - hhi)                                      # a simple diversification measure
rows = [{"indicator": "HHI（到港集中度）", "value": round(hhi, 4)},   # the concentration index
        {"indicator": "最大港口份额", "value": round(top1, 4)},         # the dominant port
        {"indicator": "可替代性 1-HHI", "value": round(substitutable, 4)},   # diversification
        {"indicator": "平均延误（天）", "value": round(float(ais["delay_days"].mean()), 2)}]   # congestion
ind = pd.DataFrame(rows)                                           # assemble
dump(ind, "l14_resilience_indicators.csv")                         # persist
print(ind.to_string(index=False))                                  # report

# ---------------------------------------------------------------- 4) the port-closure scenario
busiest = str(share.index[0])                                      # the port to close
remaining = ais[ais["dest"] != busiest].copy()                      # voyages that still arrive
lost = float(ais.loc[ais["dest"] == busiest, "delay_days"].sum())    # delay lost with the port
rerouted = remaining["delay_days"] * 1.35                           # rerouting adds 35% delay
scenario = pd.DataFrame([                                               # assemble the frame
    {"scenario": "现状", "mean_delay": round(float(ais["delay_days"].mean()), 2)},   # baseline
    {"scenario": "关闭 %s" % busiest, "mean_delay": round(float(rerouted.mean()), 2)},   # after closure
])                                                                      # continuation of the call
dump(scenario, "l14_closure_scenario.csv")                         # persist
print(scenario.to_string(index=False))                             # report
print("port closed:", busiest, "| share of arrivals: %.2f" % top1)   # report

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))                 # two panels
pos = {p: (PORTS[p][1], PORTS[p][0]) for p in G.nodes()}            # real lon/lat coordinates
weights = np.array([G[u][v]["weight"] for u, v in G.edges()])        # the route frequencies
nx.draw_networkx_edges(G, pos, ax=axes[0], width=0.6 + 2.4 * weights / weights.max(),   # continuation of the call
                       alpha=0.55, edge_color="#8C8C8C")            # the routes
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=[220 + 900 * share.get(p, 0) for p in G.nodes()],   # continuation of the call
                       node_color=OKABE[4], alpha=0.9, linewidths=0.6, edgecolors="white")   # the ports
nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=7.2)          # port names
axes[0].set_title("港口航线网络（点大小=到港份额）")                   # panel title
axes[0].axis("off")                                                # no axes for the map
axes[1].hist(ais["delay_days"], bins=24, color=OKABE[5], alpha=0.9)   # delay distribution
axes[1].axvline(float(ais["delay_days"].mean()), ls="--", lw=1.2, color="#2B2B2B")   # the mean
style_axes(axes[1], "到港延误分布（天）", "延误天数", "航次数")          # shared style
annotated(axes[1], "HHI %.3f" % hhi, "upper right")                  # annotate the concentration
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_ais_ports.png")                                     # save the figure
print("done: AIS port network, congestion and a closure scenario")    # closing log
