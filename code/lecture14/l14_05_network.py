"""l14_05_network.py -- the trade network: representation, centrality and communities.

Every line carries an English comment. The script aggregates the bilateral panel into a trade   # continuation of the call
network for one year, computes degree, strength, betweenness and community structure with   # continuation of the call
networkx, and draws the network together with the centrality ranking the deck uses in Part 4.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import networkx as nx                                              # the graph library
from _common_l14 import (make_country_panel, make_geography,       # shared generators
                         make_trade_flows, NAME_ZH, save, dump,         # shared generators
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the network for one year
countries = make_country_panel(60, (2000, 2019))                    # country panel
geo = make_geography(60)                                           # geography
flows = make_trade_flows(countries, geo, (2000, 2019))              # dyad-year flows
year = 2019                                                        # the year we visualise
slice_ = flows[(flows["year"] == year) & (flows["trade"] > 0)].copy()   # positive flows only
print("edges in", year, ":", len(slice_))                            # report

G = nx.DiGraph()                                                   # a directed graph
for _, r in slice_.iterrows():                                     # loop over the edges
    G.add_edge(r["iso_o"], r["iso_d"], weight=float(np.log1p(r["trade"])))   # log weight keeps the plot readable
G.add_nodes_from(geo["iso_o"].unique())                             # make sure every country is a node
print("nodes:", G.number_of_nodes(), "| edges:", G.number_of_edges())   # report

# ---------------------------------------------------------------- 2) centrality measures
in_strength = {n: G.in_degree(n, weight="weight") for n in G.nodes()}   # import strength
out_strength = {n: G.out_degree(n, weight="weight") for n in G.nodes()}   # export strength
betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)   # broker position
metrics = pd.DataFrame({"iso": list(G.nodes()),                      # the node list
                        "in_strength": [in_strength[n] for n in G.nodes()],   # import strength
                        "out_strength": [out_strength[n] for n in G.nodes()],   # export strength
                        "betweenness": [betweenness[n] for n in G.nodes()]})   # brokerage
metrics["name_zh"] = metrics["iso"].map(NAME_ZH).fillna(metrics["iso"])   # Chinese labels where known
dump(metrics.round(4), "l14_network_metrics.csv")                   # persist
top_in = metrics.sort_values("in_strength", ascending=False).head(10)   # top importers
top_bt = metrics.sort_values("betweenness", ascending=False).head(10)   # top brokers
print("top importers:", top_in["iso"].tolist())                      # report
print("top brokers:", top_bt["iso"].tolist())                        # report

# ---------------------------------------------------------------- 3) communities
undirected = G.to_undirected()                                      # communities need an undirected graph
communities = list(nx.community.greedy_modularity_communities(undirected, weight="weight"))   # Louvain-style
member = {}                                                        # node -> community id
for k, block in enumerate(communities):                            # loop over the blocks
    for node in block:                                             # and their members
        member[node] = k                                           # assign the id
metrics["community"] = metrics["iso"].map(member)                   # attach to the table
dump(metrics.round(4), "l14_network_metrics.csv")                   # persist again with communities
print("communities:", len(communities), "| modularity: %.3f" % nx.community.modularity(undirected, communities, weight="weight"))   # report

# ---------------------------------------------------------------- 4) the network figure
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))                 # two panels
pos = nx.spring_layout(G, seed=7, k=0.35)                           # a stable layout
sizes = np.array([out_strength[n] for n in G.nodes()])               # node size by exports
sizes = 60 + 260 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1e-9)   # rescale
colours = [OKABE[member[n] % len(OKABE)] for n in G.nodes()]         # colour by community
nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.18, width=0.35, arrows=False, edge_color="#8C8C8C")   # the edges
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=sizes, node_color=colours, alpha=0.92,   # continuation of the call
                       linewidths=0.4, edgecolors="white")          # the nodes
labels = {n: NAME_ZH.get(n, n) for n in list(metrics.sort_values("out_strength", ascending=False)["iso"])[:18]}   # label the big ones
nx.draw_networkx_labels(G, pos, labels=labels, ax=axes[0], font_size=6.4)   # draw the labels
axes[0].axis("off")                                                # no axes for a network
axes[0].set_title("贸易网络（%d，按出口强度定大小，颜色=社区）" % year)   # panel title
axes[1].barh(top_in["iso"][::-1], top_in["in_strength"][::-1], color=OKABE[4], alpha=0.9)   # import strength
style_axes(axes[1], "Top import strength (log 贸易额 之和)", "strength", "")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_trade_network.png")                                 # save the network figure

# ---------------------------------------------------------------- 5) the centrality figure
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))                 # two panels
axes[0].barh(top_bt["iso"][::-1], top_bt["betweenness"][::-1], color=OKABE[5], alpha=0.9)   # brokerage
style_axes(axes[0], "中介中心性：谁在中间传话", "betweenness", "")      # shared style
sc = axes[1].scatter(metrics["out_strength"], metrics["in_strength"], s=28,   # continuation of the call
                     c=metrics["betweenness"], cmap="RdPu", alpha=0.9)   # strength vs brokerage
axes[1].set_xlabel("出口强度")                                       # axis label
axes[1].set_ylabel("进口强度")                                       # axis label
axes[1].set_title("Strength versus brokerage")                      # panel title
plt.colorbar(sc, ax=axes[1], fraction=0.046, label="betweenness")    # colour bar
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_centrality.png")                                    # save the centrality figure
print("done: trade network, centrality and communities")              # closing log
