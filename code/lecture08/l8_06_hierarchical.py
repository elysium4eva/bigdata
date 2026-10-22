"""l8_06_hierarchical.py -- hierarchical clustering, dendrograms and the single-link effect.

Every line carries an English comment. The script builds the classic chain sample where
two dense clusters are joined by a thin bridge, compares four linkages on it, and draws
the dendrograms so the single-link effect becomes visible rather than asserted.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # the dendrograms
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster   # the scipy hierarchy tools
from sklearn.cluster import AgglomerativeClustering                # the sklearn interface
from sklearn.metrics import adjusted_rand_score                    # agreement with the truth
from _common_l8 import (make_chain, make_blobs, save, dump,        # shared helpers
                        style_axes, annotated, OKABE)                   # shared plotting helpers

# ---------------------------------------------------------------- 1) the chain sample
chain = make_chain(240)                                            # two clusters joined by a bridge
dump(chain, "l8_chain.csv")                                        # persist it for the deck
X = chain[["x1", "x2"]].values                                     # feature matrix
true_labels = np.r_[np.zeros(120), np.ones(120), [0, 0, 1, 1]]      # the intended grouping
print("rows:", X.shape[0], "| bridge points: 4")                   # report

# ---------------------------------------------------------------- 2) four linkages on the chain
methods = ["single", "complete", "average", "ward"]                # the four classic linkages
rows = []                                                          # one row per linkage
for method in methods:                                             # loop over the linkages
    Z = linkage(X, method=method)                                  # build the hierarchy
    labels = fcluster(Z, t=2, criterion="maxclust")                # cut into two clusters
    ari = adjusted_rand_score(true_labels, labels)                 # agreement with the truth
    sizes = sorted(np.bincount(labels)[1:].tolist(), reverse=True)   # cluster sizes
    rows.append({"linkage": method, "ari": round(float(ari), 4),   # agreement
                 "largest_cluster": int(max(sizes)),               # size of the biggest cluster
                 "clusters": int(len(set(labels)))})               # cluster count after the cut
    print("%-8s | ARI %.3f | largest %4d" % (method, ari, max(sizes)))   # report
link_table = pd.DataFrame(rows)                                    # assemble the table
dump(link_table, "l8_linkage_compare.csv")                         # persist it
print("note: single linkage merges the bridge first, so it cannot separate the two clusters")   # teaching note

# ---------------------------------------------------------------- 3) agglo on the blob sample
blobs = make_blobs(420, 4)                                         # four well-separated blobs
Xb = blobs[["x1", "x2"]].values                                    # features
agg = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(Xb)   # Ward linkage
print("blobs: ARI vs truth %.3f" % adjusted_rand_score(blobs["label"], agg.labels_))   # report
dump(pd.DataFrame({"x1": blobs["x1"], "x2": blobs["x2"], "cluster": agg.labels_}).round(3),   # tabulate
     "l8_agglo_blobs.csv")                                         # persist

# ---------------------------------------------------------------- 4) figure: the four dendrograms
fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.8))                # one panel per linkage
for ax, method in zip(axes, methods):                              # loop over the linkages
    Z = linkage(X, method=method)                                  # rebuild the hierarchy
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0,        # draw the tree
               above_threshold_color="#66717D", link_color_func=lambda k: "#8A0C3C")   # red branches
    ax.set_title("linkage = %s" % method)                          # panel title
    ax.set_xlabel("observations")                                  # axis label
    ax.set_ylabel("merge distance")                                # axis label
    if method == "single":                                         # annotate the failure
        annotated(ax, "single link: the bridge wins", "upper right")   # note
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_dendrogram.png")                                    # save the dendrogram figure

# ---------------------------------------------------------------- 5) a second figure: the chain sample itself
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))                # two panels
axes[0].scatter(X[:, 0], X[:, 1], s=14, color=OKABE[1], alpha=0.8)   # all points
axes[0].scatter(X[-4:, 0], X[-4:, 1], s=90, marker="X", color=OKABE[5], label="bridge points")   # the bridge
style_axes(axes[0], "Two clusters joined by a thin bridge", "x1", "x2")   # shared style
axes[0].legend(fontsize=7.4, loc="upper left")                     # compact legend
for method, colour in zip(["single", "complete"], [OKABE[5], OKABE[2]]):   # two cuts compared
    Z = linkage(X, method=method)                                  # hierarchy
    labels = fcluster(Z, t=2, criterion="maxclust")                # two clusters
    axes[1].scatter(X[:, 0], X[:, 1], s=14, c=np.where(labels == 1, colour, "#B9B3B6"),   # colour by cluster
                    alpha=0.8, label=method)                       # legend label
style_axes(axes[1], "Cut into two: single vs complete", "x1", "x2")   # shared style
axes[1].legend(fontsize=7.4, loc="upper left")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_linkage_cut.png")                                   # save the cut figure
print("done: hierarchical clustering and the single-link effect")    # closing log
