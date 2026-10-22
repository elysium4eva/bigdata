"""l8_07_dbscan.py -- density-based clustering: DBSCAN, HDBSCAN and the noise label.

Every line carries an English comment. The script builds a sample with two rings, one
blob and uniform noise, then compares DBSCAN and HDBSCAN with k-means so the deck can
show which method recovers the density structure and which one shreds it.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # four panels
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans               # the three methods
from sklearn.metrics import adjusted_rand_score                    # external check where labels exist
from _common_l8 import (make_density, save, dump, style_axes,      # shared helpers
                        annotated, OKABE)                               # shared plotting helpers

# ---------------------------------------------------------------- 1) the density sample
data = make_density("rings", 600)                                  # two rings, one blob and noise
dump(data, "l8_density.csv")                                       # persist it
X = data[["x1", "x2"]].values                                      # feature matrix
print("rows:", X.shape[0], "| no labels given")                    # report

# ---------------------------------------------------------------- 2) DBSCAN over a small grid
rows = []                                                          # one row per parameter pair
for eps in [0.3, 0.6, 0.9, 1.3]:                                   # neighbourhood radius
    for min_pts in [3, 6, 12]:                                     # minimum points per core
        model = DBSCAN(eps=eps, min_samples=min_pts).fit(X)        # fit
        labels = model.labels_                                     # -1 marks noise
        n_clusters = len(set(labels) - {-1})                       # number of clusters
        noise_share = float(np.mean(labels == -1))                 # share of noise points
        rows.append({"eps": eps, "min_samples": min_pts, "clusters": n_clusters,   # the two parameters and the count
                     "noise_share": round(noise_share, 3)})        # the noise share
        print("eps=%.1f minPts=%2d | clusters %d | noise %.2f" % (eps, min_pts, n_clusters, noise_share))   # report
grid = pd.DataFrame(rows)                                          # assemble the table
dump(grid, "l8_dbscan_grid.csv")                                   # persist it

# ---------------------------------------------------------------- 3) a good setting in detail
best = DBSCAN(eps=0.9, min_samples=6).fit(X)                       # a workable setting
labels = best.labels_                                              # cluster labels, -1 is noise
core_mask = np.zeros(len(X), dtype=bool)                           # core point mask
core_mask[best.core_sample_indices_] = True                        # mark the core points reported by sklearn
border_mask = (~core_mask) & (labels != -1)                        # border points belong to a cluster but are not core
noise_mask = labels == -1                                          # noise points
print("core %d | border %d | noise %d" % (core_mask.sum(), border_mask.sum(), noise_mask.sum()))   # report
dump(pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "cluster": labels,   # tabulate the labels
                   "core": core_mask.astype(int), "noise": noise_mask.astype(int)}).round(3),   # flags
     "l8_dbscan_labels.csv")                                       # persist

# ---------------------------------------------------------------- 4) HDBSCAN and k-means for contrast
hdb = HDBSCAN(min_cluster_size=25, min_samples=8).fit(X)           # HDBSCAN needs no radius
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)       # k-means must be told k
print("HDBSCAN clusters %d | noise %.2f" % (len(set(hdb.labels_) - {-1}), np.mean(hdb.labels_ == -1)))   # report
dump(pd.DataFrame({"method": ["dbscan", "hdbscan", "kmeans"],      # method names
                   "clusters": [len(set(labels) - {-1}), len(set(hdb.labels_) - {-1}), 3],   # cluster counts
                   "noise_share": [round(float(np.mean(labels == -1)), 3),   # DBSCAN noise
                                   round(float(np.mean(hdb.labels_ == -1)), 3), 0.0]}),   # HDBSCAN noise
     "l8_density_compare.csv")                                     # persist the comparison

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.7))                # four panels
sc = axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="RdPu", s=14, alpha=0.85)   # DBSCAN labels
axes[0].scatter(X[noise_mask, 0], X[noise_mask, 1], s=64, marker="x", color="#1E2630", label="noise")   # noise marks
style_axes(axes[0], "DBSCAN (eps=0.9, minPts=6)", "x1", "x2")       # shared style
axes[0].legend(fontsize=7.0, loc="upper left")                     # compact legend
axes[1].scatter(X[core_mask, 0], X[core_mask, 1], s=16, color=OKABE[2], label="core")   # core points
axes[1].scatter(X[border_mask, 0], X[border_mask, 1], s=16, color=OKABE[3], label="border")   # border points
axes[1].scatter(X[noise_mask, 0], X[noise_mask, 1], s=60, marker="x", color="#1E2630", label="noise")   # noise
style_axes(axes[1], "Core, border and noise points", "x1", "x2")    # shared style
axes[1].legend(fontsize=7.0, loc="upper left")                     # compact legend
axes[2].scatter(X[:, 0], X[:, 1], c=hdb.labels_, cmap="RdPu", s=14, alpha=0.85)   # HDBSCAN labels
axes[2].scatter(X[hdb.labels_ == -1, 0], X[hdb.labels_ == -1, 1], s=40, marker="x", color="#1E2630")   # noise
style_axes(axes[2], "HDBSCAN (min_cluster_size=25)", "x1", "x2")    # shared style
axes[3].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="RdPu", s=14, alpha=0.85)   # k-means labels
style_axes(axes[3], "k-means (k = 3) shreds the rings", "x1", "x2")   # shared style
annotated(axes[3], "equal-size Voronoi cells", "upper left")        # annotate the failure
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_dbscan.png")                                        # save the figure
print("done: DBSCAN, HDBSCAN and the noise label")                  # closing log
