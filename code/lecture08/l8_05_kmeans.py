"""l8_05_kmeans.py -- k-means, its Voronoi geometry and how to choose k.

Every line carries an English comment. The script fits k-means on a blob sample, draws
the Voronoi cells and the four classic failure shapes, and compares the elbow rule with
the silhouette coefficient so the deck can show that the two criteria can disagree.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
from matplotlib.colors import ListedColormap                       # discrete colours per cluster
from sklearn.cluster import KMeans                                 # the algorithm
from sklearn.metrics import silhouette_score                       # an internal metric
from _common_l8 import (make_blobs, RNG, save, dump, style_axes,   # shared helpers
                        annotated, OKABE)                               # shared plotting helpers

# ---------------------------------------------------------------- 1) the blob sample
blobs = make_blobs(420, 4)                                         # four well-separated blobs
dump(blobs, "l8_blobs.csv")                                        # persist it
X = blobs[["x1", "x2"]].values                                     # feature matrix
print("rows:", X.shape[0], "| true clusters:", blobs["label"].nunique())   # report

# ---------------------------------------------------------------- 2) k-means for several k
rows = []                                                          # one row per k
fitted = {}                                                        # keep the models for plotting
for k in range(1, 11):                                             # sweep the number of clusters
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)   # fit with ten restarts
    fitted[k] = model                                              # remember the model
    silhouette = float(silhouette_score(X, model.labels_)) if k > 1 else float("nan")   # needs k >= 2
    rows.append({"k": k, "inertia": round(float(model.inertia_), 2),   # within-cluster sum of squares
                 "silhouette": round(silhouette, 4) if k > 1 else None})   # silhouette coefficient
    print("k=%2d | inertia %8.2f | silhouette %s" % (                   # report the row for this k
        k, model.inertia_, "%.3f" % silhouette if k > 1 else "-"))   # report the row
sweep = pd.DataFrame(rows)                                         # assemble the table
dump(sweep, "l8_kmeans_sweep.csv")                                 # persist it
best_k_sil = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])       # k by silhouette
print("best k by silhouette:", best_k_sil)                         # report the choice

# ---------------------------------------------------------------- 3) the four classic failure shapes
shapes = {}                                                        # synthetic failure cases
theta = RNG.uniform(0, 2 * np.pi, 300)                             # for the ring
shapes["non-spherical (rings)"] = np.c_[3 * np.cos(theta), 3 * np.sin(theta)] + RNG.normal(0, 0.12, (300, 2))   # a ring
shapes["unequal sizes"] = np.vstack([RNG.normal([0, 0], 0.5, (280, 2)),   # one tight cluster
                                     RNG.normal([6, 6], 2.2, (40, 2))])   # one huge sparse cluster
shapes["different density"] = np.vstack([RNG.normal([0, 0], 0.35, (200, 2)),   # dense cluster
                                         RNG.normal([4, 0], 1.6, (200, 2))])   # sparse cluster
shapes["with outliers"] = np.vstack([RNG.normal([0, 0], 0.8, (260, 2)),   # a cluster
                                     RNG.uniform(-10, 10, (20, 2))])   # scattered outliers
shape_rows = []                                                    # results per shape
for name, points in shapes.items():                                # loop over the four shapes
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(points)   # two clusters
    shape_rows.append({"shape": name, "kmeans_inertia": round(float(km.inertia_), 2),   # inertia
                       "silhouette": round(float(silhouette_score(points, km.labels_)), 4)})   # silhouette
dump(pd.DataFrame(shape_rows), "l8_kmeans_failures.csv")            # persist the table

# ---------------------------------------------------------------- 4) figure: the Voronoi cells
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                # three panels
xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 320)         # horizontal grid
ys = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 320)         # vertical grid
XX, YY = np.meshgrid(xs, ys)                                       # full grid
grid = np.c_[XX.ravel(), YY.ravel()]                               # flatten for prediction
for ax, k in zip(axes[:2], (4, 8)):                                # the true k and an over-split k
    labels = fitted[k].predict(grid).reshape(XX.shape)             # cell labels on the grid
    ax.contourf(XX, YY, labels, levels=np.arange(-0.5, k + 0.5, 1), alpha=0.22, cmap="RdPu")   # Voronoi cells
    ax.scatter(X[:, 0], X[:, 1], c=fitted[k].labels_, cmap="RdPu", s=14, alpha=0.85)   # points
    ax.scatter(fitted[k].cluster_centers_[:, 0], fitted[k].cluster_centers_[:, 1], s=120,   # centres
               marker="X", color="#1E2630", label="centres")       # marker style
    style_axes(ax, "k-means with k = %d" % k, "x1", "x2")           # shared style
axes[2].plot(sweep["k"], sweep["inertia"], "-o", ms=5, color=OKABE[5], label="inertia")   # elbow curve
axes[2].set_yscale("log")                                          # inertia spans orders of magnitude
style_axes(axes[2], "Elbow: inertia versus k", "k", "inertia (log scale)")   # shared style
annotated(axes[2], "bend near the true k", "upper right")           # annotate the elbow
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_kmeans.png")                                        # save the Voronoi figure

# ---------------------------------------------------------------- 5) figure: elbow versus silhouette
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                # three panels
axes[0].plot(sweep["k"], sweep["inertia"], "-o", ms=5, color=OKABE[5])   # elbow
style_axes(axes[0], "Elbow rule: no sharp bend?", "k", "inertia")   # shared style
axes[1].plot(sweep["k"][1:], sweep["silhouette"][1:], "-s", ms=5, color=OKABE[2])   # silhouette curve
axes[1].axvline(best_k_sil, ls="--", lw=1.0, color="#66717D")       # mark the best k
style_axes(axes[1], "Silhouette prefers k = %d" % best_k_sil, "k", "silhouette")   # shared style
axes[2].barh([r["shape"] for r in shape_rows], [r["silhouette"] for r in shape_rows],   # failure shapes
             color=[OKABE[5], OKABE[1], OKABE[3], OKABE[4]], alpha=0.9)   # one colour per shape
style_axes(axes[2], "Silhouette on the four failure shapes", "silhouette", "")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_k_choice.png")                                      # save the choice figure
print("done: k-means, Voronoi cells and the k choice")              # closing log
