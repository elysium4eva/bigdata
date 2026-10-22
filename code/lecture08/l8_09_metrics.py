"""l8_09_metrics.py -- internal and external clustering metrics, and when they mislead.

Every line carries an English comment. The script runs four algorithms on the density
sample, scores each with internal metrics, and where a ground truth exists also with
external metrics, so the deck can show that the "best" algorithm depends on the number.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # two panels
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering   # three of the four families
from sklearn.mixture import GaussianMixture                       # the distribution-based family
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,   # internal metrics
                             davies_bouldin_score, adjusted_rand_score,   # internal and external metrics
                             normalized_mutual_info_score)         # external metrics
from _common_l8 import (make_blobs, make_density, save, dump,      # shared helpers
                        style_axes, annotated, OKABE)                   # shared plotting helpers

# ---------------------------------------------------------------- 1) a sample with a known truth
blobs = make_blobs(420, 4)                                         # four well-separated blobs
X = blobs[["x1", "x2"]].values                                     # feature matrix
truth = blobs["label"].values                                      # ground truth
print("rows:", X.shape[0], "| true clusters: 4")                   # report

# ---------------------------------------------------------------- 2) four algorithms, four stories
runs = {"k-means (k=4)": KMeans(n_clusters=4, n_init=10, random_state=42).fit(X),   # centroid family
        "wards (k=4)": AgglomerativeClustering(n_clusters=4, linkage="ward").fit(X),   # connectivity family
        "GMM (k=4)": GaussianMixture(n_components=4, random_state=42).fit(X),   # distribution family
        "DBSCAN": DBSCAN(eps=1.2, min_samples=5).fit(X)}            # density family
rows = []                                                          # one row per algorithm
for name, model in runs.items():                                   # loop over the four families
    labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)   # labels from the model
    labels = np.asarray(labels)                                    # ensure an array
    mask = labels != -1                                            # ignore the noise label for the metrics
    sil = float(silhouette_score(X[mask], labels[mask])) if len(set(labels[mask])) > 1 else float("nan")   # silhouette
    ch = float(calinski_harabasz_score(X[mask], labels[mask])) if len(set(labels[mask])) > 1 else float("nan")   # CH
    db = float(davies_bouldin_score(X[mask], labels[mask])) if len(set(labels[mask])) > 1 else float("nan")   # DB
    ari = float(adjusted_rand_score(truth[mask], labels[mask]))     # external: adjusted Rand
    nmi = float(normalized_mutual_info_score(truth[mask], labels[mask]))   # external: mutual information
    rows.append({"model": name, "clusters": int(len(set(labels[mask]))),   # clusters found
                 "silhouette": round(sil, 4) if sil == sil else None,   # internal
                 "calinski_harabasz": round(ch, 1) if ch == ch else None,   # internal
                 "davies_bouldin": round(db, 4) if db == db else None,   # internal (lower is better)
                 "adjusted_rand": round(ari, 4), "nmi": round(nmi, 4)})   # external
    print("%-12s | sil %.3f | ARI %.3f | NMI %.3f" % (name, sil, ari, nmi))   # report
table = pd.DataFrame(rows)                                         # assemble the table
dump(table, "l8_cluster_metrics.csv")                              # persist it

# ---------------------------------------------------------------- 3) the trap: one cluster scores best
trap_rows = []                                                     # silhouette per k on the blob sample
for k in range(1, 9):                                              # sweep k
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)   # fit
    sil = float(silhouette_score(X, model.labels_)) if k > 1 else float("nan")   # silhouette needs k >= 2
    trap_rows.append({"k": k, "silhouette": round(sil, 4) if k > 1 else None})   # record
trap = pd.DataFrame(trap_rows)                                     # assemble
dump(trap, "l8_metric_trap.csv")                                   # persist
print("silhouette by k:", trap["silhouette"].tolist())              # report the trap

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                # three panels
names = table["model"].tolist()                                    # algorithm names
axes[0].bar(names, table["silhouette"].fillna(0), color=OKABE[2], alpha=0.9)   # silhouette
axes[0].tick_params(axis="x", rotation=20, labelsize=7.6)          # rotate the labels
style_axes(axes[0], "Internal metric: silhouette", "", "silhouette")   # shared style
annotated(axes[0], "higher is better", "upper right")              # annotate the direction
axes[1].bar(names, table["adjusted_rand"], color=OKABE[5], alpha=0.9)   # ARI
axes[1].tick_params(axis="x", rotation=20, labelsize=7.6)          # rotate the labels
style_axes(axes[1], "External metric: adjusted Rand", "", "ARI")   # shared style
annotated(axes[1], "needs a ground truth", "upper right")          # annotate the requirement
axes[2].plot(trap["k"][1:], trap["silhouette"][1:], "-o", ms=5, color=OKABE[4])   # the trap curve
style_axes(axes[2], "Silhouette over k: it can prefer the wrong k", "k", "silhouette")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_cluster_metrics.png")                               # save the figure
print("done: internal and external metrics, and the trap")          # closing log
