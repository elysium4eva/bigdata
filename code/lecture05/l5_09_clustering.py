"""l5_09_clustering.py -- k-means, hierarchical clustering and DBSCAN side by side.

Every line carries an English comment. The script builds a behavioural customer
sample with four latent segments plus a few outliers, then compares internal
validation measures (inertia, silhouette, Calinski-Harabasz) and the cluster
assignments of three algorithm families.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # four-panel comparison figure
from scipy.cluster.hierarchy import linkage, fcluster              # agglomerative clustering from scipy
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN   # the three algorithm families
from sklearn.decomposition import PCA                              # two dimensional view for the plots
from sklearn.metrics import (silhouette_score,                     # internal measure: separation versus compactness
                             calinski_harabasz_score,              # internal measure based on variance ratios
                             adjusted_rand_score)                  # agreement with the known segments
from sklearn.preprocessing import StandardScaler                   # clustering is scale sensitive
from _common_l5 import RNG, save, dump, style_axes, OKABE          # shared helpers

# ---------------------------------------------------------------- 1) behavioural sample with four segments
N = 800                                                            # number of customers
centres = 1.7 * np.array([[2.2, 1.8, 0.4, 2.4, 1.1, 0.3],          # segment one: high engagement, low discount use
                          [-1.6, -1.2, 1.9, -1.4, -0.8, 1.6],      # segment two: low engagement, discount driven
                          [0.4, 2.3, -1.1, 0.8, 2.0, -0.9],        # segment three: frequent but small baskets
                          [-0.9, 0.5, 0.6, -2.1, -1.4, 0.9]])      # segment four: dormant high value customers
labels_true = RNG.integers(0, 4, N)                                # assign each customer to one segment
X = centres[labels_true] + RNG.normal(0, 0.75, (N, 6))              # add within segment variation
outliers = RNG.uniform(-6, 6, (30, 6))                             # thirty clearly abnormal customers
X = np.vstack([X, outliers])                                       # append them to the sample
labels_true = np.concatenate([labels_true, np.full(30, -1)])        # mark the outliers with label minus one
cols = ["sessions", "basket_value", "discount_use", "email_opens", "app_minutes", "returns"]   # feature names
customers = pd.DataFrame(X, columns=cols).round(3)                 # assemble the feature frame
customers["segment_true"] = labels_true                            # keep the generating segment for validation
dump(customers, "l5_customers_behavior.csv")                       # persist the sample
print("customers:", X.shape, "| segments:", np.bincount(labels_true[labels_true >= 0]))   # report the sizes
X_std = StandardScaler().fit_transform(customers[cols].values)     # standardise before clustering

# ---------------------------------------------------------------- 2) how many clusters? inertia and silhouette
k_grid = list(range(2, 11))                                        # candidate numbers of clusters
rows = []                                                          # one row per k
for k in k_grid:                                                   # loop over the candidates
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_std)   # fit k-means for this k
    sil = silhouette_score(X_std, km.labels_, sample_size=600, random_state=42)   # sampled silhouette score
    ch = calinski_harabasz_score(X_std, km.labels_)                # variance ratio criterion
    rows.append({"k": k, "inertia": round(float(km.inertia_), 1),   # within cluster sum of squares
                 "silhouette": round(float(sil), 4),                # separation measure
                 "calinski_harabasz": round(float(ch), 1)})         # variance ratio measure
k_table = pd.DataFrame(rows)                                       # assemble the table
dump(k_table, "l5_cluster_k_table.csv")                            # persist the selection table
best_k_sil = int(k_table.loc[k_table["silhouette"].idxmax(), "k"])   # k preferred by the silhouette
print(k_table.to_string(index=False))                              # print the selection table
print("best k by silhouette:", best_k_sil, "| silhouette %.3f" % k_table["silhouette"].max())   # report the choice

# ---------------------------------------------------------------- 3) three algorithm families at the chosen k
k_use = 4                                                          # the generating number of segments
kmeans = KMeans(n_clusters=k_use, n_init=10, random_state=42).fit(X_std)   # centroid based clustering
agglo = AgglomerativeClustering(n_clusters=k_use, linkage="ward").fit(X_std)   # hierarchical clustering, ward linkage
Z = linkage(X_std, method="ward")                                  # linkage matrix for the dendrogram cut
dbscan = DBSCAN(eps=1.6, min_samples=12).fit(X_std)                # density based clustering: one global radius
print("dbscan clusters:", len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0),   # how many groups it found
      "| noise:", int((dbscan.labels_ == -1).sum()))               # and how many points it called noise
assignments = pd.DataFrame({"kmeans": kmeans.labels_,             # cluster labels from k-means
                            "hierarchical": agglo.labels_,         # cluster labels from ward linkage
                            "dbscan": dbscan.labels_,              # cluster labels from DBSCAN (-1 means noise)
                            "segment_true": labels_true})          # the generating segments
dump(assignments, "l5_cluster_labels.csv")                         # persist the assignments
summary = []                                                       # one row per algorithm
for name, lab in [("kmeans", kmeans.labels_), ("hierarchical", agglo.labels_), ("dbscan", dbscan.labels_)]:   # loop
    mask = lab >= 0                                                # DBSCAN noise points have label minus one
    enough = mask.sum() > 10 and len(set(lab[mask])) > 1           # silhouette needs at least two non-empty clusters
    sil = silhouette_score(X_std[mask], lab[mask], sample_size=600, random_state=42) if enough else np.nan   # score
    ari = adjusted_rand_score(labels_true[labels_true >= 0],       # compare with the generating truth
                              lab[labels_true >= 0]) if len(set(lab)) > 1 else np.nan   # only meaningful with clusters
    summary.append({"algorithm": name, "clusters": int(len(set(lab)) - (1 if -1 in lab else 0)),   # cluster count
                    "noise_points": int((lab == -1).sum()),         # points declared as noise
                    "silhouette": round(float(sil), 4) if sil == sil else None,   # separation when defined
                    "ARI_vs_truth": round(float(ari), 4) if ari == ari else None})   # agreement with the segments
algo_table = pd.DataFrame(summary)                                 # assemble the comparison
dump(algo_table, "l5_cluster_scores.csv")                          # persist the comparison
print(algo_table.to_string(index=False))                           # print the comparison

# ---------------------------------------------------------------- 4) two dimensional projection for plotting
proj = PCA(n_components=2).fit_transform(X_std)                     # project to two dimensions for visualisation
fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5))                # four panels in one row
axes[0].plot(k_table["k"], k_table["inertia"], "-o", ms=4, color=OKABE[0], label="inertia")   # elbow curve
ax0b = axes[0].twinx()                                             # second axis for the silhouette
ax0b.plot(k_table["k"], k_table["silhouette"], "-s", ms=4, color=OKABE[2], label="silhouette")   # silhouette curve
ax0b.set_ylabel("Silhouette", fontsize=9)                          # label the right axis
ax0b.grid(False)                                                   # avoid a doubled grid
axes[0].axvline(best_k_sil, color="#66717D", ls="--", lw=1.0)       # mark the silhouette optimum
style_axes(axes[0], "Choosing k: elbow plus silhouette", "Number of clusters", "Inertia")   # shared style
axes[0].legend(fontsize=7.2, loc="upper right")                    # left axis legend
ax0b.legend(fontsize=7.2, loc="lower right")                       # right axis legend
panels = [("k-means (k=4)", kmeans.labels_, OKABE[:4]),            # centroid based result
          ("hierarchical ward (k=4)", agglo.labels_, OKABE[:4]),   # ward linkage result
          ("DBSCAN (eps=1.6)", dbscan.labels_, OKABE)]             # density based result, noise in grey
for ax, (title, lab, palette) in zip(axes[1:], panels):            # draw one scatter per algorithm
    for c in sorted(set(lab)):                                     # loop over the labels present
        m = lab == c                                               # mask for this label
        colour = "#B9B3B6" if c == -1 else palette[int(c) % len(palette)]   # grey marks DBSCAN noise
        label = "noise" if c == -1 else f"cluster {c}"             # readable legend entry
        ax.scatter(proj[m, 0], proj[m, 1], s=11, color=colour, alpha=0.75, label=label)   # scatter the members
    style_axes(ax, f"{title} | {len(set(lab)) - (1 if -1 in lab else 0)} clusters", "PC1", "PC2")   # shared style
    ax.legend(fontsize=6.4, loc="upper right", ncol=2)             # compact two column legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_clustering.png")                                    # save the clustering comparison figure
