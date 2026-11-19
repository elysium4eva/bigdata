"""l8_17_hier_real.py -- a real-data hierarchical clustering case study: Italian wine chemistry.

Every line carries an English comment. The numbers are REAL measurements, not simulated:
sklearn.datasets.load_wine() ships the UCI "Wine" table -- 178 wines grown in the same
region of Italy by three different cultivars -- with 13 chemical measurements per wine
(alcohol, malic acid, ash, magnesium, total phenols, flavanoids, colour intensity, hue,
proline, and so on). Because the cultivar of every bottle is known, the clusters can be
scored honestly with the adjusted Rand index (ARI) and not only admired in a picture.

The script standardises the 13 features, runs Ward clustering at k = 2, 3 and 4, compares
single / average / complete / ward linkage at the same k, draws a truncated dendrogram
with a cut at the 3-cluster height, profiles the three clusters in standardised units,
and writes the tidy tables that the lecture slides and captions use.
"""
from pathlib import Path                                             # locate the deck folder for the figure copy
import numpy as np                                                   # numeric arrays and counting
import pandas as pd                                                  # tidy tables for the CSV outputs
import matplotlib                                                    # choose a backend before pyplot imports
matplotlib.use("Agg")                                                # headless backend: no display needed
import matplotlib.pyplot as plt                                      # figure and axes API
from matplotlib.gridspec import GridSpec                             # uneven panel layout for the figure
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster    # scipy tree, drawing and cutting
from sklearn.datasets import load_wine                               # the real UCI wine measurements
from sklearn.preprocessing import StandardScaler                     # z-scores for the 13 features
from sklearn.cluster import AgglomerativeClustering                  # the sklearn agglomerative model
from sklearn.metrics import silhouette_score, adjusted_rand_score    # internal and external scores
from _common_l8 import save, dump, style_axes, annotated, OKABE, FIG   # shared house-style helpers

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK-capable font stack
plt.rcParams["axes.unicode_minus"] = False                            # draw a real minus sign, no tofu box
plt.rcParams["savefig.dpi"] = 200                                     # high resolution for the slides
plt.rcParams["savefig.bbox"] = "tight"                                # trim white space around the figure

# ---------------------------------------------------------------- 1) the real measured data
data = load_wine()                                                    # 178 wines, 13 chemical variables
X = data.data.astype(float)                                           # measured chemistry matrix, 178 x 13
y = data.target.astype(int)                                           # true cultivar label: 0, 1 or 2
features = [str(f) for f in data.feature_names]                       # the 13 measured variable names
Xz = StandardScaler().fit_transform(X)                                # z-score every column before clustering
n = Xz.shape[0]                                                       # number of wines in the table
counts = np.bincount(y)                                               # bottles per true cultivar
print("wine data: %d wines x %d measured features | cultivars %s (real UCI measurements)" % (n, len(features), counts.tolist()))   # report the data

# ---------------------------------------------------------------- 2) Ward at k = 2, 3, 4 and four linkages at k = 3
methods = ["single", "average", "complete", "ward"]                   # the four classic linkage rules
rows = []                                                             # one row per (linkage, k) pair
ward_labels = {}                                                      # keep the ward partitions for later panels
for method in methods:                                                # loop over the linkage rules
    for k in (2, 3, 4):                                               # the three cluster counts of the exercise
        model = AgglomerativeClustering(n_clusters=k, linkage=method).fit(Xz)   # fit on standardised features
        labels = model.labels_.astype(int)                            # cluster id per wine
        sil = float(silhouette_score(Xz, labels)) if len(set(labels)) > 1 else float("nan")   # internal score
        ari = float(adjusted_rand_score(y, labels))                    # external score against the cultivar
        rows.append({"linkage": method, "k": int(k),                    # which method and how many clusters
                     "silhouette": round(sil, 4), "ari": round(ari, 4)})   # the two scores
        if method == "ward":                                           # keep Ward for the profile and the figure
            ward_labels[k] = labels                                    # store the partition
link_tab = pd.DataFrame(rows, columns=["linkage", "k", "silhouette", "ari"])   # tidy score table, 12 rows
for k in (2, 3, 4):                                                    # report every Ward solution
    sizes = pd.Series(ward_labels[k]).value_counts().sort_values(ascending=False).tolist()   # cluster sizes
    ari_k = adjusted_rand_score(y, ward_labels[k])                     # agreement with the cultivar
    print("ward k=%d | sizes %s | ARI %.3f" % (k, sizes, ari_k))      # one compact line per k
print("k=3 linkage comparison:")                                      # header for the comparison block
for method in methods:                                                # one line per linkage method
    row = link_tab[(link_tab["linkage"] == method) & (link_tab["k"] == 3)].iloc[0]   # the k=3 row
    print("  %-8s | silhouette %.3f | ARI %+.3f" % (method, row["silhouette"], row["ari"]))   # report it

# ---------------------------------------------------------------- 3) cluster profile and a deterministic cluster order
raw = pd.DataFrame(Xz, columns=features).groupby(ward_labels[3]).mean()   # standardised mean per raw cluster
order = raw["alcohol"].sort_values(ascending=False).index.tolist()    # C1 = strongest wine, C3 = lightest
remap = {old: new + 1 for new, old in enumerate(order)}               # raw label -> presented label 1..3
ward3 = np.array([remap[c] for c in ward_labels[3]], dtype=int)       # relabelled Ward partition at k=3
profile = pd.DataFrame(Xz, columns=features).groupby(ward3).mean()    # standardised mean per presented cluster
profile.index.name = "cluster"                                         # name the index for the CSV melt
print("cluster order by mean alcohol: %s" % {int(k): round(float(profile.loc[k, "alcohol"]), 2) for k in profile.index})   # show the rule

# ---------------------------------------------------------------- 4) the scipy tree, its 3-cluster cut and a sanity check
Z = linkage(Xz, method="ward")                                        # Ward tree on the standardised data
cut3 = 0.5 * (float(Z[-3, 2]) + float(Z[-2, 2]))                      # halfway between the 3- and 2-cluster merges
scipy3 = fcluster(Z, t=cut3, criterion="distance")                    # cut the scipy tree at that height
agree = adjusted_rand_score(ward3, scipy3)                            # scipy and sklearn must agree
print("cut height for 3 clusters: %.2f | scipy cut gives %d clusters | scipy vs sklearn ARI %.3f" % (cut3, len(set(scipy3)), agree))   # report the cut

# ---------------------------------------------------------------- 5) tidy CSV outputs
labels_df = pd.DataFrame({"wine_id": np.arange(n)})                   # 0-based row index of each wine
labels_df["true_cultivar"] = y                                        # the cultivar that produced the wine
labels_df["ward_k3"] = ward3                                          # Ward cluster at k = 3, labels 1..3
labels_df = labels_df[["wine_id", "true_cultivar", "ward_k3"]]        # fix the column order for the deck
dump(labels_df, "l8_hier_real_labels.csv")                            # one row per wine
dump(link_tab, "l8_hier_real_linkage.csv")                            # four linkages times three k values
long_profile = profile.reset_index().melt(id_vars="cluster", var_name="feature", value_name="standardised_mean")   # long format
long_profile["standardised_mean"] = long_profile["standardised_mean"].round(4)   # four decimals is enough
dump(long_profile, "l8_hier_real_profile.csv")                        # cluster x feature grid, 39 rows

# ---------------------------------------------------------------- 6) the figure
fig = plt.figure(figsize=(13.4, 7.9))                                 # wide canvas for a slide panel
gs = GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.46, wspace=0.26)   # tree on top, two panels below
ax1 = fig.add_subplot(gs[0, :])                                       # full-width dendrogram panel
ax2 = fig.add_subplot(gs[1, 0])                                       # cluster profile heatmap
ax3 = fig.add_subplot(gs[1, 1])                                       # linkage comparison bars

# ---- panel (a): the truncated dendrogram with the 3-cluster cut
node_cluster = {i: int(ward3[i]) for i in range(n)}                   # a leaf belongs to one cluster
for i in range(Z.shape[0]):                                           # walk every merge of the full tree
    a, b = int(Z[i, 0]), int(Z[i, 1])                                 # the two child node ids
    node_cluster[n + i] = node_cluster[a] if node_cluster[a] == node_cluster[b] else 0   # 0 marks a mixed branch
COL = {1: OKABE[4], 2: OKABE[5], 3: OKABE[2], 0: "#9AA3AD"}           # blue / vermillion / green / grey


def branch_colour(node):                                              # colour a drawn link by its members
    """Return the colour of the cluster that owns a dendrogram node."""   # short docstring
    return COL.get(node_cluster.get(int(node), 0), "#9AA3AD")         # fall back to grey when unsure


tree = dendrogram(Z, ax=ax1, truncate_mode="lastp", p=20,             # keep the 20 broadest groups only
                  color_threshold=cut3, link_color_func=branch_colour,   # colour the three main branches
                  leaf_font_size=6.8)                                 # smaller leaf labels for the group counts
ax1.axhline(cut3, color="#8A0C3C", linestyle="--", linewidth=1.4)     # the 3-cluster cut line
ax1.set_ylim(0, float(Z[-1, 2]) * 1.06)                               # a little headroom above the root
ax1.set_ylabel("Ward merge distance (z-scored features)")             # vertical axis meaning
ax1.set_xlabel("wine groups, truncated to 20 leaves; (n) = n wines folded into one leaf")   # explain the labels
ax1.set_title("(a) Real wine data: Ward dendrogram, cut at height %.2f gives 3 clusters" % cut3)   # panel title
members = {i: [i] for i in range(n)}                                  # leaf id -> wine ids it covers
for i in range(Z.shape[0]):                                           # walk every merge again
    a, b = int(Z[i, 0]), int(Z[i, 1])                                 # the two child node ids
    members[n + i] = members[a] + members[b]                          # union of the two child groups
leaf_cluster = []                                                     # majority cluster per displayed leaf
for node in tree["leaves"]:                                           # displayed leaves, left to right
    votes = np.bincount(ward3[members[int(node)]], minlength=4)       # cluster votes inside this leaf
    leaf_cluster.append(int(votes.argmax()))                          # majority vote decides the colour
leaf_x = 5.0 + 10.0 * np.arange(len(tree["leaves"]))                  # scipy spaces displayed leaves by 10
NAMES = {1: "C1\nhigh alcohol,\nproline",                             # short profile of the first branch
         2: "C2\nhigh colour,\nlow flavanoids",                       # short profile of the second branch
         3: "C3\nlow alcohol,\nlow colour"}                           # short profile of the third branch
for c in (1, 2, 3):                                                   # one label per main branch
    xs = leaf_x[np.array(leaf_cluster) == c]                          # x positions of this branch's leaves
    ax1.text(xs.mean(), cut3 + 0.45, NAMES[c], ha="center", va="bottom", fontsize=8.0,   # place the label
             color=COL[c], weight="bold",                             # colour matches the branch
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=COL[c], lw=0.8, alpha=0.94))   # readable backing
annotated(ax1, "one dashed cut -> three interpretable wine styles", "upper left")   # teaching note

# ---- panel (b): standardised cluster profile of the most discriminative features
top_feats = profile.abs().max(axis=0).sort_values(ascending=False).index[:8].tolist()   # eight strongest features
SHORT = {"od280/od315_of_diluted_wines": "OD280/OD315",               # compact label for a long name
         "total_phenols": "total phenols",                            # compact label
         "color_intensity": "colour intensity"}                       # compact label
M = profile.loc[[1, 2, 3], top_feats].values.T                       # transpose to 8 features x 3 clusters
im = ax2.imshow(M, cmap="RdBu_r", vmin=-1.3, vmax=1.3, aspect="auto")   # diverging map, zero centred
for i, f in enumerate(top_feats):                                     # annotate every cell with its value
    for j in range(3):                                                # three clusters per feature
        v = float(M[i, j])                                            # the standardised mean
        ax2.text(j, i, "%+.2f" % v, ha="center", va="center", fontsize=7.8,   # print the number
                 color="white" if abs(v) > 0.75 else "#1E2630")       # white text on dark cells
ax2.set_xticks(range(3))                                              # three cluster columns
ax2.set_xticklabels(["C1", "C2", "C3"])                               # presented cluster labels
ax2.set_yticks(range(len(top_feats)))                                 # one row per feature
ax2.set_yticklabels([SHORT.get(f, f.replace("_", " ")) for f in top_feats], fontsize=7.6)   # readable names
ax2.set_title("(b) Cluster profile: standardised mean of the 8 most discriminative features")   # panel title
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)             # add a compact colour bar
cbar.set_label("standardised mean (z)", fontsize=8.2)                 # explain the colour scale
cbar.ax.tick_params(labelsize=7.4)                                    # small colour-bar ticks
ax2.grid(False)                                                       # no grid over the heatmap cells

# ---- panel (c): how the linkage rule changes the answer at k = 3
k3 = link_tab[link_tab["k"] == 3].set_index("linkage").loc[methods]    # the k=3 rows in display order
xx = np.arange(len(methods))                                          # one group of bars per linkage
width = 0.38                                                          # width of each bar in a group
bars_sil = ax3.bar(xx - width / 2, k3["silhouette"], width, color=OKABE[4], label="silhouette")   # internal score
bars_ari = ax3.bar(xx + width / 2, k3["ari"], width, color=OKABE[5], label="ARI vs cultivar")   # external score
ax3.axhline(0, color="#4A5058", linewidth=0.8)                        # zero baseline for negative bars
ax3.bar_label(bars_sil, fmt="%.2f", fontsize=7.0, padding=1.5)        # value labels on the silhouette bars
ax3.bar_label(bars_ari, fmt="%.2f", fontsize=7.0, padding=1.5)        # value labels on the ARI bars
ax3.set_xticks(xx)                                                    # tick per linkage
ax3.set_xticklabels(methods)                                          # linkage names
ax3.set_ylim(-0.07, 0.93)                                             # headroom for the value labels
style_axes(ax3, "(c) Same data, same k = 3: linkage decides the answer", "linkage", "score")   # shared style
ax3.legend(fontsize=7.4, loc="upper left")                            # two-metric legend

save(fig, "fig_hier_real.png")                                        # write website/figures/lecture08

# ---------------------------------------------------------------- 7) copy the figure into the Beamer deck
deck_dir = Path(__file__).resolve().parents[3] / "lecture08-2026-szu" / "figures"   # the deck's figure folder
deck_dir.mkdir(parents=True, exist_ok=True)                           # create it if it is missing
src = FIG / "fig_hier_real.png"                                       # the figure just written
dst = deck_dir / "fig_hier_real.png"                                  # its copy inside the deck
dst.write_bytes(src.read_bytes())                                     # byte-identical copy, same file name
print("[deck]", dst)                                                  # log the deck copy

# ---------------------------------------------------------------- 8) the compact summary for the slides
print("summary: %d wines, 13 chemistry features, 3 true cultivars" % n)   # data size
for method in methods:                                                # k=3 numbers per linkage
    row = link_tab[(link_tab["linkage"] == method) & (link_tab["k"] == 3)].iloc[0]   # the k=3 row
    print("  %-8s | silhouette %.3f | ARI %+.3f" % (method, row["silhouette"], row["ari"]))   # two scores
for c in (1, 2):                                                      # the two branches named in the caption
    top = profile.loc[c].abs().sort_values(ascending=False).index[:3]   # three strongest features
    text = " | ".join("%s %+.2f" % (f, profile.loc[c, f]) for f in top)   # feature and signed mean
    print("cluster %d top-3 features: %s" % (c, text))                # report the profile
print("chosen cut height: %.2f" % cut3)                               # the dendrogram cut
print("done: real-data hierarchical case study, %d wines" % n)        # closing log
