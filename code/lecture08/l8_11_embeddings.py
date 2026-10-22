"""l8_11_embeddings.py -- PCA, t-SNE and UMAP on the same high-dimensional sample.

Every line carries an English comment. The script builds a 40-dimensional sample with six
latent groups, projects it with the three standard methods, and measures how well the
two-dimensional picture preserves the neighbourhood structure so the deck can argue that
an embedding is a picture, not a proof.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
from sklearn.decomposition import PCA                              # the linear baseline
from sklearn.manifold import TSNE                                  # t-SNE
from sklearn.neighbors import NearestNeighbors                     # neighbourhood preservation
from _common_l8 import (make_highdim, save, dump, style_axes,      # shared helpers
                        annotated, OKABE)                               # shared plotting helpers
try:                                                               # UMAP is optional
    from umap import UMAP                                          # the modern default
    HAS_UMAP = True                                                # remember that it loaded
except Exception:                                                  # fall back gracefully
    HAS_UMAP = False                                               # no UMAP available

# ---------------------------------------------------------------- 1) the high-dimensional sample
frame, labels = make_highdim(900, 40, 6)                           # 900 rows, 40 dimensions, 6 groups
X = frame.values                                                   # the matrix
dump(frame.assign(group=labels), "l8_highdim.csv")                 # persist it with the groups
print("rows:", X.shape[0], "| dimensions:", X.shape[1], "| groups: 6")   # report

# ---------------------------------------------------------------- 2) neighbourhood preservation helper
def trustworthiness(embedding, k=10):                              # compare neighbourhoods
    """Return the share of the k nearest neighbours preserved in the embedding."""   # short docstring
    high = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]   # high-dim neighbours
    low = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(embedding, return_distance=False)[:, 1:]   # low-dim neighbours
    keep = 0                                                       # count preserved pairs
    for i in range(len(X)):                                        # loop over the rows
        keep += len(set(high[i]).intersection(set(low[i])))        # overlap of the two neighbourhoods
    return keep / (len(X) * k)                                     # share preserved

# ---------------------------------------------------------------- 3) the three projections
rows = []                                                          # one row per method
embeddings = {}                                                    # keep the coordinates
pca = PCA(n_components=2, random_state=42).fit_transform(X)        # linear projection
embeddings["PCA"] = pca                                            # store it
rows.append({"method": "PCA", "trustworthiness": round(trustworthiness(pca), 4)})   # score it
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)   # t-SNE
embeddings["t-SNE"] = tsne                                         # store it
rows.append({"method": "t-SNE", "trustworthiness": round(trustworthiness(tsne), 4)})   # score it
if HAS_UMAP:                                                       # only when UMAP is installed
    umap_emb = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)   # UMAP
    embeddings["UMAP"] = umap_emb                                  # store it
    rows.append({"method": "UMAP", "trustworthiness": round(trustworthiness(umap_emb), 4)})   # score it
    print("UMAP available")                                        # report
else:                                                              # otherwise
    print("UMAP not installed, skipping")                          # report the fallback
compare = pd.DataFrame(rows)                                       # assemble the table
dump(compare, "l8_embedding_compare.csv")                          # persist it
print(compare.to_string(index=False))                              # report the scores

# ---------------------------------------------------------------- 4) the figure
n_panels = len(embeddings)                                         # number of projections available
fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 3.9))   # one panel per method
if n_panels == 1:                                                  # normalise to a list
    axes = [axes]                                                  # a single axes becomes a list
for ax, (name, coords) in zip(axes, embeddings.items()):           # loop over the projections
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="RdPu", s=10, alpha=0.85)   # colour by group
    score = float(compare.loc[compare["method"] == name, "trustworthiness"].iloc[0])   # neighbourhood score
    style_axes(ax, "%s | trust %.3f" % (name, score), "dim 1", "dim 2")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_embedding.png")                                     # save the embedding figure
print("done: PCA, t-SNE and UMAP on the same data")                 # closing log
