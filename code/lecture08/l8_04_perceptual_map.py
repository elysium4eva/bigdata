"""l8_04_perceptual_map.py -- the perceptual map: brands placed on two interpretable axes.

Every line carries an English comment. The script aggregates respondent ratings into a
brand-by-attribute matrix, runs a PCA to place the brands on two axes, and draws the
classic positioning map with the market share encoded as circle size.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # the map
from sklearn.decomposition import PCA                              # the two axes come from a PCA
from sklearn.preprocessing import StandardScaler                   # centre and scale the attributes
from _common_l8 import (make_brands, save, dump, style_axes,       # shared helpers
                        annotated, OKABE)                               # shared plotting helpers

# ---------------------------------------------------------------- 1) the sample
ratings = make_brands(12, 260)                                     # respondent level brand ratings
dump(ratings, "l8_brands.csv")                                     # persist the long format
brands = sorted(ratings["brand"].unique())                         # brand labels in order
summary = ratings.groupby("brand").agg(price=("price", "mean"),    # average perceived price
                                       performance=("performance", "mean"),   # average perceived performance
                                       share=("share", "mean")).reset_index()   # average share
dump(summary.round(3), "l8_brand_summary.csv")                     # persist the summary
print("brands:", len(summary), "| respondents per brand:", ratings.groupby("brand").size().iloc[0])   # report

# ---------------------------------------------------------------- 2) PCA on the two attributes
attrs = summary[["price", "performance"]].values                   # the attribute matrix
scaled = StandardScaler().fit_transform(attrs)                     # centre and scale
pca = PCA(n_components=2).fit(scaled)                              # two components (the two axes)
coords = pca.transform(scaled)                                     # brand coordinates
loadings = pca.components_                                         # how the axes relate to price/performance
print("variance explained:", np.round(pca.explained_variance_ratio_, 3))   # report
print("axis loadings:", np.round(loadings, 3))                      # report the axis meaning
mapping = pd.DataFrame({"brand": summary["brand"],                 # brand label
                        "axis1": np.round(coords[:, 0], 3),        # first axis coordinate
                        "axis2": np.round(coords[:, 1], 3),        # second axis coordinate
                        "share": summary["share"].round(2)})       # market share
dump(mapping, "l8_perceptual_coords.csv")                          # persist the coordinates
print(mapping.to_string(index=False))                              # report the map

# ---------------------------------------------------------------- 3) the map
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))                # the map plus a reading guide
sizes = 60 + 420 * (summary["share"] / summary["share"].max())      # share encoded as circle size
sc = axes[0].scatter(coords[:, 0], coords[:, 1], s=sizes,          # the brand circles
                     c=summary["share"], cmap="RdPu", alpha=0.85, edgecolors="#1E2630", linewidths=0.8)   # style
for i, name in enumerate(summary["brand"]):                        # label every circle
    axes[0].annotate(name, (coords[i, 0], coords[i, 1]), xytext=(6, 4),   # small offset
                     textcoords="offset points", fontsize=9, fontweight="bold")   # bold brand label
axes[0].axhline(0, color="#66717D", lw=0.8, ls="--")                # axis origin
axes[0].axvline(0, color="#66717D", lw=0.8, ls="--")                # axis origin
style_axes(axes[0], "Perceptual map (size = market share)", "dimension 1 (price)", "dimension 2 (performance)")   # style
plt.colorbar(sc, ax=axes[0], fraction=0.046, label="market share (%)")   # colour bar
axes[1].scatter(summary["price"], summary["performance"], s=sizes, c=summary["share"], cmap="RdPu",   # raw axes
                alpha=0.85, edgecolors="#1E2630", linewidths=0.8)   # same encoding
for i, name in enumerate(summary["brand"]):                        # label the points
    axes[1].annotate(name, (summary["price"].iloc[i], summary["performance"].iloc[i]),   # position
                     xytext=(6, 4), textcoords="offset points", fontsize=9)   # small offset
style_axes(axes[1], "Same brands on the raw attributes", "perceived price", "perceived performance")   # style
annotated(axes[1], "the map is a rotation of this", "upper left")   # annotate the relationship
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_perceptual_map.png")                                # save the map
print("done: perceptual map with %d brands" % len(summary))         # closing log
