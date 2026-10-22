"""l8_01_pca_basics.py -- PCA on the smartphone sample: directions, scree and scaling.

Every line carries an English comment. The script fits a PCA on a synthetic phone-spec
sample, draws the two leading components, compares the standardised and raw analyses,
and records the explained variance so the deck can show a scree plot.
"""
import numpy as np                                                 # arrays and the projections
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
from sklearn.decomposition import PCA, IncrementalPCA              # batch and out-of-core PCA
from sklearn.preprocessing import StandardScaler                   # standardisation before PCA
from _common_l8 import (make_smartphone, save, dump, style_axes,   # shared helpers
                        annotated, OKABE)                               # shared plotting helpers

# ---------------------------------------------------------------- 1) the sample
phones = make_smartphone(60)                                       # synthetic phone specifications
dump(phones, "l8_smartphone.csv")                                  # persist it for the deck
features = ["cpu", "gpu", "cores", "ram_gb", "camera_mp", "n_cameras", "price"]   # numeric features
X = phones[features].values                                        # feature matrix
print("rows:", X.shape[0], "| features:", X.shape[1])              # report the sample

# ---------------------------------------------------------------- 2) PCA on standardised features
scaler = StandardScaler().fit(X)                                   # fit the scaler on all rows (teaching sample)
Xs = scaler.transform(X)                                           # standardised matrix
pca_std = PCA().fit(Xs)                                            # full PCA on the standardised data
ratio_std = pca_std.explained_variance_ratio_                      # share of variance per component
print("standardised: first three variance shares:", np.round(ratio_std[:3], 3))   # report
print("cumulative:", np.round(np.cumsum(ratio_std)[:5], 3))         # cumulative shares

# ---------------------------------------------------------------- 3) PCA on the raw features
pca_raw = PCA().fit(X)                                             # full PCA on the raw scale
ratio_raw = pca_raw.explained_variance_ratio_                      # share of variance per component
print("raw scale: first component share %.3f" % ratio_raw[0])       # the raw first component dominates
print("raw loadings of PC1:", np.round(pca_raw.components_[0], 3))  # which features it loads on

# ---------------------------------------------------------------- 4) incremental PCA against batch PCA
incs = []                                                          # reconstruction error per component count
for k in [1, 2, 3, 4, 5]:                                          # a few component counts
    ipca = IncrementalPCA(n_components=k, batch_size=10)           # out-of-core PCA
    ipca.fit(Xs)                                                   # fit in mini batches
    recon = ipca.inverse_transform(ipca.transform(Xs))              # reconstruct from k components
    incs.append({"components": k, "batch_rmse": float(np.sqrt(np.mean((Xs - recon) ** 2))),   # batch error
                 "explained": float(ipca.explained_variance_ratio_.sum())})   # explained share
    batch = PCA(n_components=k).fit(Xs)                            # batch PCA for comparison
    recon_b = batch.inverse_transform(batch.transform(Xs))          # its reconstruction
    incs[-1]["batch_vs_incremental_gap"] = float(np.abs(np.sqrt(np.mean((Xs - recon_b) ** 2))   # the gap between batch and incremental PCA
                                                          - incs[-1]["batch_rmse"]))   # the gap
inc_table = pd.DataFrame(incs)                                     # assemble the table
dump(inc_table.round(5), "l8_pca_incremental.csv")                 # persist it
print(inc_table.round(4).to_string(index=False))                   # report the comparison

# ---------------------------------------------------------------- 5) explained variance table
variance_table = pd.DataFrame({"component": np.arange(1, len(ratio_std) + 1),   # component index
                               "share_std": np.round(ratio_std, 4),   # share on standardised data
                               "cum_std": np.round(np.cumsum(ratio_std), 4),   # cumulative share
                               "share_raw": np.round(ratio_raw, 4),   # share on raw data
                               "cum_raw": np.round(np.cumsum(ratio_raw), 4)})   # cumulative share
dump(variance_table, "l8_pca_variance.csv")                        # persist it

# ---------------------------------------------------------------- 6) figure: the two leading components
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))                # three panels
proj = pca_std.transform(Xs)[:, :2]                                # project on the first two components
sc = axes[0].scatter(proj[:, 0], proj[:, 1], c=phones["price"], cmap="RdPu", s=26)   # colour by price
style_axes(axes[0], "Standardised PCA: first two components", "PC1", "PC2")   # shared style
plt.colorbar(sc, ax=axes[0], fraction=0.046, label="price")        # colour bar
proj_raw = pca_raw.transform(X)[:, :2]                             # projection on the raw scale
sc2 = axes[1].scatter(proj_raw[:, 0], proj_raw[:, 1], c=phones["price"], cmap="RdPu", s=26)   # colour by price
style_axes(axes[1], "Raw-scale PCA: dominated by price", "PC1", "PC2")   # shared style
plt.colorbar(sc2, ax=axes[1], fraction=0.046, label="price")       # colour bar
for i, name in enumerate(features):                                # draw the loading arrows
    axes[2].arrow(0, 0, pca_std.components_[0, i], pca_std.components_[1, i],   # arrow for this feature
                  color=OKABE[i % len(OKABE)], width=0.006, head_width=0.05)   # arrow style
    axes[2].text(pca_std.components_[0, i] * 1.12, pca_std.components_[1, i] * 1.12,   # label position
                 name, fontsize=7.6)                               # the feature name
axes[2].set_aspect("equal")                                        # keep the circle round
style_axes(axes[2], "Loadings: what PC1 and PC2 mean", "PC1 loading", "PC2 loading")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_pca_2d.png")                                        # save the geometry figure

# ---------------------------------------------------------------- 7) figure: scree and cumulative variance
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))                # three panels
axes[0].plot(variance_table["component"], variance_table["share_std"], "-o", ms=5, color=OKABE[5])   # scree
axes[0].bar(variance_table["component"], variance_table["share_std"], color=OKABE[5], alpha=0.25)   # bars
style_axes(axes[0], "Scree plot (standardised)", "component", "variance share")   # shared style
axes[1].plot(variance_table["component"], variance_table["cum_std"], "-s", ms=5, color=OKABE[2])   # cumulative
axes[1].axhline(0.8, ls="--", lw=1.0, color="#66717D")             # the 80 per cent reference
axes[1].axhline(0.9, ls=":", lw=1.0, color="#66717D")              # the 90 per cent reference
style_axes(axes[1], "Cumulative variance explained", "components kept", "cumulative share")   # shared style
axes[2].plot(variance_table["component"], variance_table["share_raw"], "-o", ms=5, color=OKABE[1])   # raw scree
axes[2].plot(variance_table["component"], variance_table["share_std"], "-s", ms=5, color=OKABE[5])   # std scree
style_axes(axes[2], "Raw versus standardised", "component", "variance share")   # shared style
axes[2].legend(["raw scale", "standardised"], fontsize=7.4, loc="upper right")   # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_pca_scree.png")                                     # save the scree figure

# ---------------------------------------------------------------- 8) figure: what standardisation changes
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))                # two panels
corr = np.corrcoef(X.T)                                            # correlation matrix of the raw features
im = axes[0].imshow(np.abs(corr), cmap="RdPu", vmin=0, vmax=1)     # absolute correlations
axes[0].set_xticks(range(len(features)), features, rotation=60, fontsize=7.2)   # feature labels
axes[0].set_yticks(range(len(features)), features, fontsize=7.2)   # feature labels
axes[0].grid(False)                                                # a heatmap needs no grid
axes[0].set_title("Feature correlations")                          # panel title
plt.colorbar(im, ax=axes[0], fraction=0.046, label="|corr|")       # colour bar
loadings = pd.DataFrame({"feature": features,                      # feature names
                         "PC1_std": np.round(pca_std.components_[0], 3),   # standardised loading
                         "PC1_raw": np.round(pca_raw.components_[0], 3)})   # raw loading
axes[1].barh(loadings["feature"], loadings["PC1_std"], color=OKABE[2], alpha=0.9, label="standardised")   # bars
axes[1].barh(loadings["feature"], loadings["PC1_raw"], color=OKABE[5], alpha=0.45, label="raw scale")   # overlay
axes[1].axvline(0, color="#66717D", lw=0.9)                        # zero reference
style_axes(axes[1], "PC1 loadings under two scalings", "loading", "")   # shared style
axes[1].legend(fontsize=7.2, loc="lower right")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_pca_scaling.png")                                   # save the scaling figure
print("done: PCA directions, scree and scaling")                   # closing log
