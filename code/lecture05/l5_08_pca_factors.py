"""l5_08_pca_factors.py -- principal components and latent factors on a macro panel.

Every line carries an English comment. The script simulates a monthly macro panel
driven by three latent factors, then shows the scree plot, the loadings, the score
projection and the reconstruction error, and finally interprets the components.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # four-panel figure
from sklearn.decomposition import PCA                              # principal component analysis
from sklearn.preprocessing import StandardScaler                   # standardise before decomposing
from _common_l5 import RNG, save, dump, style_axes, OKABE          # shared helpers

# ---------------------------------------------------------------- 1) simulate a macro panel with three factors
T = 240                                                            # twenty years of monthly data
t = np.arange(T)                                                   # time index in months
growth_factor = np.sin(2 * np.pi * t / 60) + 0.35 * np.sin(2 * np.pi * t / 18) + RNG.normal(0, 0.25, T)   # real cycle
price_factor = 0.8 * np.cos(2 * np.pi * t / 48) + RNG.normal(0, 0.3, T)   # inflation cycle
risk_factor = np.cumsum(RNG.normal(0, 0.12, T))                    # slow moving risk appetite trend
factors = np.c_[growth_factor, price_factor, risk_factor]          # stack the three latent drivers
loadings_true = np.array([                                          # true loadings of each observable series
    [0.9, 0.1, 0.2],                                               # industrial production follows growth
    [0.8, 0.2, 0.1],                                               # exports follow growth
    [0.1, 0.9, 0.0],                                               # CPI follows the price factor
    [0.2, 0.8, 0.1],                                               # producer prices follow prices
    [0.3, 0.1, 0.8],                                               # credit spread follows risk
    [0.2, 0.0, 0.9],                                               # VIX-like index follows risk
    [0.7, 0.1, 0.3],                                               # retail sales follow growth
    [0.1, 0.4, 0.6]])                                              # commodity index mixes prices and risk
cols = ["industrial_prod", "exports", "cpi", "ppi", "credit_spread", "vix", "retail_sales", "commodity"]   # column names
noise = RNG.normal(0, 0.35, (T, len(cols)))                        # idiosyncratic noise per series
panel = pd.DataFrame(factors @ loadings_true.T + noise, columns=cols)   # observables equal factors times loadings
panel["month"] = pd.date_range("2006-01-01", periods=T, freq="MS")   # attach a monthly date index
dump(panel, "l5_macro_panel.csv")                                  # persist the sample
print("macro panel:", panel[cols].shape, "| three latent factors")   # report the data shape

# ---------------------------------------------------------------- 2) standardise and decompose
X = panel[cols].values                                             # numeric matrix of observables
X_std = StandardScaler().fit_transform(X)                          # standardising is mandatory before PCA
pca = PCA().fit(X_std)                                             # keep all components for the scree plot
evr = pca.explained_variance_ratio_ * 100                          # share of variance explained by each component
cum = np.cumsum(evr)                                               # cumulative share of variance
scores = pca.transform(X_std)[:, :3]                               # keep the first three component scores
print("explained variance (first five):", np.round(evr[:5], 1), "| cumulative %.1f%%" % cum[4])   # report the scree

# ---------------------------------------------------------------- 3) loadings and reconstruction
loadings = pd.DataFrame(pca.components_[:3].T, index=cols,         # rows are series, columns are components
                        columns=["PC1", "PC2", "PC3"]).round(3)    # round for readability
dump(loadings.reset_index(names="series"), "l5_pca_loadings.csv")   # persist the loadings table with series names
print(loadings.to_string())                                        # print the loading table
recon_error = []                                                   # reconstruction error per component count
for k in range(1, len(cols) + 1):                                  # from one component to all of them
    pca_k = PCA(n_components=k).fit(X_std)                         # fit a truncated decomposition
    approx = pca_k.inverse_transform(pca_k.transform(X_std))       # project down and back up
    recon_error.append(float(np.mean((X_std - approx) ** 2)))      # mean squared reconstruction error
recon_table = pd.DataFrame({"n_components": list(range(1, len(cols) + 1)),   # component counts
                            "recon_mse": np.round(recon_error, 4)})   # matching errors
dump(recon_table, "l5_pca_reconstruction.csv")                     # persist the reconstruction table
dump(pd.DataFrame(scores, columns=["PC1", "PC2", "PC3"]).round(4), "l5_pca_scores.csv")   # persist the scores

# ---------------------------------------------------------------- 4) figure: four views of the decomposition
fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5))                # four panels in one row
axes[0].bar(range(1, len(evr) + 1), evr, color=OKABE[0], alpha=0.85, label="individual")   # scree bars
axes[0].plot(range(1, len(evr) + 1), cum, "-o", ms=4, color=OKABE[2], label="cumulative")   # cumulative curve
axes[0].axhline(80, color="#66717D", ls="--", lw=1.0)              # the common eighty per cent rule of thumb
style_axes(axes[0], "Scree: three factors explain most variance", "Component", "Variance explained (%)")   # shared style
axes[0].legend(fontsize=7.4, loc="center right")                   # compact legend
im = axes[1].imshow(loadings.values, cmap="RdBu_r", vmin=-1, vmax=1)   # loadings as a diverging heatmap
axes[1].set_xticks(range(3), loadings.columns, fontsize=8.4)       # component labels on the x axis
axes[1].set_yticks(range(len(cols)), cols, fontsize=7.4)           # series names on the y axis
axes[1].grid(False)                                                # a heatmap needs no grid
for i in range(loadings.shape[0]):                                 # annotate every cell
    for j in range(loadings.shape[1]):                             # loop over components
        axes[1].text(j, i, f"{loadings.values[i, j]:.2f}", ha="center", va="center", fontsize=6.6,   # print the loading
                     color="white" if abs(loadings.values[i, j]) > 0.55 else "#1E2630")   # contrast with the fill
axes[1].set_title("Loadings: which series drive which factor")     # panel title
plt.colorbar(im, ax=axes[1], fraction=0.046)                       # colour bar for the loadings
sc = axes[2].scatter(scores[:, 0], scores[:, 1], c=t, cmap="RdPu", s=13)   # score projection coloured by time
axes[2].axhline(0, color="#66717D", lw=0.8, ls="--")               # horizontal reference
axes[2].axvline(0, color="#66717D", lw=0.8, ls="--")               # vertical reference
style_axes(axes[2], "Score plot: the growth-price cycle", "PC1 score", "PC2 score")   # shared style
plt.colorbar(sc, ax=axes[2], fraction=0.046, label="month index")   # colour bar for time
axes[3].plot(recon_table["n_components"], recon_table["recon_mse"], "-o", ms=4, color=OKABE[5])   # reconstruction curve
axes[3].axvline(3, color="#66717D", ls="--", lw=1.0)               # the true number of factors
style_axes(axes[3], "Reconstruction error: where it flattens", "Components kept", "Mean squared error")   # shared style
axes[3].text(0.42, 0.85, "true k = 3", transform=axes[3].transAxes, fontsize=8.0, color="#1E2630")   # label the marker
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_pca_factors.png")                                   # save the decomposition figure
