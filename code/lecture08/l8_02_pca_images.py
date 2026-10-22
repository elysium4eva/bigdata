"""l8_02_pca_images.py -- PCA as image compression: patches, components and reconstruction.

Every line carries an English comment. The script turns a synthetic image into small
patches, learns a PCA basis on the patches, reconstructs the image from a growing
number of components, and records the reconstruction error so the deck can show the
classic compression figure.
"""
import numpy as np                                                 # arrays for the patches
import pandas as pd                                                # tabular error tables
import matplotlib.pyplot as plt                                    # reconstruction panels
from matplotlib.colors import ListedColormap                       # not used but kept for consistent imports
from sklearn.decomposition import PCA, IncrementalPCA              # batch and incremental PCA
from _common_l8 import (RNG, DAT, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) build a synthetic image
def make_image(size=128):                                          # a smooth image with edges and texture
    """Return a grey-scale synthetic image with gradients, a disc and noise."""   # short docstring
    yy, xx = np.mgrid[0:size, 0:size]                              # pixel coordinates
    base = 0.45 + 0.35 * np.sin(xx / 9.0) * np.cos(yy / 11.0)       # low-frequency structure
    disc = ((xx - size * 0.62) ** 2 + (yy - size * 0.4) ** 2) < (size * 0.2) ** 2   # a bright disc
    stripes = 0.12 * np.sin((xx + yy) / 3.0)                       # high-frequency texture
    img = base + 0.35 * disc + stripes + RNG.normal(0, 0.03, (size, size))   # assemble the image
    return np.clip(img, 0, 1)                                      # keep it in [0, 1]

size, patch = 128, 8                                               # image size and patch size
img = make_image(size)                                             # the synthetic image
patches = []                                                       # patch matrix rows
for y in range(0, size, patch):                                    # loop over the rows of patches
    for x in range(0, size, patch):                                # loop over the columns of patches
        patches.append(img[y:y + patch, x:x + patch].ravel())      # flatten one patch into a row
P = np.array(patches)                                              # patch matrix: n_patches by 64
print("image:", img.shape, "| patches:", P.shape)                  # report the shapes
np.savez(DAT / "l8_patches.npz", patches=P, image=img)           # persist the data
print("[data]", DAT / "l8_patches.npz")                            # log the artefact

# ---------------------------------------------------------------- 2) fit the basis and reconstruct
pca = PCA(n_components=32).fit(P)                                  # keep up to 32 components
rows = []                                                          # reconstruction error per k
reconstructions = {}                                               # keep the images for plotting
for k in [2, 5, 10, 20, 32]:                                       # component counts to compare
    Z = pca.transform(P)[:, :k]                                    # project on the first k components
    back = pca.inverse_transform(np.hstack([Z, np.zeros((P.shape[0], 32 - k))]))   # reconstruct the patches
    recon = np.zeros_like(img)                                     # empty canvas
    i = 0                                                          # patch counter
    for y in range(0, size, patch):                                # rebuild the image row by row
        for x in range(0, size, patch):                            # and column by column
            recon[y:y + patch, x:x + patch] = back[i].reshape(patch, patch)   # paste the patch back
            i += 1                                                 # advance the counter
    reconstructions[k] = np.clip(recon, 0, 1)                      # keep the reconstruction
    mse = float(np.mean((img - recon) ** 2))                       # mean squared error
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))             # peak signal-to-noise ratio in dB
    rows.append({"components": k, "mse": round(mse, 6), "psnr_db": round(psnr, 2),   # the two errors
                 "compression_ratio": round(64.0 / k, 1)})         # float's per patch divided by k
    print("k=%2d | MSE %.5f | PSNR %.2f dB | ratio %.1f" % (k, mse, psnr, 64.0 / k))   # report
error_table = pd.DataFrame(rows)                                   # assemble the table
dump(error_table, "l8_pca_image_errors.csv")                       # persist it

# ---------------------------------------------------------------- 3) incremental PCA on the patches
ipca = IncrementalPCA(n_components=10, batch_size=32).fit(P)       # out-of-core fit on the same patches
gap = float(np.abs(np.sqrt(np.mean((P - ipca.inverse_transform(ipca.transform(P))) ** 2))   # incremental reconstruction error
                   - np.sqrt(np.mean((P - pca.inverse_transform(        # batch reconstruction error for the gap
                       np.hstack([pca.transform(P)[:, :10], np.zeros((P.shape[0], 22))]))) ** 2))))   # gap to batch
print("incremental versus batch RMSE gap: %.2e" % gap)              # should be tiny

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 5, figsize=(15.6, 3.5))                # original plus four reconstructions
axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)                   # the original image
axes[0].set_title("original")                                      # panel title
axes[0].axis("off")                                                # hide the axes
for ax, k in zip(axes[1:], [2, 5, 10, 32]):                        # the four reconstructions
    ax.imshow(reconstructions[k], cmap="gray", vmin=0, vmax=1)      # reconstructed image
    row = error_table.loc[error_table["components"] == k].iloc[0]   # the matching row
    ax.set_title("k=%d | PSNR %.1f dB" % (k, row["psnr_db"]))       # panel title with the score
    ax.axis("off")                                                 # hide the axes
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_pca_image.png")                                     # save the compression figure

# ---------------------------------------------------------------- 5) a second figure: error versus k
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))                # two panels
axes[0].plot(error_table["components"], error_table["psnr_db"], "-o", ms=5, color=OKABE[2])   # PSNR curve
style_axes(axes[0], "Reconstruction quality versus components", "components kept", "PSNR (dB)")   # style
axes[1].semilogy(error_table["components"], error_table["mse"], "-s", ms=5, color=OKABE[5])   # MSE curve
style_axes(axes[1], "Error falls fast then flattens", "components kept", "MSE (log scale)")   # style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_pca_image_curve.png")                               # save the error figure
print("done: PCA image compression, %d patches" % P.shape[0])        # closing log
