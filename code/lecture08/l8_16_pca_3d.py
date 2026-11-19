"""l8_16_pca_3d.py -- where the first two principal components come from, drawn in 3-D.

Every line carries an English comment. The script generates 420 stores from two latent
factors, standardises the three business metrics, eigen-decomposes the correlation matrix,
and then shows the sampling geometry twice: the left panel compresses the cloud onto the
PC1 axis, the right panel compresses it onto the PC1-PC2 plane. The variance shares in the
suptitle are computed from the data, never typed in.
"""
import shutil                                                      # copy the finished figure into the slide folder
from pathlib import Path                                           # portable path handling

import matplotlib                                                  # choose a backend before importing pyplot
matplotlib.use("Agg")                                              # headless backend: render without a display
import matplotlib.pyplot as plt                                    # figure and axes API
import matplotlib.patheffects as pe                                # white halo keeps labels readable on dots
from mpl_toolkits.mplot3d.art3d import Line3DCollection           # many thin 3-D segments in one artist
from mpl_toolkits.mplot3d.art3d import Poly3DCollection           # the translucent plane patch as one polygon
import numpy as np                                                 # arrays, eigen-decomposition and checks
import pandas as pd                                                # the sample table and CSV export
from mpl_toolkits.mplot3d import Axes3D                            # registers the 3-D projection (side effect)

SEED = 20260808                                                    # fixed seed: the same 420 stores on every run
N = 420                                                            # number of observations in the sample
RNG = np.random.default_rng(SEED)                                  # one generator for factors and noise
WEB = Path(__file__).resolve().parents[2]                          # .../website: this file lives in website/code/lecture08
BASE = WEB.parent                                                  # the course root that holds the lecture folders
FIG = WEB / "figures" / "lecture08"                                # figure output folder for lecture 8
DAT = WEB / "data" / "lecture08"                                   # data output folder for lecture 8
DECK = BASE / "lecture08-2026-szu" / "figures"                     # the lecture-8 deck folder
for _d in (FIG, DAT, DECK):                                        # make sure every output folder exists
    _d.mkdir(parents=True, exist_ok=True)                          # create it when missing

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK-capable font stack
plt.rcParams["axes.unicode_minus"] = False                         # draw real minus signs, not tofu boxes
plt.rcParams["savefig.dpi"] = 200                                  # slide resolution when saving
plt.rcParams["savefig.bbox"] = "tight"                             # trim surrounding white space
plt.rcParams["figure.dpi"] = 130                                   # comfortable on-screen draft resolution
INK = "#1E2630"                                                    # near-black used for all text
GREY = "#AEB8C2"                                                   # light grey for the raw point cloud
BLUE = "#0072B2"                                                   # blue for PC1, its line and its projections
GREEN = "#009E73"                                                  # green for PC2 and the plane projection dots
ORANGE = "#D55E00"                                                 # vermillion for the PC1 arrow (maximum contrast)
PANEL = "#F7F8FA"                                                  # pale pane fill inside the 3-D box

# ---------------------------------------------------------------- 1) three business metrics from two latent factors
f_scale = RNG.normal(0.0, 1.0, N)                                  # latent factor 1: overall store size / volume
f_eff = RNG.normal(0.0, 1.0, N)                                    # latent factor 2: how busy the store is per square metre
area = 120.0 + 45.0 * f_scale + 10.0 * f_eff + RNG.normal(0.0, 28.0, N)        # floor area in square metres
traffic = 900.0 + 260.0 * f_scale + 240.0 * f_eff + RNG.normal(0.0, 120.0, N)   # footfall: visitors per day
sales = 22000.0 + 5000.0 * f_scale + 5200.0 * f_eff + RNG.normal(0.0, 2600.0, N)   # sales in yuan per day
raw = pd.DataFrame({"门店面积": area, "客流": traffic, "销售额": sales})   # keep the three raw metrics side by side
print("样本量 n = %d；原始量纲（均值 / 标准差）：" % N)                # report the sample size and the raw scales
for _name in raw.columns:                                          # one line per metric
    print("  %-6s mean %10.1f   sd %8.1f" % (_name, raw[_name].mean(), raw[_name].std(ddof=1)))   # raw mean and sd

X = raw.to_numpy(dtype=float)                                      # the 420 x 3 raw matrix
mu = X.mean(axis=0)                                                # column means, removed before PCA
sd = X.std(axis=0, ddof=1)                                         # sample standard deviations (match np.cov)
Z = (X - mu) / sd                                                  # standardised matrix: unit variance per column
print("相关系数矩阵（也就是协方差矩阵）:")                           # the matrix that PCA actually diagonalises
print(pd.DataFrame(np.cov(Z, rowvar=False),                        # the 3 x 3 correlation matrix of the data
                   index=raw.columns, columns=raw.columns).round(3).to_string())   # print it with readable labels
Z_std = pd.DataFrame(Z, columns=["门店面积", "客流", "销售额"]).round(4)   # tidy table of the standardised data
Z_std.to_csv(DAT / "l8_pca_3d.csv", index=False, encoding="utf-8-sig")   # Excel-friendly UTF-8 CSV
print("[data]", DAT / "l8_pca_3d.csv")                             # log the data artefact

# ---------------------------------------------------------------- 2) eigen-decomposition of the covariance matrix
C = np.cov(Z, rowvar=False)                                        # covariance of standardised data = correlation matrix
eigval, eigvec = np.linalg.eigh(C)                                 # symmetric solver: eigenvalues ascending
order = np.argsort(eigval)[::-1]                                   # put the largest eigenvalue first
eigval = eigval[order]                                             # reorder the eigenvalues descending
eigvec = eigvec[:, order]                                          # reorder the matching eigenvectors as columns
lead = np.abs(eigvec).argmax(axis=0)                               # index of the dominant loading in each column
sign = np.sign(eigvec[lead, np.arange(3)])                         # make that dominant loading positive
sign[sign == 0.0] = 1.0                                            # guard against an exactly zero loading
eigvec = eigvec * sign                                             # flip whole columns, never single entries
share = eigval / eigval.sum()                                      # variance explained: eigenvalue / trace
PC1, PC2, PC3 = eigvec[:, 0], eigvec[:, 1], eigvec[:, 2]           # the three unit directions in standardised space
print("\n特征值:", np.round(eigval, 4), "(sum = %.4f)" % eigval.sum())   # eigenvalues should sum to the trace
print("方差解释率:", np.round(share, 4), "| 累计:", np.round(np.cumsum(share), 4))   # shares and their running sum
print("主成分方向（载荷，行 = 门店面积 / 客流 / 销售额）:")           # loadings are the eigenvector entries
for _k, _v in enumerate(eigvec.T, start=1):                        # one line per component
    print("  PC%d  %s" % (_k, np.round(_v, 4)))                    # print the direction

vectors = pd.DataFrame({"component": ["PC1", "PC2", "PC3"],        # component label used by the deck
                        "eigenvalue": np.round(eigval, 4),         # variance carried by the component
                        "var_explained": np.round(share, 4),       # share of the total variance
                        "loading_x": np.round(eigvec[0], 4),       # loading on floor area
                        "loading_y": np.round(eigvec[1], 4),       # loading on footfall
                        "loading_z": np.round(eigvec[2], 4)})      # loading on sales
vectors.to_csv(DAT / "l8_pca_3d_vectors.csv", index=False, encoding="utf-8-sig")   # persist the components
print("[data]", DAT / "l8_pca_3d_vectors.csv")                     # log the vector artefact

# ---------------------------------------------------------------- 3) the two projections that the figure shows
proj1 = Z @ PC1                                                    # signed coordinate of each point on the PC1 axis
proj2 = Z @ PC2                                                    # signed coordinate of each point on the PC2 axis
on_line = np.outer(proj1, PC1)                                     # every point compressed onto the PC1 line
on_plane = on_line + np.outer(proj2, PC2)                          # every point compressed onto the PC1-PC2 plane
var_line = float(np.var(proj1, ddof=1))                            # measured variance of the 1-D projection
var_plane = float(np.var(proj1, ddof=1) + np.var(proj2, ddof=1))   # variance kept by the 2-D projection
print("\n投影到 PC1 直线上的方差 = %.4f（= 最大特征值 %.4f）" % (var_line, eigval[0]))   # check the 1-D claim
print("投影到 PC1-PC2 平面上的方差 = %.4f（= 前两个特征值之和 %.4f）" % (var_plane, eigval[:2].sum()))   # check the 2-D claim

# ---------------------------------------------------------------- 4) a measured check that no direction beats PC1 / PC2
vrng = np.random.default_rng(SEED + 7)                             # separate generator, does not disturb the sample
dirs = vrng.normal(size=(40000, 3))                                # 40000 random directions in R^3
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)                # rescale them all to unit length
var_random = np.einsum("ni,ij,nj->n", dirs, C, dirs)               # projected variance w' C w for every direction
best1 = float(var_random.max())                                    # the best random direction, unconstrained
orth = dirs - np.outer(dirs @ PC1, PC1)                            # remove the PC1 part from every direction
orth /= np.linalg.norm(orth, axis=1, keepdims=True)                # rescale to unit length: now orthogonal to PC1
var_orth = np.einsum("ni,ij,nj->n", orth, C, orth)                 # projected variance inside the PC1 plane
best2 = float(var_orth.max())                                      # the best direction that stays orthogonal to PC1
print("40000 个随机方向的最大投影方差 %.6f <= PC1 的 %.6f" % (best1, eigval[0]))   # PC1 is the unconstrained maximum
print("其中与 PC1 正交者最大投影方差 %.6f <= PC2 的 %.6f" % (best2, eigval[1]))   # PC2 wins the remaining variance

# ---------------------------------------------------------------- 5) shared 3-D panel styling
LIM = 3.55                                                         # symmetric axis half-width in standard deviations
LINE = LIM / float(np.abs(PC1).max())                              # how far the PC1 line runs before leaving the cube
LABELS = ["门店面积 (z)", "客流 (z)", "销售额 (z)"]                  # axis labels; (z) means standardised


def clip_halfplane(poly, normal, bound):                           # Sutherland-Hodgman clip of one convex polygon
    """Keep the part of a convex polygon where normal . p <= bound."""   # short docstring
    out = []                                                       # vertices of the clipped polygon
    for i in range(len(poly)):                                     # walk over the polygon edges
        p, q = poly[i], poly[(i + 1) % len(poly)]                   # current vertex and its successor
        dp = float(p @ normal) - bound                             # signed distance of p to the cut line
        dq = float(q @ normal) - bound                             # signed distance of q to the cut line
        if dp <= 0.0:                                              # the current vertex is inside
            out.append(p)                                          # so keep it
        if (dp < 0.0 < dq) or (dq < 0.0 < dp):                     # the edge crosses the cut line
            out.append(p + (dp / (dp - dq)) * (q - p))             # so add the crossing point
    return out                                                     # the clipped polygon


def plane_polygon(direction1, direction2, limit):                  # the part of a plane that lies inside the cube
    """Clip the plane spanned by two directions to the cube [-limit, limit]^3."""   # short docstring
    poly = [np.array([-6.6, -3.4]), np.array([6.6, -3.4]),          # start from a square in (a, b) coordinates
            np.array([6.6, 3.4]), np.array([-6.6, 3.4])]            # large enough to contain the cube section
    for j in range(3):                                             # one pair of cube faces per axis
        for s in (-1.0, 1.0):                                      # both faces of that pair
            normal = s * np.array([direction1[j], direction2[j]])   # half-plane normal in (a, b) space
            poly = clip_halfplane(poly, normal, limit)              # keep s * (a*d1[j] + b*d2[j]) <= limit
    return np.array([a * direction1 + b * direction2 for a, b in poly])   # corners mapped back to R^3


def style_3d(ax, title, elev, azim):                               # configure one 3-D panel
    """Apply the shared look to a 3-D axes."""                       # short docstring
    ax.set_xlim(-LIM, LIM)                                         # same range on x
    ax.set_ylim(-LIM, LIM)                                         # same range on y
    ax.set_zlim(-LIM, LIM)                                         # same range on z
    ax.set_box_aspect((1.0, 1.0, 1.0))                             # equal edge lengths: no stretched geometry
    ax.set_proj_type("ortho")                                      # orthographic view keeps parallel lines parallel
    ax.set_xlabel(LABELS[0], fontsize=8.6, labelpad=1)             # short axis label
    ax.set_ylabel(LABELS[1], fontsize=8.6, labelpad=1)             # short axis label
    ax.set_zlabel(LABELS[2], fontsize=8.6, labelpad=1)             # short axis label
    ax.tick_params(labelsize=7.0, pad=-1, colors="#5B6672")        # small ticks, tucked towards the box
    ax.set_title(title, fontsize=11.8, color=INK, pad=0)           # panel title
    ax.view_init(elev=elev, azim=azim)                             # camera position chosen for readable geometry
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):                    # restyle the three coordinate panes
        axis.pane.set_facecolor(PANEL)                             # very pale pane background
        axis.pane.set_edgecolor("#DDE3EA")                         # light pane border
    ax.grid(True)                                                  # keep faint grid lines for orientation
    return ax                                                      # return the axes for chaining


def note(ax, text, y=0.97, x=0.015):                               # put an explanatory note inside a panel
    """Draw a small multi-line note with a translucent box."""       # short docstring
    ax.text2D(x, y, text, transform=ax.transAxes, va="top", ha="left", fontsize=8.4, color=INK, linespacing=1.45,   # place and style
              bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#D6DCE4", lw=0.7, alpha=0.9))   # pale box
    return ax                                                      # return the axes for chaining


def arrow(ax, direction, length, colour, label):                   # draw one component arrow from the mean
    """Draw a labelled arrow from the origin along a unit direction."""   # short docstring
    tip = direction * length                                       # arrow tip at length standard deviations
    ax.quiver(0.0, 0.0, 0.0, tip[0], tip[1], tip[2], color=colour,  # start at the mean point (the origin)
              lw=3.6, arrow_length_ratio=0.14, normalize=False)    # thick shaft, visible head, no rescaling
    txt = ax.text(tip[0] * 1.11, tip[1] * 1.11, tip[2] * 1.13, label,   # label a little beyond the arrow head
                  color=colour, fontsize=13.5, fontweight="bold", ha="center", va="center", zorder=20)   # bold, centred
    txt.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])   # white halo over any background
    return ax                                                      # return the axes for chaining


def drop_lines(ax, start, stop, colour, alpha, lw):                # many thin "compression" segments
    """Add segments from the observations to their projections."""   # short docstring
    segments = np.stack([start, stop], axis=1)                      # shape (n, 2, 3): one segment per point
    coll = Line3DCollection(segments, colors=colour, linewidths=lw, alpha=alpha)   # build the collection
    ax.add_collection3d(coll)                                      # attach it to the 3-D axes
    return ax                                                      # return the axes for chaining


# ---------------------------------------------------------------- 6) figure with the two sampling panels
fig = plt.figure(figsize=(13.2, 5.7))                              # two 3-D panels side by side
ax1 = fig.add_subplot(1, 2, 1, projection="3d")                    # left panel: compress onto PC1
ax2 = fig.add_subplot(1, 2, 2, projection="3d")                    # right panel: compress onto the PC1-PC2 plane
VIEW = (22, -38)                                                   # one camera for both panels: PC1 stays long on screen
ARROW1, ARROW2 = 2.9, 2.5                                          # arrow lengths in standard deviations (2-3 sd)

# --- panel 1: compress the cloud onto the PC1 line
ax1.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=9, color=GREY, alpha=0.55, depthshade=False, edgecolors="none")   # raw cloud
ax1.plot([-LINE * PC1[0], LINE * PC1[0]],                           # the PC1 line, from one cube corner to the other
         [-LINE * PC1[1], LINE * PC1[1]],                           # its y extent
         [-LINE * PC1[2], LINE * PC1[2]],                           # its z extent
         color=BLUE, lw=1.1, alpha=0.7, zorder=4)                   # a thin blue guide line
drop_lines(ax1, Z, on_line, BLUE, 0.15, 0.4)                       # faint connectors: point -> its projection
ax1.scatter(on_line[:, 0], on_line[:, 1], on_line[:, 2], s=10, color=BLUE, alpha=0.6,   # the 1-D sample itself
            depthshade=False, edgecolors="none", zorder=10)         # every point flattened onto the line
ax1.scatter([0.0], [0.0], [0.0], s=62, color=INK, edgecolors="white", linewidths=0.9,   # the mean of the sample
            depthshade=False, zorder=15)                            # black dot at the origin
ax1.text(-0.95, -0.95, 0.45, "均值点", fontsize=9.0, color=INK, ha="right", va="center", zorder=15,   # name the mean
         path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])   # white halo over the cloud
arrow(ax1, PC1, ARROW1, ORANGE, "PC1")                              # the leading component, 2.9 sd long
style_3d(ax1, "第一个主成分怎么选", *VIEW)                           # title and camera
note(ax1, "PC1：让投影方差最大的方向\n就是协方差矩阵最大的特征向量\n把 %d 个点压到这条线上：保留 %.1f%% 的方差"   # explain PC1
     % (N, 100 * share[0]))                                        # shares come from the data, not the text

# --- panel 2: compress the cloud onto the PC1-PC2 plane
ax2.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=8, color=GREY, alpha=0.40, depthshade=False, edgecolors="none")   # the same cloud
patch = Poly3DCollection([plane_polygon(PC1, PC2, LIM)], facecolor="#6FAFD6",   # the plane, clipped to the cube
                         edgecolor="none", alpha=0.28)              # translucent so the cloud stays visible
ax2.add_collection3d(patch)                                        # add the plane to the 3-D scene
drop_lines(ax2, Z, on_plane, "#3F7FA8", 0.13, 0.35)                # faint connectors: point -> its plane projection
ax2.scatter(on_plane[:, 0], on_plane[:, 1], on_plane[:, 2], s=9, color=GREEN, alpha=0.6,   # the 2-D sample
            depthshade=False, edgecolors="none", zorder=10)         # dots lying on the plane
ax2.scatter([0.0], [0.0], [0.0], s=62, color=INK, edgecolors="white", linewidths=0.9,   # the mean of the sample
            depthshade=False, zorder=15)                            # black dot at the origin
ax2.text(-0.95, -0.95, 0.45, "均值点", fontsize=9.0, color=INK, ha="right", va="center", zorder=15,   # name the mean
         path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])   # white halo over the plane
arrow(ax2, PC1, ARROW1, ORANGE, "PC1")                              # first component, same length as panel 1
arrow(ax2, PC2, ARROW2, GREEN, "PC2")                               # second component, orthogonal to PC1
style_3d(ax2, "第二个主成分怎么选", *VIEW)                           # title and camera
note(ax2, "PC2：在与 PC1 正交的方向里\n挑剩余方差最大的那一个\n两方向张成平面，累计保留 %.1f%% 的方差"   # explain PC2
     % (100 * share[:2].sum()))                                    # cumulative share, computed above

fig.suptitle("三维 PCA 的投影几何：PC1 保留 %.1f%%、PC2 保留 %.1f%%、PC3 只剩 %.1f%% 的方差（n = %d）"   # headline with measured shares
             % (100 * share[0], 100 * share[1], 100 * share[2], N), fontsize=13.2, color=INK, y=0.995)   # data-driven title
fig.subplots_adjust(left=0.01, right=0.995, top=0.90, bottom=0.01, wspace=0.0)   # tighten the two panels
PNG = FIG / "fig_pca_3d.png"                                       # the figure path required by the deck
AXIS_TEXT = [ax.xaxis.label for ax in (ax1, ax2)] + [ax.yaxis.label for ax in (ax1, ax2)]   # 3-D axis labels
AXIS_TEXT += [ax.zaxis.label for ax in (ax1, ax2)]                 # the vertical labels stick out to the right
fig.savefig(PNG, facecolor="white", bbox_extra_artists=AXIS_TEXT)  # keep the outlying labels inside the crop
plt.close(fig)                                                     # release the figure from memory
print("[figure]", PNG)                                             # log the figure artefact

shutil.copyfile(PNG, DECK / "fig_pca_3d.png")                      # place the same PNG in the slide folder
print("[copy]", DECK / "fig_pca_3d.png")                           # log the copy
print("done: 3-D PCA sampling geometry with PC1 and PC2")           # closing log
