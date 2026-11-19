"""l8_18_ff_pca_compare.py -- statistical (PCA) versus observable (Fama-French) factors.

Every line carries an English comment. The script extracts principal components from the
25 size and book-to-market portfolios, estimates the three-factor model on exactly the
same excess returns, and measures how much of each statistical factor the observable
factors span. The comparison is deliberately on one sample so the numbers are comparable.
"""
import shutil                                                      # copy the finished figure into the deck folder
import numpy as np                                                 # eigenvalues, scores and correlations
import pandas as pd                                                # read and write the return panels
import matplotlib.pyplot as plt                                    # the three-panel comparison figure
import statsmodels.api as sm                                       # OLS with t values and R squared
from _common_l8 import DAT, FIG, ROOT, save, dump, style_axes, OKABE   # shared paths, style and palette

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK-capable font stack
plt.rcParams["axes.unicode_minus"] = False                        # true minus signs instead of tofu boxes
plt.rcParams["savefig.dpi"] = 200                                  # slide-ready resolution
plt.rcParams["savefig.bbox"] = "tight"                             # trim the surrounding margins

START = "1990-01"                                                  # post-1989 window: modern era, calmer than the 1930s
ff3 = pd.read_csv(DAT / "ff3_factors_monthly.csv")                 # three factors, percent per month
ff25 = pd.read_csv(DAT / "ff25_portfolios_monthly.csv")            # 25 size and book-to-market portfolios
ff25 = ff25.drop(columns=[c for c in ("date", "RF") if c in ff25.columns])   # keep the month key plus the 25 returns only
panel = ff3.merge(ff25, on="yyyymm", how="inner")                  # join both files on the month key
panel = panel[panel["date"] >= START].reset_index(drop=True)       # restrict to the stated sample window
COLS = [c for c in ff25.columns if c != "yyyymm"]                  # the 25 portfolio column names
assert len(COLS) == 25                                             # the grid must be complete
X = panel[COLS].to_numpy(dtype=float)                              # 25 excess-return series, already in percent
T = X.shape[0]                                                     # number of monthly observations

# ---------------------------------------------------------------- 1) covariance PCA on the 25 portfolios
# Covariance (not correlation) PCA is the textbook choice here: all 25 columns are the same
# object measured in the same unit (percent excess return per month), so standardising would
# give tiny-variance corner portfolios the same weight as the market-wide level factor and
# would exaggerate noise. Covariance PCA keeps the economically meaningful common variation.
Xc = X - X.mean(axis=0)                                            # centre every portfolio before the decomposition
cov = (Xc.T @ Xc) / (T - 1)                                        # sample covariance in percent squared
eigval, eigvec = np.linalg.eigh(cov)                               # symmetric eigendecomposition of the covariance
order = np.argsort(eigval)[::-1]                                   # sort components from largest to smallest
eigval = eigval[order]                                             # eigenvalues in descending order
eigvec = eigvec[:, order]                                          # eigenvectors as columns, same order
flip = np.sign(eigvec[np.abs(eigvec).argmax(axis=0), np.arange(eigvec.shape[1])])   # sign of the largest loading
flip[flip == 0] = 1.0                                              # guard against an exact zero sign
eigvec = eigvec * flip                                             # orient every PC towards its dominant positive loading
scores = Xc @ eigvec                                               # principal-component return series
var_expl = eigval / eigval.sum()                                   # share of total variance per component
cum_expl = np.cumsum(var_expl)                                     # cumulative share of total variance
n80 = int(np.searchsorted(cum_expl, 0.80) + 1)                     # components needed to pass 80 percent
n90 = int(np.searchsorted(cum_expl, 0.90) + 1)                     # components needed to pass 90 percent

var_tab = pd.DataFrame({"component": ["PC%d" % (k + 1) for k in range(len(eigval))],   # component labels
                        "eigenvalue": np.round(eigval, 6),         # variance carried by each component
                        "var_explained": np.round(var_expl, 6),    # share of total variance
                        "cumulative": np.round(cum_expl, 6)})      # running share of total variance
dump(var_tab, "l8_ff_pca_variance.csv")                            # persist the variance table
load_tab = pd.DataFrame(eigvec[:, :3], columns=["PC1", "PC2", "PC3"])   # the first three loading vectors
load_tab.insert(0, "portfolio", COLS)                              # restore the portfolio labels
dump(load_tab.round(6), "l8_ff_pca_loadings.csv")                  # persist the loadings

# ---------------------------------------------------------------- 2) Fama-French three-factor regressions
fac = panel[["Mkt-RF", "SMB", "HML"]].to_numpy(dtype=float)        # the three observable factors
Xf = sm.add_constant(fac)                                          # design matrix with an intercept
rows = []                                                          # one result row per portfolio
for j, name in enumerate(COLS):                                    # loop over the 25 size and BM cells
    fit = sm.OLS(X[:, j], Xf).fit()                                # excess return on market, SMB and HML
    rows.append({"portfolio": name,                               # portfolio label
                 "alpha": float(fit.params[0]),                    # monthly alpha in percent
                 "alpha_t": float(fit.tvalues[0]),                 # alpha t statistic
                 "b_mkt": float(fit.params[1]),                    # market beta
                 "b_smb": float(fit.params[2]),                    # size beta
                 "b_hml": float(fit.params[3]),                    # value beta
                 "r2": float(fit.rsquared)})                       # explained share of variance
ff = pd.DataFrame(rows)                                            # tidy regression table
dump(ff.round(6), "l8_ff_regression.csv")                          # persist the regression table
mean_r2_ff3 = float(ff["r2"].mean())                               # average R squared across the 25 portfolios
mean_abs_alpha = float(ff["alpha"].abs().mean())                   # average absolute pricing error

# ---------------------------------------------------------------- 3) how far do the PCs lie inside the FF3 span
pc_rows = []                                                       # one result row per principal component
for k in range(3):                                                 # PC1, PC2 and PC3
    fit = sm.OLS(scores[:, k], Xf).fit()                           # project the PC return on the same three factors
    pc_rows.append({"component": "PC%d" % (k + 1),                 # component label
                    "r2": float(fit.rsquared),                     # share of the PC spanned by FF3
                    "corr_mkt": float(np.corrcoef(scores[:, k], fac[:, 0])[0, 1]),   # correlation with the market
                    "corr_smb": float(np.corrcoef(scores[:, k], fac[:, 1])[0, 1]),   # correlation with SMB
                    "corr_hml": float(np.corrcoef(scores[:, k], fac[:, 2])[0, 1])})  # correlation with HML
pcff = pd.DataFrame(pc_rows)                                       # tidy comparison table
dump(pcff.round(6), "l8_ff_pc_on_factors.csv")                     # persist the comparison table

# ---------------------------------------------------------------- 4) compact printed summary
top3 = load_tab.sort_values("PC1", ascending=False).head(3)        # three largest PC1 loadings
print("sample window %s .. %s | %d months" % (panel["date"].iloc[0][:7], panel["date"].iloc[-1][:7], T))   # window
print("variance explained PC1..PC5 (%):", np.round(var_expl[:5] * 100, 2))   # leading variance shares
print("PCs to reach 80%%: %d | 90%%: %d" % (n80, n90))              # concentration of the spectrum
print("FF3: mean R2 over 25 portfolios %.4f | mean |alpha| %.4f %%/month" % (mean_r2_ff3, mean_abs_alpha))   # average fit and pricing error
print("FF3: 3 PCs cumulative variance %.4f vs mean portfolio R2 %.4f" % (cum_expl[2], mean_r2_ff3))   # honest compare
print("PC1..PC3 on FF3, R2:", np.round(pcff["r2"].to_numpy(), 4))   # spanned share per PC
print("PC1..PC3 vs factors, correlation matrix:\n", pcff[["corr_mkt", "corr_smb", "corr_hml"]].round(3).to_string(index=False))   # correlations
print("largest PC1 loadings:\n", top3.to_string(index=False))      # what PC1 loads on most

# ---------------------------------------------------------------- 5) figure: scree, loadings and the comparison
fig = plt.figure(figsize=(14.8, 4.7))                              # wide three-panel slide figure
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.34)   # room for labels in every panel
ax0 = fig.add_subplot(gs[0, 0])                                    # panel (a): scree plus cumulative curve
ax0b = ax0.twinx()                                                 # second y axis for the cumulative share
k_show = 10                                                        # show the first ten components
ax0.bar(np.arange(1, k_show + 1), var_expl[:k_show] * 100, color=OKABE[1], alpha=0.8)   # individual shares
ax0b.plot(np.arange(1, k_show + 1), cum_expl[:k_show] * 100, color=OKABE[5], marker="o", lw=1.7)   # cumulative
ax0b.axhline(80, color="#66717D", ls="--", lw=0.9)                 # 80 percent reference line
ax0b.axhline(90, color="#66717D", ls=":", lw=0.9)                  # 90 percent reference line
ax0.text(1.0, var_expl[0] * 100 * 0.5, "PC1 %.1f%%" % (var_expl[0] * 100), ha="center",   # label the dominant bar
         va="center", rotation=90, fontsize=9.0, color="white")    # white text sits on the bar
ax0.axvline(n80 + 0.5, color=OKABE[5], ls="--", lw=0.8, alpha=0.6)   # mark the 80 percent cut
ax0.axvline(n90 + 0.5, color=OKABE[5], ls=":", lw=0.8, alpha=0.6)   # mark the 90 percent cut
ax0b.annotate("80%% at k=%d\n90%% at k=%d" % (n80, n90), xy=(0.97, 0.05), xycoords="axes fraction",   # cut point note
              ha="right", va="bottom", fontsize=8.4, color="#1E2630")   # place it in the lower right corner
style_axes(ax0, "(a) PCA spectrum on 25 portfolios", "principal component", "variance explained (%)")   # panel title
ax0b.set_ylabel("cumulative variance (%)")                         # label the cumulative axis
ax0b.set_ylim(0, 100)                                              # the cumulative curve is a percentage
ax0b.grid(False)                                                   # avoid a doubled grid
ax0b.spines["right"].set_visible(True)                             # keep the right axis line visible
h1, l1 = ax0.get_legend_handles_labels()                           # individual-bar handles
h2, l2 = ax0b.get_legend_handles_labels()                          # cumulative-line handles
ax0.legend(h1 + h2, ["individual", "cumulative"], loc="center right", frameon=False)   # compact legend

sub = gs[0, 1].subgridspec(3, 1, hspace=0.42)                      # panel (b): three stacked 5 by 5 loading maps
grid = eigvec[:, :3].T.reshape(3, 5, 5)                            # rows ME1..ME5, columns BM1..BM5 per PC
lim = float(np.abs(grid).max())                                    # common symmetric colour limits
b_axes = []                                                        # keep the axes for one shared colour bar
for k in range(3):                                                 # one map per principal component
    axk = fig.add_subplot(sub[k, 0])                               # slot for this component
    im = axk.imshow(grid[k], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")   # signed loadings
    axk.set_xticks(range(5), ["Lo", "BM2", "BM3", "BM4", "Hi"] if k == 2 else [""] * 5)   # BM labels at the bottom
    axk.set_yticks(range(5), ["ME1", "ME2", "ME3", "ME4", "ME5"])  # size labels on every map
    axk.set_ylabel("PC%d" % (k + 1), fontsize=9.2, rotation=0, labelpad=20)   # component name on the left
    axk.grid(False)                                                # a heatmap needs no grid
    for i in range(5):                                             # annotate the five size rows
        for j in range(5):                                         # annotate the five BM columns
            v = grid[k, i, j]                                      # the loading in this cell
            axk.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=5.6,   # small numeric label
                     color="white" if abs(v) > 0.62 * lim else "#1E2630")   # contrast against the cell colour
    b_axes.append(axk)                                             # remember the axes
fig.colorbar(im, ax=b_axes, fraction=0.035, pad=0.02)              # one colour bar shared by the three maps
b_axes[0].set_title("(b) PC loadings across size and book-to-market", fontsize=12.0, pad=6)   # panel title

ax2 = fig.add_subplot(gs[0, 2])                                    # panel (c): the honest comparison bars
labels = ["FF3 mean R2\n(25 portfolios)", "3 PCs cum.\nvariance", "PC1\non FF3", "PC2\non FF3", "PC3\non FF3"]   # bar labels
vals = [mean_r2_ff3, float(cum_expl[2]), float(pcff["r2"].iloc[0]), float(pcff["r2"].iloc[1]), float(pcff["r2"].iloc[2])]   # bar heights
colors = [OKABE[4], OKABE[2], OKABE[0], OKABE[1], OKABE[6]]        # distinct colour per bar
ax2.bar(range(5), vals, color=colors, alpha=0.9, edgecolor="white")   # the five comparison bars
ax2.set_ylim(0, 1.08)                                              # all quantities are shares in [0, 1]
for i, v in enumerate(vals):                                       # label every bar with its value
    ax2.text(i, v + 0.02, "%.3f" % v, ha="center", va="bottom", fontsize=8.6)   # value above the bar
ax2.set_xticks(range(5), labels, fontsize=7.9)                     # two-line tick labels
ax2.axhline(mean_r2_ff3, color=OKABE[4], ls="--", lw=0.9, alpha=0.7)   # reference at the FF3 fit
style_axes(ax2, "(c) What each model explains", "", "share of variance")   # panel title and axis label
fig.subplots_adjust(left=0.045, right=0.985, top=0.88, bottom=0.17)   # explicit margins; tight_layout cannot handle the colorbar
save(fig, "fig_ff_vs_pca.png")                                     # save into website/figures/lecture08

# ---------------------------------------------------------------- 6) mirror the figure into the LaTeX deck
deck_fig = ROOT.parent / "lecture08-2026-szu" / "figures"          # the deck's own figure folder
deck_fig.mkdir(parents=True, exist_ok=True)                        # create it when missing
shutil.copy2(FIG / "fig_ff_vs_pca.png", deck_fig / "fig_ff_vs_pca.png")   # byte-identical copy for the slides
print("[copy]", deck_fig / "fig_ff_vs_pca.png")                    # log the mirrored path
print("done: PCA versus the Fama-French three-factor model")       # closing log
