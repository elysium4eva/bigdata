"""l8_19_ff_timeseries.py -- observable versus statistical factors on one monthly axis.

Every line carries an English comment. The script extracts the first three principal
components from the 25 size and book-to-market portfolios and plots the returns of their
factor-mimicking portfolios next to the Fama-French factors, so a slide can show that the
statistical factors move like the observable ones. It writes one figure, three tidy CSVs
and a short summary.
"""
import shutil                                                      # copy the figure into the deck folder
import numpy as np                                                 # eigenvalues, scores and correlations
import pandas as pd                                                # monthly panels and rolling windows
import matplotlib.pyplot as plt                                    # the stacked time-series figure
import matplotlib.dates as mdates                                  # date locators and formatters
from matplotlib.ticker import FuncFormatter                        # readable multiples on log axes
from _common_l8 import DAT, FIG, ROOT, save, dump, style_axes, OKABE   # shared paths, style and palette

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK-capable font stack
plt.rcParams["axes.unicode_minus"] = False                         # true minus signs instead of tofu boxes
plt.rcParams["savefig.dpi"] = 200                                  # slide-ready resolution
plt.rcParams["savefig.bbox"] = "tight"                             # trim the surrounding margins

START = "1990-01"                                                  # readable modern window, same as l8_18
END = "2026-09"                                                    # last month present in the source files
ROLL = 36                                                          # rolling window length in months
BANDS = [("1999-01", "2000-12", "dot-com"),                        # the late-1990s technology bubble
         ("2007-12", "2009-06", "GFC"),                            # the global financial crisis
         ("2020-02", "2020-12", "COVID")]                          # the pandemic shock


def mult_fmt(v, _pos):                                             # format a growth multiple on a log axis
    """Return a compact multiplier such as 2x or 0.5x."""
    return "%gx" % v                                               # print the multiplier without padding


def log_ticks(ax, lo, hi):                                         # pick readable multiples for a log y axis
    """Set tick positions and labels for a growth-multiple axis."""
    cand = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]          # candidate growth multiples
    ticks = [t for t in cand if lo * 1.05 <= t <= hi * 1.35]       # keep labels inside the visible range
    ax.set_yticks(ticks)                                           # apply the tick positions
    ax.yaxis.set_major_formatter(FuncFormatter(mult_fmt))          # label every tick as a multiple


# ---------------------------------------------------------------- 1) the common monthly panel
ff3 = pd.read_csv(DAT / "ff3_factors_monthly.csv")                 # three factors, percent per month
ff25 = pd.read_csv(DAT / "ff25_portfolios_monthly.csv")            # 25 size and book-to-market portfolios
ff25 = ff25.drop(columns=[c for c in ("date", "RF") if c in ff25.columns])   # keep the month key and the 25 returns
panel = ff3.merge(ff25, on="yyyymm", how="inner")                  # join the two files on the month key
panel = panel[(panel["date"] >= START) & (panel["date"] <= END + "-31")].reset_index(drop=True)   # restrict the sample window
COLS = [c for c in ff25.columns if c != "yyyymm"]                  # the 25 portfolio column names
assert len(COLS) == 25                                             # the size and BM grid must be complete
X = panel[COLS].to_numpy(dtype=float)                              # 25 excess-return series in percent
T = X.shape[0]                                                     # number of monthly observations
dates = pd.to_datetime(panel["date"])                              # datetime index for plotting and rolling

# ---------------------------------------------------------------- 2) covariance PCA on the 25 portfolios
# Covariance PCA is used because all 25 columns are the same object in the same unit
# (percent excess return per month); standardising would over-weight low-variance corners.
Xc = X - X.mean(axis=0)                                            # centre every portfolio before the decomposition
cov = (Xc.T @ Xc) / (T - 1)                                        # sample covariance in percent squared
eigval, eigvec = np.linalg.eigh(cov)                               # symmetric eigendecomposition of the covariance
order = np.argsort(eigval)[::-1]                                   # sort components from largest to smallest
eigval = eigval[order]                                             # eigenvalues in descending order
eigvec = eigvec[:, order]                                          # eigenvectors as columns in the same order
flip = np.sign(eigvec[np.abs(eigvec).argmax(axis=0), np.arange(eigvec.shape[1])])   # sign of the largest loading
flip[flip == 0] = 1.0                                              # guard against an exact zero sign
eigvec = eigvec * flip                                             # orient every PC towards its dominant positive loading
var_expl = eigval / eigval.sum()                                   # share of total variance per component

# ---------------------------------------------------------------- 3) factor-mimicking returns and growth
# A raw PC score is an in-sample demeaned series, so compounding it would only show
# variance drag. The score is therefore turned into the return of the loading portfolio:
# the loading vector is rescaled to one dollar of gross exposure (sum of absolute weights
# is 1) and multiplied by the uncentred excess returns. The result is a percent-per-month
# factor-mimicking return, directly comparable with Mkt-RF, SMB and HML.
gross = np.abs(eigvec[:, :3]).sum(axis=0)                          # gross exposure of each loading vector
weights = eigvec[:, :3] / gross                                    # unit-gross factor-mimicking weights
pc_ret = X @ weights                                               # PC portfolio returns, percent per month
fac = panel[["Mkt-RF", "SMB", "HML"]].to_numpy(dtype=float)        # the three observable factors
cum_fac = np.cumprod(1.0 + fac / 100.0, axis=0)                    # growth of one unit, compounded monthly
cum_pc = np.cumprod(1.0 + pc_ret / 100.0, axis=0)                  # same units for the three PC factors
s_pc1 = pd.Series(pc_ret[:, 0], index=dates)                       # PC1 as a monthly return series
s_pc2 = pd.Series(pc_ret[:, 1], index=dates)                       # PC2 as a monthly return series
s_mkt = pd.Series(fac[:, 0], index=dates)                          # market excess return series
s_hml = pd.Series(fac[:, 2], index=dates)                          # value factor series
roll1 = s_pc1.rolling(ROLL).corr(s_mkt)                            # rolling PC1 versus market correlation
roll2 = s_pc2.rolling(ROLL).corr(s_hml)                            # rolling PC2 versus value correlation

# ---------------------------------------------------------------- 4) tidy CSV artefacts
fac_tab = pd.DataFrame({"yyyymm": panel["yyyymm"].astype(int),     # month key as in the source files
                        "date": panel["date"],                     # ISO first-of-month date
                        "mkt_rf": fac[:, 0],                       # market excess return, percent
                        "smb": fac[:, 1],                          # size factor, percent
                        "hml": fac[:, 2],                          # value factor, percent
                        "cum_mkt": cum_fac[:, 0],                  # growth of one unit in the market
                        "cum_smb": cum_fac[:, 1],                  # growth of one unit in SMB
                        "cum_hml": cum_fac[:, 2]})                 # growth of one unit in HML
dump(fac_tab.round(6), "l8_ff_ts_factors.csv")                     # persist the factor time series
pc_tab = pd.DataFrame({"yyyymm": panel["yyyymm"].astype(int),      # month key as in the source files
                       "date": panel["date"],                      # ISO first-of-month date
                       "pc1": pc_ret[:, 0],                        # PC1 mimicking return, percent per month
                       "pc2": pc_ret[:, 1],                        # PC2 mimicking return, percent per month
                       "pc3": pc_ret[:, 2],                        # PC3 mimicking return, percent per month
                       "cum_pc1": cum_pc[:, 0],                    # growth of one unit in PC1
                       "cum_pc2": cum_pc[:, 1],                    # growth of one unit in PC2
                       "cum_pc3": cum_pc[:, 2]})                   # growth of one unit in PC3
dump(pc_tab.round(6), "l8_ff_ts_pcs.csv")                          # persist the PC time series
roll_tab = pd.DataFrame({"date": panel["date"],                    # month of the rolling estimate
                         "roll_corr_pc1_mkt": roll1.to_numpy(),    # PC1 versus the market
                         "roll_corr_pc2_hml": roll2.to_numpy()})   # PC2 versus value
roll_tab = roll_tab.dropna().reset_index(drop=True)                # drop the first 35 incomplete windows
dump(roll_tab.round(6), "l8_ff_ts_rolling.csv")                    # persist the rolling correlations

# ---------------------------------------------------------------- 5) compact printed summary
print("window %s .. %s | %d months | 25 portfolios" % (panel["date"].iloc[0][:7], panel["date"].iloc[-1][:7], T))   # sample span
print("variance explained PC1..PC3 (%%):", np.round(var_expl[:3] * 100, 2))   # leading variance shares
print("growth multiple, observable: Mkt-RF %.3f | SMB %.3f | HML %.3f" % (cum_fac[-1, 0], cum_fac[-1, 1], cum_fac[-1, 2]))   # final multiples
print("growth multiple, PCA factors: PC1 %.3f | PC2 %.3f | PC3 %.3f" % (cum_pc[-1, 0], cum_pc[-1, 1], cum_pc[-1, 2]))   # final multiples
print("average rolling %dm correlation: PC1~Mkt-RF %.4f | PC2~HML %.4f" % (ROLL, float(roll1.mean()), float(roll2.mean())))   # mean co-movement
print("final sample date %s | rolling file starts %s" % (panel["date"].iloc[-1], roll_tab["date"].iloc[0]))   # sample end

# ---------------------------------------------------------------- 6) the three-panel time-series figure
fig = plt.figure(figsize=(12.4, 7.2))                              # slide-sized stacked figure
fig.patch.set_facecolor("white")                                   # white background for the slide
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.05, 0.85], hspace=0.16)   # three panels on one time axis
axA = fig.add_subplot(gs[0, 0])                                    # panel A: observable factors
axB = fig.add_subplot(gs[1, 0], sharex=axA)                        # panel B: statistical factors
axC = fig.add_subplot(gs[2, 0], sharex=axA)                        # panel C: rolling correlations
axA.plot(dates, cum_fac[:, 0], color=OKABE[4], lw=1.8, label="Mkt-RF (market)")   # market excess return
axA.plot(dates, cum_fac[:, 1], color=OKABE[2], lw=1.8, label="SMB (size)")   # small minus big
axA.plot(dates, cum_fac[:, 2], color=OKABE[5], lw=1.8, label="HML (value)")   # high minus low
axA.set_yscale("log")                                              # log scale keeps small and large multiples readable
log_ticks(axA, float(cum_fac.min()), float(cum_fac.max()))         # readable growth multiples
axA.legend(loc="upper left", frameon=False, ncol=1)                # compact legend for the three factors
style_axes(axA, "(a) Observable Fama-French factors: growth of 1 unit (log scale)", None, "growth of 1 unit")   # panel A labels
axB.plot(dates, cum_pc[:, 0], color=OKABE[4], lw=1.8, label="PC1 (%.1f%% of variance)" % (var_expl[0] * 100))   # market-like PC
axB.plot(dates, cum_pc[:, 1], color=OKABE[2], lw=1.8, label="PC2 (%.1f%% of variance)" % (var_expl[1] * 100))   # second PC
axB.plot(dates, cum_pc[:, 2], color=OKABE[5], lw=1.8, label="PC3 (%.1f%% of variance)" % (var_expl[2] * 100))   # third PC
axB.set_yscale("log")                                              # same units as panel A
log_ticks(axB, float(cum_pc.min()), float(cum_pc.max()))           # readable growth multiples
axB.legend(loc="upper left", frameon=False, ncol=1)                # legend with variance shares
style_axes(axB, "(b) Statistical factors (unit-gross PC portfolios): same units", None, "growth of 1 unit")   # panel B labels
axC.axhline(0.0, color="#66717D", lw=0.9, ls="--", alpha=0.8)      # zero line separates co-movement from reversal
axC.plot(dates, roll1, color=OKABE[4], lw=1.7, label="PC1 vs Mkt-RF")   # market-like statistical factor
axC.plot(dates, roll2, color=OKABE[5], lw=1.7, label="PC2 vs HML")   # value-like statistical factor
axC.set_ylim(-1.05, 1.05)                                          # correlations live in the unit interval
axC.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])                        # readable correlation ticks
axC.legend(loc="center right", frameon=False, ncol=1)               # legend sits between the two lines
style_axes(axC, "(c) Rolling 36-month correlation with the observable factors", "year", "correlation")   # panel C labels
axC.text(0.015, 0.06, "mean PC1~Mkt-RF %.2f" % float(roll1.mean()),   # in-panel note of the average correlation
         transform=axC.transAxes, ha="left", va="bottom", fontsize=8.4, color="#1E2630")   # place it in the lower left
for s, e, lab in BANDS:                                            # shade the three crisis windows
    for ax in (axA, axB, axC):                                     # the band spans all three panels
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#B9C2CC", alpha=0.28, lw=0, zorder=0)   # light band behind the lines
    mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2   # centre of the band
    axA.text(mid, 0.965, lab, transform=axA.get_xaxis_transform(),  # small label inside panel A
             ha="center", va="top", fontsize=7.4, color="#5A6572")   # grey label above the lines
axA.tick_params(labelbottom=False)                                 # hide x tick labels on panel A
axB.tick_params(labelbottom=False)                                 # hide x tick labels on panel B
axC.xaxis.set_major_locator(mdates.YearLocator(5))                 # one tick every five years
axC.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))          # show the calendar year only
axC.set_xlim(dates.iloc[0], dates.iloc[-1])                        # shared window ends at the last month
fig.suptitle("Observable and statistical factors on one monthly axis, 1990-2026", fontsize=13.0, y=0.988)   # figure title
fig.subplots_adjust(left=0.062, right=0.986, top=0.915, bottom=0.085)   # explicit margins around the three panels
save(fig, "fig_ff_ts.png")                                         # save into website/figures/lecture08

# ---------------------------------------------------------------- 7) mirror the figure into the LaTeX deck
deck_fig = ROOT.parent / "lecture08-2026-szu" / "figures"          # the deck's own figure folder
deck_fig.mkdir(parents=True, exist_ok=True)                        # create it when missing
shutil.copy2(FIG / "fig_ff_ts.png", deck_fig / "fig_ff_ts.png")    # byte-identical copy for the slides
print("[copy]", deck_fig / "fig_ff_ts.png")                        # log the mirrored path
print("done: factors and principal components on one time axis")   # closing log
