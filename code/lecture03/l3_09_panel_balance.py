"""l3_09_panel_balance.py -- balanced and unbalanced panels, gaps, and interpolation.

The lecture's panel examples are rebuilt as data: a balanced panel with n=2 and
T=11, a nominal panel where different firms have different T_i, and a panel with
block missingness. The script diagnoses the structure, fills or interpolates the
gaps and marks every imputed cell so the repair stays visible.
"""
import numpy as np                                                 # random walks for the panel values
import pandas as pd                                                # panel reshaping and interpolation
import matplotlib.pyplot as plt                                    # heat maps of the panel
from _common_l3 import RNG, save, dump                             # shared seed, figure and data writers

MONTHS = list(range(1, 12))                                        # T = 11 periods, as in the lecture

# ---------------------------------------------------------------- 1) balanced panel: n=2, T=11
def make_series(n, start):                                         # helper: one firm level time series
    """Random walk with a firm-specific level and trend, one value per month."""   # helper contract
    level = start + np.cumsum(RNG.normal(0.6, 1.1, n))             # accumulate random monthly changes
    return np.round(level, 2)                                      # rounded for readability
balanced = pd.DataFrame({"firm_id": np.repeat(["F1", "F2"], len(MONTHS)),   # two firms
                         "month": MONTHS * 2,                      # the same eleven months for both
                         "revenue": np.concatenate([make_series(len(MONTHS), 100),   # firm F1 series
                                                    make_series(len(MONTHS), 130)])})   # firm F2 series
dump(balanced, "l3_panel_balanced.csv")                            # publish the balanced panel
wide_balanced = balanced.pivot(index="firm_id", columns="month", values="revenue")   # firm x month matrix
print("balanced panel shape (n x T):", wide_balanced.shape)        # n=2, T=11 as in the lecture

# ---------------------------------------------------------------- 2) unbalanced panel: T_i differs
spec = {"F1": 9, "F2": 11, "F3": 8}                                # T1=9, T2=11, T3=8, n=3
frames = []                                                        # collect one frame per firm
for i, (firm, periods) in enumerate(spec.items()):                 # build each firm's shorter series
    frames.append(pd.DataFrame({"firm_id": firm,                   # firm identifier
                                "month": list(range(1, periods + 1)),   # this firm's own horizon
                                "revenue": make_series(periods, 90 + 25 * i)}))   # value series
unbalanced = pd.concat(frames, ignore_index=True)                  # the unbalanced panel
unbalanced.loc[(unbalanced["firm_id"] == "F2") &                # plant a block of missing months
               (unbalanced["month"].isin([4, 5])), "revenue"] = np.nan   # F2 loses months 4 and 5
dump(unbalanced, "l3_panel_unbalanced.csv")                        # publish the unbalanced panel
wide_unbalanced = unbalanced.pivot(index="firm_id", columns="month", values="revenue")   # with NaN gaps
coverage = (unbalanced.assign(observed=unbalanced["revenue"].notna())   # per firm coverage
            .groupby("firm_id")["observed"].agg(periods="count", observed="sum"))   # counts per firm
coverage["T_i"] = [spec[f] for f in coverage.index]                # declared horizon per firm
coverage["coverage_pct"] = (100 * coverage["observed"] / coverage["periods"]).round(1)   # coverage rate
coverage["balanced"] = coverage["periods"] == coverage["periods"].max()   # balanced means the same T
print("panel coverage table:\n", coverage.to_string())              # structure diagnosis for the slide
dump(coverage.reset_index(), "l3_panel_coverage.csv")               # publish the coverage table

# ---------------------------------------------------------------- 3) repair the gaps and mark them
repaired = unbalanced.sort_values(["firm_id", "month"]).copy()      # sort so that interpolation is ordered
repaired["revenue_interpolated"] = (repaired.groupby("firm_id")["revenue"]   # within each firm separately
                                    .transform(lambda s: s.interpolate(limit_direction="both")))   # linear fill
repaired["filled_flag"] = repaired["revenue"].isna() & repaired["revenue_interpolated"].notna()   # mark fills
repaired["revenue"] = repaired["revenue_interpolated"]             # promote the filled values
dump(repaired, "l3_panel_repaired.csv")                            # publish the repaired panel
wide_repaired = repaired.pivot(index="firm_id", columns="month", values="revenue")   # matrix after repair
still_missing = int(wide_repaired.isna().sum().sum())              # cells that interpolation could not fill
print("cells still missing after interpolation:", still_missing)   # interpolation cannot create new periods
print("imputed cells:", int(repaired["filled_flag"].sum()))        # how many cells were manufactured

# ---------------------------------------------------------------- figure A: the balanced panel
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))                 # lines plus heat map
for firm in balanced["firm_id"].unique():                          # one line per firm
    sub = balanced[balanced["firm_id"] == firm]                    # take the firm's rows
    axes[0].plot(sub["month"], sub["revenue"], "-o", ms=3.5, lw=1.4, label=firm)   # monthly path
axes[0].set_title("balanced panel: n = 2, T = 11")                  # panel title
axes[0].set_xlabel("month"); axes[0].set_ylabel("revenue")          # axis labels
axes[0].legend(title="firm")                                        # firm legend
im = axes[1].imshow(wide_balanced.values, cmap="RdPu", aspect="auto")   # firm x month heat map
axes[1].set_xticks(range(len(MONTHS)), MONTHS, fontsize=8)          # month ticks
axes[1].set_yticks(range(wide_balanced.shape[0]), wide_balanced.index, fontsize=10)   # firm ticks
axes[1].set_title("heat map: every cell is observed")               # panel title
axes[1].grid(False)                                                 # heat maps do not need a grid
plt.colorbar(im, ax=axes[1], fraction=0.046)                        # value scale
plt.tight_layout()                                                  # balance the panels
save(fig, "fig_panel_balanced.png")                                 # figure for the balanced-panel slide

# ---------------------------------------------------------------- figure B: before and after repair
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # two heat maps plus coverage
for ax, matrix, title in [(axes[0], wide_unbalanced, "before: gaps and shorter T_i"),   # raw structure
                          (axes[1], wide_repaired, "after: interpolated (marked cells)")]:   # repaired
    ax.imshow(matrix.values, cmap="RdPu", aspect="auto")            # heat map of the matrix
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, fontsize=8)   # month ticks
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=10)    # firm ticks
    ax.set_title(title, fontsize=10.5)                              # panel title
    ax.grid(False)                                                  # heat maps do not need a grid
for _, row in repaired[repaired["filled_flag"]].iterrows():         # mark every interpolated cell
    axes[1].scatter(row["month"] - 1, list(wide_repaired.index).index(row["firm_id"]),   # cell position
                    marker="o", s=90, facecolors="none", edgecolors="#1E2630", lw=1.2)   # hollow marker
axes[2].bar(coverage.index, coverage["coverage_pct"], color="#8A0C3C", alpha=0.85)   # coverage per firm
axes[2].axhline(100, color="#C9A227", ls="--", lw=1.2, label="full coverage")   # 100% reference line
axes[2].set_ylim(0, 115)                                            # leave room for the reference line
axes[2].set_title("coverage per firm (%)")                          # panel title
axes[2].legend(fontsize=8.5)                                        # legend for the reference line
plt.tight_layout()                                                  # balance the panels
save(fig, "fig_panel_heatmap.png")                                  # figure for the panel-repair slide
