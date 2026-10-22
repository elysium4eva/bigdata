"""l8_03_factor_fama.py -- factor analysis on phone specs and the Fama-French three-factor model.

Every line carries an English comment. The script first extracts two factors from the
standardised phone specifications and reads the loadings, then estimates the classic
three-factor regression on a synthetic return panel so the deck can show betas, t
values and the explanatory power.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
import statsmodels.api as sm                                       # ordinary least squares with inference
from sklearn.decomposition import FactorAnalysis                   # maximum-likelihood factor analysis
from sklearn.preprocessing import StandardScaler                   # standardisation
from _common_l8 import (RNG, DAT, save, dump, style_axes,          # shared helpers
                        annotated, OKABE)                               # shared plotting helpers

# ---------------------------------------------------------------- 1) factor analysis on the phone sample
phones = pd.read_csv(DAT / "l8_smartphone.csv")                    # the sample from lesson 8.1
features = ["cpu", "gpu", "cores", "ram_gb", "camera_mp", "n_cameras", "price"]   # numeric features
X = StandardScaler().fit_transform(phones[features].values)         # standardise before factoring
fa = FactorAnalysis(n_components=2, random_state=42).fit(X)         # two latent factors
loadings = pd.DataFrame(fa.components_.T, index=features, columns=["F1", "F2"])   # loading matrix
dump(loadings.round(3).reset_index().rename(columns={"index": "feature"}), "l8_factor_loadings.csv")   # persist
print(loadings.round(3).to_string())                                # report the loadings
scores = fa.transform(X)                                           # factor scores per phone
print("factor score variance:", np.round(scores.var(axis=0), 3))    # report the score spread

# ---------------------------------------------------------------- 2) the Fama-French three-factor model
def make_ff3(n=180):                                               # a synthetic monthly panel
    """Return a synthetic panel of portfolio returns and the three factors."""   # short docstring
    mkt = RNG.normal(0.006, 0.045, n)                              # market excess return
    smb = RNG.normal(0.002, 0.028, n)                              # small minus big
    hml = RNG.normal(0.001, 0.030, n)                              # high minus low
    beta_m, beta_s, beta_h = 1.05, 0.62, -0.28                     # the true exposures
    ret = (0.0005 + beta_m * mkt + beta_s * smb + beta_h * hml      # the true model
           + RNG.normal(0, 0.012, n))                              # idiosyncratic noise
    return pd.DataFrame({"ret": ret, "mkt_excess": mkt, "smb": smb, "hml": hml})   # tabulate

panel = make_ff3(180)                                              # the synthetic panel
dump(panel.round(5), "l8_ff3.csv")                                 # persist it for the deck
Y = panel["ret"].values                                            # portfolio excess return
Xm = sm.add_constant(panel[["mkt_excess", "smb", "hml"]].values)    # design matrix with a constant
ols = sm.OLS(Y, Xm).fit()                                          # the three-factor regression
ci = ols.conf_int()                                                # 95 per cent intervals
table = pd.DataFrame({"term": ["alpha", "beta_mkt", "beta_smb", "beta_hml"],   # coefficient names
                      "coef": np.round(ols.params, 4),             # estimates
                      "std_err": np.round(ols.bse, 4),             # standard errors
                      "t_value": np.round(ols.tvalues, 3),         # t statistics
                      "p_value": np.round(ols.pvalues, 4),         # p values
                      "ci_low": np.round(ci[:, 0], 4),             # lower interval bound
                      "ci_high": np.round(ci[:, 1], 4)})           # upper interval bound
dump(table, "l8_ff3_regression.csv")                               # persist the table
print(table.to_string(index=False))                                # report the estimates
print("R^2 %.4f | adj R^2 %.4f | F %.2f" % (ols.rsquared, ols.rsquared_adj, ols.fvalue))   # fit statistics

# ---------------------------------------------------------------- 3) figure: loadings and the factor regression
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                # three panels
im = axes[0].imshow(loadings.values, cmap="RdPu", vmin=-1, vmax=1)  # loading heatmap
axes[0].set_xticks([0, 1], ["F1", "F2"], fontsize=8.4)             # factor labels
axes[0].set_yticks(range(len(features)), features, fontsize=7.6)    # feature labels
axes[0].grid(False)                                                # a heatmap needs no grid
for i in range(len(features)):                                     # annotate each cell
    for j in range(2):                                             # loop over the factors
        axes[0].text(j, i, "%.2f" % loadings.values[i, j], ha="center", va="center",   # the value
                     fontsize=7.4, color="white" if abs(loadings.values[i, j]) > 0.55 else "#1E2630")   # contrast
axes[0].set_title("Factor loadings")                               # panel title
plt.colorbar(im, ax=axes[0], fraction=0.046)                       # colour bar
axes[1].scatter(scores[:, 0], scores[:, 1], c=phones["price"], cmap="RdPu", s=24)   # factor scores
style_axes(axes[1], "Phones in factor space", "factor 1", "factor 2")   # shared style
coefs = table["coef"].values[1:]                                   # the three betas
labels = ["market", "SMB", "HML"]                                  # factor names
axes[2].bar(labels, coefs, color=[OKABE[4], OKABE[2], OKABE[5]], alpha=0.9)   # beta bars
axes[2].axhline(0, color="#66717D", lw=0.9)                        # zero reference
for i, (b, t) in enumerate(zip(coefs, table["t_value"].values[1:])):   # annotate each beta
    axes[2].text(i, b, "%.2f\n(t=%.1f)" % (b, t), ha="center",     # value and t statistic
                 va="bottom" if b >= 0 else "top", fontsize=7.4)   # place the label on the right side
style_axes(axes[2], "Three-factor exposures ($R^2$ %.2f)" % ols.rsquared, "", "beta")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_factor_loadings.png")                               # save the loadings figure

# ---------------------------------------------------------------- 4) figure: fitted versus actual and residuals
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))                # two panels
axes[0].scatter(ols.fittedvalues, Y, s=16, color=OKABE[1], alpha=0.75)   # fitted versus actual
lims = [min(Y.min(), ols.fittedvalues.min()), max(Y.max(), ols.fittedvalues.max())]   # common limits
axes[0].plot(lims, lims, ls="--", lw=1.0, color="#66717D")          # perfect fit line
style_axes(axes[0], "Fitted versus actual returns", "fitted", "actual")   # shared style
axes[1].scatter(ols.fittedvalues, ols.resid, s=16, color=OKABE[5], alpha=0.75)   # residual plot
axes[1].axhline(0, color="#66717D", lw=0.9)                        # zero reference
style_axes(axes[1], "Residuals show no trend", "fitted", "residual")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_ff3.png")                                           # save the regression figure
print("done: factor loadings and the Fama-French regression")       # closing log
