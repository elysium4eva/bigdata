"""l6_13_ridge_business.py -- a business case for regularised multiple linear regression.

Every line carries an English comment. The script builds a store-level sales panel whose drivers are
strongly collinear (advertising, traffic, promotion, seasonality), then compares ordinary least
squares with Ridge and Lasso: coefficient instability under resampling, coefficient paths, and the
out-of-sample error curve. The numbers feed two frames in the regularisation section.
"""
import numpy as np                                                 # arrays
import matplotlib                                                  # backend before pyplot
matplotlib.use("Agg")                                              # headless backend
import matplotlib.pyplot as plt                                    # figures
import pandas as pd                                                # tables
from pathlib import Path                                           # paths
from sklearn.linear_model import LinearRegression, Ridge, LassoCV  # the three estimators
from sklearn.model_selection import train_test_split, cross_val_score   # splits and CV
from sklearn.preprocessing import StandardScaler                   # standardisation
from sklearn.pipeline import make_pipeline                         # glue scaler and model

RNG = np.random.default_rng(20261106)                              # a fixed seed
ROOT = Path(__file__).resolve().parents[2]                         # website/
FIG = ROOT / "figures" / "lecture06"                               # figure folder
DAT = ROOT / "data" / "lecture06"                                  # data folder
for d in (FIG, DAT):                                               # ensure both exist
    d.mkdir(parents=True, exist_ok=True)                           # create when missing

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK stack
plt.rcParams["axes.unicode_minus"] = False                         # real minus signs
plt.rcParams["savefig.dpi"] = 200                                  # slide resolution
plt.rcParams["savefig.bbox"] = "tight"                             # trim white space
plt.rcParams["axes.grid"] = True                                   # light grid
plt.rcParams["grid.alpha"] = 0.25                                  # subtle grid
plt.rcParams["axes.spines.top"] = False                            # drop the top spine
plt.rcParams["axes.spines.right"] = False                          # drop the right spine
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# ---------------------------------------------------------------- 1) a collinear business panel
n = 500                                                            # 500 store-months
month = np.tile(np.arange(1, 13), 42)[:n]                          # calendar month
traffic = RNG.normal(12000, 2500, n)                               # store traffic
base = traffic / 1000                                              # a common driver of every media column
ad_tv = 0.75 * base + RNG.normal(0, 0.7, n)                        # TV spend, strongly tied to traffic
ad_social = 0.78 * base + RNG.normal(0, 0.7, n)                    # social spend, strongly tied
ad_search = 0.80 * base + RNG.normal(0, 0.7, n)                    # search spend, strongly tied
ad_ooh = 0.74 * base + RNG.normal(0, 0.8, n)                       # out-of-home, also tied
promo = (RNG.random(n) < 0.3).astype(float)                        # promotion weeks
season = 1 + 0.18 * np.sin((month - 3) / 12 * 2 * np.pi)            # seasonal multiplier
# seven further columns that only look related: they carry traffic but no real effect
decoys = {"ad_radio": 0.72 * base + RNG.normal(0, 0.9, n),
          "ad_print": 0.70 * base + RNG.normal(0, 0.9, n),
          "ad_coupon": 0.76 * base + RNG.normal(0, 0.9, n),
          "ad_email": 0.71 * base + RNG.normal(0, 0.9, n),
          "ad_affiliate": 0.73 * base + RNG.normal(0, 0.9, n),
          "ad_influencer": 0.69 * base + RNG.normal(0, 0.9, n),
          "ad_marketplace": 0.77 * base + RNG.normal(0, 0.9, n)}
sales = (8.0 + 0.9 * ad_tv + 1.6 * ad_social + 2.1 * ad_search      # the true drivers only
         + 0.00042 * traffic - 1.8 * promo + 12 * (season - 1)
         + RNG.normal(0, 3.4, n)) * season                          # noise plus season
cols_data = {"month": month, "traffic": traffic.round(0),
             "ad_tv": ad_tv.round(2), "ad_social": ad_social.round(2),
             "ad_search": ad_search.round(2), "ad_ooh": ad_ooh.round(2), "promo": promo}
for k, v in decoys.items():                                        # add the decoy columns
    cols_data[k] = v.round(2)                                      # rounded for readability
cols_data["sales"] = sales.round(2)                                # the target
frame = pd.DataFrame(cols_data)                                    # the panel
frame.to_csv(DAT / "l6_store_sales.csv", index=False, encoding="utf-8-sig")   # persist
cols = ["traffic", "ad_tv", "ad_social", "ad_search", "ad_ooh", "promo",
        "ad_radio", "ad_print", "ad_coupon", "ad_email", "ad_affiliate",
        "ad_influencer", "ad_marketplace"]                          # thirteen drivers, four of them real
X = frame[cols].to_numpy()                                          # design matrix
y = frame["sales"].to_numpy()                                       # target
print("rows:", n, "| corr(traffic, ad_tv) %.3f | corr(ad_tv, ad_social) %.3f" % (
    np.corrcoef(X[:, 0], X[:, 1])[0, 1], np.corrcoef(X[:, 1], X[:, 2])[0, 1]))   # report collinearity

# ---------------------------------------------------------------- 2) OLS vs Ridge vs Lasso
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)   # a hold-out split
ols = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)   # ordinary least squares
ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)  # ridge with a fixed strength
lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42)).fit(Xtr, ytr)   # lasso with CV
rows = []                                                          # one row per model
for name, model in (("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)):   # loop over the models
    pred = model.predict(Xte)                                      # the test predictions
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))               # test RMSE
    coefs = model[-1].coef_                                        # standardised coefficients
    rows.append({"model": name, "rmse": round(rmse, 3),            # the record
                 "max_abs_coef": round(float(np.max(np.abs(coefs))), 3),
                 "n_nonzero": int(np.sum(np.abs(coefs) > 1e-8))})    # sparsity
scores = pd.DataFrame(rows)                                        # assemble
scores.to_csv(DAT / "l6_regularisation_scores.csv", index=False, encoding="utf-8-sig")   # persist
print(scores.to_string(index=False))                                # report
print("lasso alpha chosen: %.3f" % float(lasso[-1].alpha_))          # report

# ---------------------------------------------------------------- 3) coefficient instability
reps = 40                                                          # resampling repetitions
coef_ols, coef_ridge = [], []                                      # the coefficient draws
for _ in range(reps):                                              # loop over the repetitions
    idx = RNG.integers(0, len(Xtr), len(Xtr))                       # bootstrap the training rows
    m1 = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr[idx], ytr[idx])   # OLS on the resample
    m2 = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr[idx], ytr[idx])    # Ridge on the resample
    coef_ols.append(m1[-1].coef_)                                  # record the OLS coefficients
    coef_ridge.append(m2[-1].coef_)                                # record the Ridge coefficients
coef_ols = np.array(coef_ols)                                      # to array
coef_ridge = np.array(coef_ridge)                                  # to array
spread_ols = float(np.mean(coef_ols.std(axis=0)))                  # average standard deviation
spread_ridge = float(np.mean(coef_ridge.std(axis=0)))              # average standard deviation
print("coefficient spread: OLS %.3f | Ridge %.3f" % (spread_ols, spread_ridge))   # report

# ---------------------------------------------------------------- 4) coefficient paths
alphas = np.logspace(-2, 4, 60)                                    # strengths from weak to strong
path_ridge, path_lasso = [], []                                    # the coefficient paths
for a in alphas:                                                   # loop over the strengths
    mr = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xtr, ytr)    # ridge at this strength
    ml = make_pipeline(StandardScaler(), LassoCV(cv=3, random_state=42)).fit(Xtr, ytr)   # lasso (CV per fit)
    path_ridge.append(mr[-1].coef_)                                # record ridge coefficients
    path_lasso.append(ml[-1].coef_)                                # record lasso coefficients
path_ridge = np.array(path_ridge)                                  # to array
path_lasso = np.array(path_lasso)                                  # to array
paths = pd.DataFrame({"alpha": alphas.round(4)})                   # the path table
for j, c in enumerate(cols):                                       # one column per driver
    paths["ridge_" + c] = path_ridge[:, j].round(4)                 # ridge path
    paths["lasso_" + c] = path_lasso[:, j].round(4)                 # lasso path
paths.to_csv(DAT / "l6_regularisation_paths.csv", index=False, encoding="utf-8-sig")   # persist

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))                 # three panels
for j, c in enumerate(cols):                                       # loop over the drivers
    axes[0].plot(alphas, path_ridge[:, j], lw=1.6, label=c)         # ridge paths
axes[0].set_xscale("log")                                          # log x
axes[0].legend(fontsize=7.0)                                       # compact legend
axes[0].set_xlabel("$\\lambda$（对数）"); axes[0].set_ylabel("标准化系数")   # labels
axes[0].set_title("Ridge：系数整体收缩，谁都不归零")                   # the panel title
for j, c in enumerate(cols):                                       # loop over the drivers
    axes[1].plot(alphas, path_lasso[:, j], lw=1.6, label=c)         # lasso paths
axes[1].set_xscale("log")                                          # log x
axes[1].set_xlabel("$\\lambda$（对数）"); axes[1].set_ylabel("标准化系数")   # labels
axes[1].set_title("Lasso：不重要的系数逐个归零")                      # the panel title
labels = ["traffic", "ad_tv", "ad_social", "ad_search", "promo"]    # the five main drivers
pos = np.arange(len(labels))                                       # positions
idx5 = [cols.index(c) for c in labels]               # positions of the five main drivers
axes[2].boxplot([coef_ols[:, j] for j in idx5], positions=pos - 0.17,
                widths=0.3, patch_artist=True,
                boxprops=dict(facecolor=OKABE[5], alpha=0.75), showfliers=False)   # OLS spread
axes[2].boxplot([coef_ridge[:, j] for j in idx5], positions=pos + 0.17,
                widths=0.3, patch_artist=True,
                boxprops=dict(facecolor=OKABE[2], alpha=0.75), showfliers=False)   # ridge spread
axes[2].axhline(0, color="#2B2B2B", lw=0.8)                         # zero line
axes[2].set_xticks(pos, labels, rotation=18, ha="right")            # labels
axes[2].set_title("重采样 40 次：OLS（红）比 Ridge（绿）抖得多")          # the panel title
plt.tight_layout()                                                 # tidy the layout
fig.savefig(FIG / "fig_ridge_business.png", facecolor="white")      # save
plt.close(fig)                                                     # release memory
print("[figure]", FIG / "fig_ridge_business.png")                   # log it
summary = pd.DataFrame([{"metric": "OLS 系数平均标准差", "value": round(spread_ols, 3)},
                        {"metric": "Ridge 系数平均标准差", "value": round(spread_ridge, 3)},
                        {"metric": "Lasso alpha", "value": round(float(lasso[-1].alpha_), 3)}])   # the summary
summary.to_csv(DAT / "l6_regularisation_summary.csv", index=False, encoding="utf-8-sig")   # persist
print(summary.to_string(index=False))                               # report
print("done: business case for regularised regression")             # closing log
