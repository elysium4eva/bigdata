"""l15_04_hedonic.py -- hedonic price regression and the marketing response curve.

Every line carries an English comment. The script estimates a log-linear hedonic price model on
1,200 housing transactions, reads the semi-elasticities, and then fits advertising/promotion
response curves with adstock so the deck can compare "attribute value" with "spend response".
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # OLS formulas
from _common_l15 import (RNG, make_housing, save, dump, style_axes,     # shared helpers
                         annotated, OKABE)                              # shared plotting helpers

# ---------------------------------------------------------------- 1) the housing data
housing = make_housing(1200)                                       # 1,200 transactions
housing["log_price"] = np.log(housing["price"])                     # work in logs
dump(housing, "l15_housing.csv")                                   # persist
print("transactions:", len(housing), "| median price: %.0f 元/㎡" % housing["price"].median())   # report

# ---------------------------------------------------------------- 2) the hedonic regression
model = smf.ols("log_price ~ area + school + metro_km + age + floor", data=housing).fit(cov_type="HC1")   # OLS with robust SEs
print(model.summary().tables[1])                                    # report the coefficient table
coefs = pd.DataFrame({"term": model.params.index,                   # coefficient names
                      "coef": model.params.values.round(4),         # point estimates
                      "se": model.bse.values.round(4),              # robust standard errors
                      "p": model.pvalues.values.round(4)})          # p values
dump(coefs, "l15_hedonic_coefs.csv")                                # persist
semi = (np.exp(model.params) - 1) * 100                             # approximate semi-elasticities in percent
print("semi-elasticities (%):", semi.round(2).to_dict())             # report
print("R2: %.3f" % model.rsquared)                                  # report the fit

# ---------------------------------------------------------------- 3) a quantile check
quant = smf.quantreg("log_price ~ area + school + metro_km + age + floor", housing)   # quantile regression
q_rows = []                                                        # coefficients at three quantiles
for q in (0.1, 0.5, 0.9):                                          # three points of the distribution
    fit = quant.fit(q=q)                                           # fit at that quantile
    q_rows.append({"quantile": q, "area": float(fit.params["area"]),   # area effect
                   "school": float(fit.params["school"]),           # school effect
                   "metro_km": float(fit.params["metro_km"])})      # metro effect
qframe = pd.DataFrame(q_rows)                                       # assemble the table
dump(qframe.round(4), "l15_hedonic_quantiles.csv")                   # persist
print(qframe.round(4).to_string(index=False))                        # report

# ---------------------------------------------------------------- 4) the response curve with adstock
weeks = np.arange(1, 53)                                           # fifty-two weeks
spend = np.where(weeks % 6 < 3, 120.0, 40.0)                        # a pulsing media plan (万元)
adstock = np.zeros_like(spend)                                      # the carryover series
decay = 0.55                                                       # weekly carryover rate
for t in range(len(spend)):                                        # geometric adstock recursion
    adstock[t] = spend[t] + (decay * adstock[t - 1] if t > 0 else 0.0)   # current spend plus decayed past
half_sat = 260.0                                                   # the half-saturation point
hill = adstock / (adstock + half_sat)                               # a concave saturation curve
base_sales = 520.0                                                 # baseline sales
sales = base_sales + 380.0 * hill + RNG.normal(0, 18.0, len(weeks))   # sales with the seeded noise
curve = pd.DataFrame({"week": weeks, "spend": spend, "adstock": adstock.round(2),   # the series
                      "hill": hill.round(4), "sales": sales.round(1)})   # more columns
dump(curve, "l15_response_curve.csv")                              # persist
print("peak adstock %.0f | max saturation %.2f" % (adstock.max(), hill.max()))   # report

# ---------------------------------------------------------------- 5) the hedonic figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                 # three panels
terms = ["area", "school", "metro_km", "age", "floor"]              # the five attributes
vals = [float(model.params[t]) for t in terms]                      # their coefficients
errs = [1.96 * float(model.bse[t]) for t in terms]                  # 95% intervals
colours = [OKABE[2] if v > 0 else OKABE[5] for v in vals]           # colour by sign
axes[0].barh(terms[::-1], np.array(vals)[::-1], xerr=np.array(errs)[::-1],   # the bars with intervals
             color=colours[::-1], alpha=0.9)                         # colours
axes[0].axvline(0, color="#2B2B2B", lw=0.9)                         # zero line
style_axes(axes[0], "Hedonic coefficients (log price)", "coefficient", "")   # shared style
axes[1].bar(["10%", "50%", "90%"], qframe["school"], color=OKABE[4], alpha=0.9)   # quantile school effect
style_axes(axes[1], "School premium by quantile of price", "", "coefficient")   # shared style
for i, v in enumerate(qframe["school"]):                            # annotate
    axes[1].text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=8.4)   # value
axes[2].scatter(housing["area"], housing["price"] / 1e4, s=8, alpha=0.35, color=OKABE[1])   # raw scatter
grid = np.linspace(housing["area"].min(), housing["area"].max(), 60)   # a grid for the fitted line
pred = np.exp(model.params["Intercept"] + model.params["area"] * grid   # hold other terms at their means
              + model.params["school"] * housing["school"].mean()       # school at its mean
              + model.params["metro_km"] * housing["metro_km"].mean()   # metro at its mean
              + model.params["age"] * housing["age"].mean()             # age at its mean
              + model.params["floor"] * housing["floor"].mean()) / 1e4   # convert to 万元
axes[2].plot(grid, pred, lw=2.0, color=OKABE[5])                     # the fitted curve
style_axes(axes[2], "Price versus area with the fitted curve", "面积 (㎡)", "总价 (万元)")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_hedonic_coefs.png")                                 # save the hedonic figure

# ---------------------------------------------------------------- 6) the response figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                 # three panels
axes[0].bar(weeks, spend, color=OKABE[3], alpha=0.9, label="每周投放")   # the media plan
axes[0].plot(weeks, adstock, lw=2.0, color=OKABE[5], label="adstock")   # the carryover series
style_axes(axes[0], "Media plan and its adstock", "week", "万元")      # shared style
axes[0].legend(fontsize=7.4)                                        # compact legend
axes[1].plot(adstock, hill, "-o", ms=3.5, color=OKABE[4])            # the saturation curve
axes[1].axvline(half_sat, ls="--", lw=1.0, color="#66717D")          # the half-saturation point
style_axes(axes[1], "Saturation: marginal effect falls", "adstock", "响应比例")   # shared style
annotated(axes[1], "半饱和点 %.0f" % half_sat, "lower right")         # annotate
axes[2].plot(spend, sales - base_sales, "-o", ms=3.5, color=OKABE[2])   # incremental sales
axes[2].axhline(0, color="#2B2B2B", lw=0.9)                          # zero increment line
style_axes(axes[2], "Incremental sales versus spend", "投放 (万元)", "增量销量")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_response_curve.png")                                # save the response figure
print("done: hedonic regression, quantiles and the response curve")   # closing log
