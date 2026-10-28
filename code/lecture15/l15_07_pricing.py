"""l15_07_pricing.py -- own-price elasticity, optimal markup and promotion lift.

Every line carries an English comment. The script estimates a log-log demand curve from weekly
price and sales, recovers the elasticity, solves the profit-maximising price, and separates
promotion lift from the baseline so the deck can discuss discount depth and profitability.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # OLS formulas
from _common_l15 import (make_price_sales, save, dump,             # shared helpers
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the weekly data
sales = make_price_sales(104)                                      # two years of weekly data
sales["log_price"] = np.log(sales["price"])                         # log price
sales["log_units"] = np.log(sales["units"])                         # log units
dump(sales, "l15_price_sales.csv")                                 # persist
print("weeks:", len(sales), "| price range: %.1f-%.1f" % (sales["price"].min(), sales["price"].max()))   # report

# ---------------------------------------------------------------- 2) log-log elasticity
model = smf.ols("log_units ~ log_price + promo", data=sales).fit(cov_type="HC1")   # robust standard errors
elasticity = float(model.params["log_price"])                       # the own-price elasticity
print("elasticity: %.3f (se %.3f) | promo effect: %.3f" % (             # report the estimates
    elasticity, float(model.bse["log_price"]), float(model.params["promo"])))   # report
print("R2: %.3f" % model.rsquared)                                  # report the fit
coef_table = pd.DataFrame({"term": model.params.index,               # names
                           "coef": model.params.values.round(4),      # estimates
                           "se": model.bse.values.round(4)})          # standard errors
dump(coef_table, "l15_elasticity.csv")                              # persist

# ---------------------------------------------------------------- 3) average cost and the optimal price
unit_cost = 17.0                                                    # a constant marginal cost
intercept = float(model.params["Intercept"])                        # demand intercept in logs
promo_mean = float(sales["promo"].mean())                           # average promotion share
def demand(price):                                                 # expected units at a price
    """Return expected weekly units at a given price."""             # short docstring
    return float(np.exp(intercept + elasticity * np.log(price) + 0.35 * promo_mean))   # log-linear demand

def profit(price):                                                 # profit at a price
    """Return weekly profit at a given price."""                    # short docstring
    return (price - unit_cost) * demand(price)                      # margin times units

grid = np.linspace(20, 70, 400)                                     # a price grid
profits = np.array([profit(p) for p in grid])                       # profit at every price
best = float(grid[np.argmax(profits)])                              # the profit-maximising price
markup_rule = unit_cost * elasticity / (1 + elasticity)             # the Lerner price for log-linear demand
print("optimal price: %.2f | Lerner benchmark: %.2f | profit at current price: %.0f" % (   # report the optimal prices
    best, markup_rule, profit(39.0)))                               # report
print("profit at optimal price: %.0f" % profits.max())              # report the gain

# ---------------------------------------------------------------- 4) promotion decomposition
sales["baseline"] = np.exp(intercept + elasticity * np.log(sales["price"]))   # no-promotion baseline
sales["increment"] = sales["units"] - sales["baseline"]             # incremental units
promo_weeks = sales[sales["promo"] == 1]                            # promotion weeks only
non_promo = sales[sales["promo"] == 0]                              # regular weeks
print("promo weeks: %d | mean lift: %.1f units (%.0f%%)" % (            # report the promotion lift
    len(promo_weeks), promo_weeks["increment"].mean(),                  # the promotion weeks
    100 * promo_weeks["increment"].mean() / non_promo["units"].mean()))   # report
dump(sales[["week", "price", "promo", "units", "baseline", "increment"]].round(2),   # persist the decomposition
     "l15_promotion_decomposition.csv")                              # persist

# ---------------------------------------------------------------- 5) the elasticity figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                 # three panels
axes[0].scatter(sales["price"], sales["units"], s=16, alpha=0.65,   # the observed weeks
                color=np.where(sales["promo"] == 1, OKABE[5], OKABE[1]))   # colour promotion weeks
px = np.linspace(sales["price"].min(), sales["price"].max(), 100)    # a price grid for the curve
axes[0].plot(px, np.exp(intercept + elasticity * np.log(px)), lw=1.8, color="#2B2B2B")   # fitted demand
style_axes(axes[0], "Demand curve (promotion weeks in red)", "价格 (元)", "周销量")   # shared style
axes[1].plot(grid, profits / 1e3, lw=2.0, color=OKABE[2])            # the profit curve
axes[1].axvline(best, ls="--", lw=1.2, color=OKABE[5])               # the optimum
axes[1].axvline(39.0, ls=":", lw=1.2, color="#66717D")               # the current price
style_axes(axes[1], "Weekly profit versus price", "价格 (元)", "利润 (千元)")   # shared style
annotated(axes[1], "最优 %.0f 元" % best, "lower right")              # annotate
axes[2].bar(sales["week"][-26:], sales["baseline"][-26:], color=OKABE[3], alpha=0.9, label="基线")   # baseline
axes[2].bar(sales["week"][-26:], sales["increment"][-26:], bottom=sales["baseline"][-26:],   # the stacked increment
            color=OKABE[5], alpha=0.9, label="增量")                 # increment stacked on top
style_axes(axes[2], "Baseline plus promotion increment (last 26 weeks)", "week", "units")   # shared style
axes[2].legend(fontsize=7.4)                                        # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_price_elasticity.png")                              # save the figure
print("done: elasticity, optimal pricing and promotion decomposition")   # closing log
