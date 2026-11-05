"""l6_14_discrete_choice.py -- logit demand, elasticities and a merger simulation.

Every line carries an English comment. The script builds a synthetic four-brand market panel, shows
why regressing log shares on price is biased, recovers the price coefficient with a cost shifter
through two-stage least squares, then turns the estimates into an elasticity matrix, a diversion
ratio and a small merger simulation. The numbers feed the discrete-choice section of the deck.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib                                                  # backend before pyplot
matplotlib.use("Agg")                                              # headless backend
import matplotlib.pyplot as plt                                    # figures
import statsmodels.api as sm                                       # OLS and 2SLS
from pathlib import Path                                           # paths

RNG = np.random.default_rng(20261113)                              # a fixed seed
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

# ---------------------------------------------------------------- 1) a four-brand market
BRANDS = ["A 可乐", "B 可乐", "C 气泡水", "D 气泡水"]                  # four inside goods
T = 300                                                            # market-week observations per brand
alpha_true = -2.0                                                  # the true price coefficient
beta_true = 1.2                                                    # the true feature coefficient
intercepts = np.array([1.6, 0.9, 0.2, -0.4])                       # brand fixed utilities
rows = []                                                          # the market-level rows
for t in range(T):                                                 # loop over markets
    cost = RNG.normal(0, 1, 4)                                      # marginal cost shifters (instruments)
    xi = RNG.normal(0, 0.6, 4)                                      # unobserved quality
    feature = RNG.normal(0, 1, 4) + np.array([0.4, -0.2, 0.6, -0.5])   # a brand-level feature (e.g. sugar)
    price = 1.6 + 0.22 * cost + 0.12 * xi + RNG.normal(0, 0.08, 4)   # price depends on cost AND quality
    u = intercepts + alpha_true * price + beta_true * feature + xi   # the mean utility delta
    share_out = 1.0 / (1.0 + np.exp(u).sum())                        # the outside good share
    shares = np.exp(u) / (1.0 + np.exp(u).sum())                     # the inside shares
    for j in range(4):                                              # loop over the brands
        rows.append({"market": t, "brand": BRANDS[j], "price": round(float(price[j]), 3),
                     "feature": round(float(feature[j]), 3), "cost": round(float(cost[j]), 3),
                     "share": round(float(shares[j]), 5), "share_out": round(float(share_out), 5)})
market = pd.DataFrame(rows)                                        # the panel
market.to_csv(DAT / "l6_market_shares.csv", index=False, encoding="utf-8-sig")   # persist
print("markets:", T, "brands:", len(BRANDS), "| mean outside share %.3f" % market["share_out"].mean())

# ---------------------------------------------------------------- 2) the logit transformation
market["log_ratio"] = np.log(market["share"]) - np.log(market["share_out"])   # log(s_j) - log(s_0)
X = pd.get_dummies(market["brand"], prefix="b", drop_first=True).astype(float)   # brand dummies
X["price"] = market["price"]                                        # the endogenous regressor
X["feature"] = market["feature"]                                    # the exogenous one
Xc = sm.add_constant(X)                                             # add the intercept column
ols = sm.OLS(market["log_ratio"], Xc).fit()                         # OLS: biased by design
alpha_ols = float(ols.params["price"])                              # the biased estimate
# two-stage least squares with the cost shifter as the instrument for price
Z = Xc.drop(columns=["price"]).copy()                              # instruments: exogenous columns only
Z["cost"] = market["cost"]                                          # the excluded instrument
first = sm.OLS(Xc["price"], sm.add_constant(Z)).fit()               # the first stage
price_hat = first.fittedvalues                                      # the fitted price
X2 = Xc.copy()                                                     # the second-stage design
X2["price"] = price_hat                                            # replace price by its projection
second = sm.OLS(market["log_ratio"], X2).fit()                      # the second stage
alpha_iv = float(second.params["price"])                            # the IV estimate
print("price coefficient: true %.2f | OLS %.3f | 2SLS %.3f | first-stage F %.1f" % (
    alpha_true, alpha_ols, alpha_iv, float(first.fvalue)))          # report
overid = pd.DataFrame([{"method": "OLS", "alpha": round(alpha_ols, 3)},
                       {"method": "2SLS", "alpha": round(alpha_iv, 3)},
                       {"method": "true", "alpha": alpha_true}])    # the comparison table
overid.to_csv(DAT / "l6_logit_estimates.csv", index=False, encoding="utf-8-sig")   # persist

# ---------------------------------------------------------------- 3) elasticities from the logit
last = market[market["market"] == T - 1].reset_index(drop=True)      # the final market
p = last["price"].to_numpy()                                        # prices in that market
s = last["share"].to_numpy()                                        # inside shares
s0 = float(last["share_out"].iloc[0])                               # the outside share
alpha = alpha_iv                                                    # use the IV estimate
elastic = np.zeros((4, 4))                                          # the elasticity matrix
for j in range(4):                                                  # own elasticities
    elastic[j, j] = alpha * p[j] * (1 - s[j])                        # d log s_j / d log p_j
for j in range(4):                                                  # cross elasticities
    for k in range(4):                                              # loop over the other brands
        if j != k:                                                  # off-diagonal only
            elastic[j, k] = -alpha * p[k] * s[k]                    # d log s_j / d log p_k
own = np.diag(elastic)                                              # the own-price elasticities
print("own elasticities:", np.round(own, 2))                        # report
print("cross A->B: %.2f | A->C: %.2f | A->D: %.2f" % (elastic[0, 1], elastic[0, 2], elastic[0, 3]))
pd.DataFrame(elastic.round(3), index=BRANDS, columns=BRANDS).to_csv(
    DAT / "l6_elasticity_matrix.csv", encoding="utf-8-sig")          # persist

# ---------------------------------------------------------------- 4) diversion and a merger
def shares_at(price_vec):                                          # shares implied by prices
    """Return (inside shares, outside share) for a price vector."""   # short docstring
    delta = np.array([intercepts[j] + alpha * price_vec[j] + beta_true * last["feature"].iloc[j]
                      for j in range(4)])                           # the mean utilities
    expd = np.exp(delta)                                            # exponentiate
    inside = expd / (1 + expd.sum())                                # inside shares
    return inside, 1.0 / (1 + expd.sum())                           # inside and outside

base_inside, base_out = shares_at(p)                                # the baseline
bump = p.copy(); bump[0] *= 1.05                                     # a 5% price rise for brand A
new_inside, new_out = shares_at(bump)                               # the new shares
lost = base_inside[0] - new_inside[0]                               # share lost by A
divert = (new_inside[1] - base_inside[1]) / lost if lost > 0 else np.nan   # diverted to B
print("A +5%%: share %.4f -> %.4f | diversion to B %.2f | to outside %.2f" % (
    base_inside[0], new_inside[0], divert, (new_out - base_out) / lost))   # report
# a merger of A and B: internalise their cross elasticity, so the joint price rises
merged_price = p.copy()                                             # start from the baseline
merged_price[0] *= 1.06                                             # A rises 6%
merged_price[1] *= 1.06                                             # B rises 6%
m_inside, m_out = shares_at(merged_price)                           # the post-merger shares
merger = pd.DataFrame([{"stage": "并购前", "share_A": round(float(base_inside[0]), 4),
                        "share_B": round(float(base_inside[1]), 4),
                        "outside": round(float(base_out), 4)},
                       {"stage": "并购后（合并品牌各提价 6%）", "share_A": round(float(m_inside[0]), 4),
                        "share_B": round(float(m_inside[1]), 4),
                        "outside": round(float(m_out), 4)}])        # the table
merger.to_csv(DAT / "l6_merger_simulation.csv", index=False, encoding="utf-8-sig")   # persist
print(merger.to_string(index=False))                                # report

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))                 # three panels
im = axes[0].imshow(elastic, cmap="RdBu_r", vmin=-3, vmax=3)        # the elasticity matrix
axes[0].set_xticks(range(4), [b.split()[0] for b in BRANDS], fontsize=8)   # x labels
axes[0].set_yticks(range(4), [b.split()[0] for b in BRANDS], fontsize=8)   # y labels
axes[0].grid(False)                                                # no grid in the heatmap
for j in range(4):                                                 # write the numbers
    for k in range(4):                                             # loop over columns
        axes[0].text(k, j, "%.2f" % elastic[j, k], ha="center", va="center",
                     fontsize=7.6, color="white" if abs(elastic[j, k]) > 1.6 else "#2B2B2B")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03, label="弹性")   # colour bar
axes[0].set_title("价格弹性矩阵：对角线是自身弹性")                     # panel title
axes[1].bar([b.split()[0] for b in BRANDS], own, color=[OKABE[5], OKABE[5], OKABE[2], OKABE[2]],
            alpha=0.9)                                             # own elasticities
for i, v in enumerate(own):                                        # annotate
    axes[1].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=8.6)   # the value
axes[1].set_ylabel("自身价格弹性")                                   # label
axes[1].set_title("自有品牌弹性更大（份额高 → 弹性小）")                 # panel title
groups = ["并购前 A", "并购后 A", "并购前 B", "并购后 B"]               # the four bars
vals = [base_inside[0], m_inside[0], base_inside[1], m_inside[1]]     # their values
axes[2].bar(groups, vals, color=[OKABE[5], OKABE[5], OKABE[2], OKABE[2]], alpha=0.55)   # the bars
for i, v in enumerate(vals):                                       # annotate
    axes[2].text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=8.4)   # the value
axes[2].set_xticks(range(4), groups, rotation=18, ha="right", fontsize=8)   # labels
axes[2].set_ylabel("市场份额")                                      # label
axes[2].set_title("并购模拟：合并品牌各提价 6%，份额下降")               # panel title
plt.tight_layout()                                                 # tidy the layout
fig.savefig(FIG / "fig_discrete_choice.png", facecolor="white")      # save
plt.close(fig)                                                     # release memory
print("[figure]", FIG / "fig_discrete_choice.png")                   # log it
print("done: logit demand, elasticities and merger simulation")       # closing log
