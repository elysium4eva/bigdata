"""l15_03_choice_logit.py -- conditional logit estimation, marginal effects and the IIA test.

Every line carries an English comment. The script widens the choice data into alternative rows,
estimates a conditional logit with statsmodels, recovers the true coefficients, computes
marginal effects and demonstrates why the independence of irrelevant alternatives fails when a
new alternative is added.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.api as sm                                       # the MNLogit estimator
from _common_l15 import (make_choice_data, save, dump,             # shared helpers
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the choice data
choice = make_choice_data(800)                                     # 800 choice occasions
dump(choice, "l15_choice_wide.csv")                                # persist the wide frame
print("occasions:", len(choice), "| share:", choice["chosen"].value_counts(normalize=True).round(3).to_dict())   # report

# ---------------------------------------------------------------- 2) reshape to long (alternative-level) rows
rows = []                                                          # one row per alternative
for _, r in choice.iterrows():                                     # loop over occasions
    for k in (1, 2, 3):                                            # three alternatives
        rows.append({"choice_id": r["choice_id"], "alt": k,         # ids
                     "brand": r["brand_%d" % k],                    # brand
                     "price": r["price_%d" % k],                    # price
                     "range": r["range_%d" % k],                    # range
                     "fast": r["fast_%d" % k],                      # charging minutes
                     "chosen": int(r["chosen"] == r["brand_%d" % k])})   # 1 for the chosen alternative
long = pd.DataFrame(rows)                                          # assemble the long frame
dump(long, "l15_choice_long.csv")                                  # persist
print("alternative rows:", len(long))                               # report

# ---------------------------------------------------------------- 3) conditional logit by maximum likelihood
# The likelihood of a conditional logit is P(chosen) = exp(V_j) / sum_k exp(V_k) within a choice set.
X = long[["price", "range", "fast"]].values                        # alternative attributes
X = np.column_stack([X, np.where(long["brand"] == "A", 1.0, 0.0),   # brand intercept for A
                     np.where(long["brand"] == "B", 1.0, 0.0)])     # brand intercept for B
y = long["chosen"].values                                          # the outcome
groups = long["choice_id"].values                                   # the choice set id

def neg_loglik(beta):                                              # negative log-likelihood
    """Return the negative conditional log-likelihood of the parameters."""   # short docstring
    v = X @ beta                                                   # deterministic utilities
    total = 0.0                                                    # accumulator
    for g in np.unique(groups):                                    # loop over choice sets
        mask = groups == g                                         # the rows of this set
        vg = v[mask]                                               # utilities in the set
        vg = vg - vg.max()                                         # stabilise with the max trick
        denom = np.log(np.exp(vg).sum())                            # log-sum-exp denominator
        pick = y[mask].astype(bool)                                # the chosen row
        total += (vg[pick].sum() - denom)                          # log probability of the choice
    return -total                                                  # maximise, so negate

from scipy.optimize import minimize                                # the optimiser
result = minimize(neg_loglik, np.zeros(X.shape[1]), method="BFGS")   # start from zeros
beta = result.x                                                    # the estimates
names = ["price", "range", "fast", "brandA", "brandB"]              # coefficient names
print("estimates:", dict(zip(names, np.round(beta, 4))))            # report

# ---------------------------------------------------------------- 4) standard errors from the Hessian
def hessian(func, theta, eps=1e-4):                                # numerical Hessian
    """Return the finite-difference Hessian of a scalar function."""   # short docstring
    n = len(theta)                                                 # number of parameters
    H = np.zeros((n, n))                                           # the output matrix
    for i in range(n):                                             # loop over parameters
        for j in range(i, n):                                      # upper triangle
            e_i = np.zeros(n); e_i[i] = eps                        # step for i
            e_j = np.zeros(n); e_j[j] = eps                        # step for j
            H[i, j] = (func(theta + e_i + e_j) - func(theta + e_i - e_j)   # central difference
                       - func(theta - e_i + e_j) + func(theta - e_i - e_j)) / (4 * eps * eps)   # four points
            H[j, i] = H[i, j]                                      # symmetric
    return H                                                       # return the matrix

H = hessian(neg_loglik, beta)                                      # Hessian at the estimates
se = np.sqrt(np.diag(np.linalg.inv(H)))                            # standard errors
z = beta / se                                                      # z statistics
pvals = 2 * (1 - sm.distributions.ECDF(np.abs(z))) if False else 2 * (1 - __import__("scipy").stats.norm.cdf(np.abs(z)))   # two-sided p
table = pd.DataFrame({"term": names, "coef": np.round(beta, 4), "se": np.round(se, 4),   # the table
                      "z": np.round(z, 2), "p": np.round(pvals, 4)})   # inference columns
dump(table, "l15_choice_logit.csv")                                # persist
print(table.to_string(index=False))                                # report
ll = -neg_loglik(beta)                                             # the maximised log-likelihood
ll0 = -neg_loglik(np.zeros_like(beta))                              # the null log-likelihood
print("McFadden pseudo R2: %.3f" % (1 - ll / ll0))                  # report the fit

# ---------------------------------------------------------------- 5) marginal effects and elasticities
def choice_probs(price, rng_km, fast, brand="A"):                   # probabilities for one profile
    """Return the three alternatives' probabilities for a synthetic choice set."""   # short docstring
    V = np.array([X[0, :3] @ beta[:3] + beta[3],                    # alternative A
                  np.array([price, rng_km, fast]) @ beta[:3] + beta[4],   # alternative B with this profile
                  np.array([28.0, 500, 30]) @ beta[:3] + beta[2] * 0])   # alternative C at the worst levels
    V = V - V.max()                                                # stabilise
    e = np.exp(V)                                                  # exponentials
    return e / e.sum()                                             # probabilities

p_base = choice_probs(23.0, 650, 20)                               # baseline probabilities
p_price = choice_probs(18.0, 650, 20)                              # a 5 万元 discount on B
print("P(B) at 23万: %.3f | at 18万: %.3f" % (p_base[1], p_price[1]))   # report the price effect
elasticity = ((p_price[1] - p_base[1]) / p_base[1]) / ((18.0 - 23.0) / 23.0)   # arc elasticity
print("own-price elasticity of choice probability: %.2f" % elasticity)   # report

# ---------------------------------------------------------------- 6) the IIA demonstration
base_share = choice_probs(23.0, 650, 20)                           # three-alternative shares
V_extra = np.array([28.0, 500, 30]) @ beta[:3] + 0.0                # a new, inferior alternative
extended = np.append(np.exp(np.array([X[0, :3] @ beta[:3] + beta[3],   # A
                                      np.array([23.0, 650, 20]) @ beta[:3] + beta[4],   # B
                                      np.array([28.0, 500, 30]) @ beta[:3],   # C
                                      V_extra])), 0)                # placeholder
v_ext = np.array([X[0, :3] @ beta[:3] + beta[3],                    # A
                  np.array([23.0, 650, 20]) @ beta[:3] + beta[4],    # B
                  np.array([28.0, 500, 30]) @ beta[:3],              # C
                  V_extra])                                         # D (a near-duplicate of C)
v_ext = v_ext - v_ext.max()                                         # stabilise
share_ext = np.exp(v_ext) / np.exp(v_ext).sum()                     # shares with the new alternative
print("shares before:", np.round(base_share, 3), "| after adding D:", np.round(share_ext, 3))   # report
ratio_before = base_share[0] / base_share[1]                        # A/B odds before
ratio_after = share_ext[0] / share_ext[1]                           # A/B odds after
print("A/B odds before %.3f | after %.3f (conditional logit keeps them equal: that is IIA)" % (ratio_before, ratio_after))   # report

# ---------------------------------------------------------------- 7) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                 # three panels
prices = np.linspace(14, 34, 80)                                   # a price grid
for k, km in enumerate([500, 650, 800]):                           # one curve per range level
    probs = [choice_probs(pr, km, 20)[1] for pr in prices]           # P(choose B)
    axes[0].plot(prices, probs, lw=1.8, color=OKABE[k], label="续航 %dkm" % km)   # the curve
style_axes(axes[0], "Choice probability versus price", "价格 (万元)", "P(选择 B)")   # shared style
axes[0].legend(fontsize=7.4)                                        # compact legend
axes[1].bar(range(3), base_share, color=[OKABE[4], OKABE[2], OKABE[5]], alpha=0.9)   # shares before
axes[1].set_xticks(range(3), ["A", "B", "C"])                       # brand labels
style_axes(axes[1], "Shares with three alternatives", "", "share")   # shared style
axes[2].bar(range(4), share_ext, color=[OKABE[4], OKABE[2], OKABE[5], OKABE[0]], alpha=0.9)   # shares after
axes[2].set_xticks(range(4), ["A", "B", "C", "D(新增)"], fontsize=8)   # brand labels
style_axes(axes[2], "Shares after adding a similar D", "", "share")   # shared style
annotated(axes[2], "A/B 比值不变 = IIA 成立", "upper right")          # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_logit_curves.png")                                  # save the figure
print("done: conditional logit, marginal effects and the IIA check")   # closing log
