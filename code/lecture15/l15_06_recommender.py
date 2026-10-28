"""l15_06_recommender.py -- matrix factorisation by alternating least squares and top-k evaluation.

Every line carries an English comment. The script factorises a 500x60 rating matrix with a
hand-written ALS loop (no LightFM dependency), measures precision@k against a held-out set, and
shows the cold-start gap so the deck can explain what a recommender can and cannot do.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l15 import (make_rating_matrix, save, dump,           # shared helpers
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the rating matrix
ratings, truth = make_rating_matrix(500, 60, 4, 0.16)              # 500 users x 60 items, 16% observed
dump(ratings, "l15_ratings.csv")                                   # persist the long frame
users = sorted(ratings["user"].unique())                           # user ids
items = sorted(ratings["item"].unique())                           # item ids
u_index = {u: i for i, u in enumerate(users)}                      # user -> row
i_index = {i: j for j, i in enumerate(items)}                      # item -> column
R = np.full((len(users), len(items)), np.nan)                      # the dense matrix with holes
for u, i, r in zip(ratings["user"], ratings["item"], ratings["rating"]):   # fill the observed entries
    R[u_index[u], i_index[i]] = r                                  # store the rating
print("matrix:", R.shape, "| observed: %.3f" % np.isfinite(R).mean())   # report

# ---------------------------------------------------------------- 2) train/test split on observed entries
mask = np.isfinite(R)                                              # observed positions
obs = np.argwhere(mask)                                            # their coordinates
rng = np.random.default_rng(0)                                     # a generator for the split
shuffled = obs[rng.permutation(len(obs))]                          # shuffle the coordinates
n_test = int(0.2 * len(shuffled))                                  # twenty percent for testing
test = shuffled[:n_test]                                           # test coordinates
train = shuffled[n_test:]                                          # training coordinates
Rtr = np.full_like(R, np.nan)                                      # training matrix
Rtr[train[:, 0], train[:, 1]] = R[train[:, 0], train[:, 1]]         # copy training ratings
print("train/test:", len(train), len(test))                         # report

# ---------------------------------------------------------------- 3) ALS from scratch
def als(Rmat, k=8, reg=0.08, iters=25):                            # alternating least squares
    """Return user and item factors fitted by ALS on a matrix with missing entries."""   # short docstring
    n_u, n_i = Rmat.shape                                          # dimensions
    U = rng.normal(0, 0.1, (n_u, k))                               # user factors
    V = rng.normal(0, 0.1, (n_i, k))                               # item factors
    for _ in range(iters):                                         # alternating updates
        for u in range(n_u):                                       # fix V, solve for each user
            idx = np.where(np.isfinite(Rmat[u]))[0]                # items this user rated
            if len(idx) == 0:                                      # no data for this user
                continue                                           # keep the initial factors
            Vt = V[idx]                                            # the relevant item factors
            A = Vt.T @ Vt + reg * np.eye(k)                        # normal equations
            b = Vt.T @ Rmat[u, idx]                                # right-hand side
            U[u] = np.linalg.solve(A, b)                           # solve
        for i in range(n_i):                                       # fix U, solve for each item
            idx = np.where(np.isfinite(Rmat[:, i]))[0]             # users who rated this item
            if len(idx) == 0:                                      # no data
                continue                                           # keep the initial factors
            Ut = U[idx]                                            # the relevant user factors
            A = Ut.T @ Ut + reg * np.eye(k)                        # normal equations
            b = Ut.T @ Rmat[idx, i]                                # right-hand side
            V[i] = np.linalg.solve(A, b)                           # solve
    return U, V                                                    # return the factors

U, V = als(Rtr, k=8, reg=0.08, iters=25)                            # fit the model
pred = U @ V.T                                                     # the full prediction matrix
rmse = float(np.sqrt(np.mean((pred[test[:, 0], test[:, 1]] - R[test[:, 0], test[:, 1]]) ** 2)))   # test RMSE
print("test RMSE: %.3f" % rmse)                                    # report

# ---------------------------------------------------------------- 4) precision@k
def precision_at_k(k=10, n_eval=100):                              # ranking quality on held-out items
    """Return mean precision@k over a sample of users."""           # short docstring
    hits, total = 0, 0                                             # accumulators
    for u in range(min(n_eval, len(users))):                       # loop over users
        seen = np.isfinite(Rtr[u])                                 # items already in the training data
        test_items = test[test[:, 0] == u][:, 1]                    # held-out items for this user
        if len(test_items) == 0:                                   # skip users without a hold-out
            continue                                               # next user
        scores = pred[u].copy()                                    # predicted scores
        scores[seen] = -np.inf                                     # do not recommend seen items
        top = np.argsort(scores)[::-1][:k]                          # the top-k ranking
        hits += len(set(top) & set(test_items))                     # hits among the top-k
        total += 1                                                 # one more user evaluated
    return hits / (total * k) if total else 0.0                     # mean precision

p10 = precision_at_k(10)                                           # precision at ten
print("precision@10: %.3f" % p10)                                  # report

# ---------------------------------------------------------------- 5) cold start
rows = []                                                          # one row per profile size
for n_seen in [1, 2, 4, 8, 16, 32]:                                # increasing history lengths
    scores = []                                                    # RMSE values
    for _ in range(40):                                            # repeat for stability
        u = int(rng.integers(0, len(users)))                       # a random user
        idx = np.where(np.isfinite(Rtr[u]))[0]                     # the items they rated
        if len(idx) < n_seen:                                      # not enough history
            continue                                               # skip
        keep = rng.choice(idx, n_seen, replace=False)              # keep only n_seen items
        Rsub = Rtr.copy()                                          # a copy of the training matrix
        Rsub[u, np.setdiff1d(idx, keep)] = np.nan                   # hide the rest
        Us, Vs = als(Rsub, k=8, reg=0.08, iters=8)                  # refit
        hidden = np.setdiff1d(idx, keep)                            # the hidden items
        if len(hidden) == 0:                                       # nothing hidden
            continue                                               # skip
        ps = Us[u] @ Vs[hidden].T                                  # predictions for hidden items
        scores.append(float(np.sqrt(np.mean((ps - R[u, hidden]) ** 2))))   # RMSE
    rows.append({"history": n_seen, "rmse": round(float(np.mean(scores)), 3) if scores else np.nan})   # the row
cold = pd.DataFrame(rows)                                          # assemble
dump(cold, "l15_cold_start.csv")                                   # persist
print(cold.to_string(index=False))                                 # report

# ---------------------------------------------------------------- 6) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                 # three panels
axes[0].imshow(np.where(np.isfinite(Rtr), Rtr, np.nan)[:80, :40], cmap="RdPu", aspect="auto")   # the sparse matrix
axes[0].set_title("Rating matrix (80 users x 40 items)")            # panel title
axes[0].set_xlabel("item")                                         # axis label
axes[0].set_ylabel("user")                                         # axis label
axes[0].grid(False)                                                # a heatmap needs no grid
axes[1].plot(cold["history"], cold["rmse"], "-o", ms=5, color=OKABE[5])   # cold-start curve
style_axes(axes[1], "Cold start: fewer ratings, worse RMSE", "history length", "RMSE")   # shared style
annotated(axes[1], "新用户最吃亏", "upper right")                    # annotate
axes[2].bar(["precision@10"], [p10], color=OKABE[2], alpha=0.9)     # the ranking metric
axes[2].set_ylim(0, max(0.3, p10 * 1.4))                            # room for the label
axes[2].text(0, p10, "%.3f" % p10, ha="center", va="bottom", fontsize=10)   # value
style_axes(axes[2], "Top-k precision on held-out items", "", "precision@10")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_recommender.png")                                   # save the figure
print("done: ALS matrix factorisation, precision@k and cold start")   # closing log
