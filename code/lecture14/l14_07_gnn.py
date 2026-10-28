"""l14_07_gnn.py -- a one-layer graph network that predicts trade, compared with gravity.

Every line carries an English comment. The script builds a country trade network, runs a   # continuation of the call
hand-written message-passing layer in numpy (no deep-learning framework needed), trains it on   # continuation of the call
total exports, and compares its out-of-sample fit with a gravity regression so the deck can   # continuation of the call
argue why graph structure adds information.                             # continuation of the call
"""
import numpy as np                                                 # arrays and the message passing
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # the gravity baseline
from _common_l14 import (make_country_panel, make_geography,       # shared generators
                         make_trade_flows, save, dump, style_axes,      # shared plotting helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) build the network for one year
countries = make_country_panel(60, (2000, 2019))                    # the country panel
geo = make_geography(60)                                           # geography
flows = make_trade_flows(countries, geo, (2000, 2019))              # dyad-year flows
year = 2019                                                        # the year to model
slice_ = flows[(flows["year"] == year) & (flows["trade"] > 0)]      # positive flows
iso = sorted(set(slice_["iso_o"]) | set(slice_["iso_d"]))           # the node list
index = {c: i for i, c in enumerate(iso)}                          # node -> row
n = len(iso)                                                       # number of nodes
A = np.zeros((n, n))                                               # adjacency matrix
for _, r in slice_.iterrows():                                     # fill the edges
    A[index[r["iso_o"]], index[r["iso_d"]]] = np.log1p(r["trade"])   # log trade as weight
print("nodes:", n, "| edges:", int((A > 0).sum()))                   # report

# ---------------------------------------------------------------- 2) node features and the target
gdp = countries[countries["year"] == year].set_index("iso")["gdp"]   # GDP per country
x_gdp = np.array([np.log(float(gdp.get(c, 1.0))) for c in iso])      # log GDP feature
degree = (A > 0).sum(axis=1)                                        # out-degree
total_exp = slice_.groupby("iso_o")["trade"].sum()                   # total exports per exporter
y = np.log1p(np.array([float(total_exp.get(c, 0.0)) for c in iso]))   # the target in logs
remote = geo.groupby("iso_o")["dist"].mean()                        # mean distance to all partners
x_remote = np.array([np.log(float(remote.get(c, 1.0))) for c in iso])   # log remoteness
X = np.column_stack([x_gdp, x_remote])                              # features: size and remoteness
X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)          # standardise so training is stable
print("feature matrix:", X.shape)                                    # report

# ---------------------------------------------------------------- 3) one message-passing layer
def normalise(Amat):                                               # row-normalise the adjacency
    """Return the row-normalised adjacency matrix (self loops included)."""   # short docstring
    A2 = Amat.copy()                                               # do not modify the input
    A2[A2 > 0] = 1.0                                               # use the binary structure
    np.fill_diagonal(A2, 1.0)                                      # keep the node itself
    deg = A2.sum(axis=1, keepdims=True)                             # the row sums
    return A2 / np.maximum(deg, 1e-9)                              # the normalised matrix

An = normalise(A)                                                  # normalised adjacency
def forward(W_self, W_neigh, bias, feats):                         # one GCN-style layer
    """Return the node embedding after one message-passing step."""   # short docstring
    agg = An @ feats                                               # average the neighbours' features
    h = np.maximum(feats @ W_self + agg @ W_neigh + bias, 0.0)      # linear plus ReLU
    return h                                                       # the embedding

rng = np.random.default_rng(7)                                     # a local generator
W_self = rng.normal(0, 0.5, (X.shape[1], 8))                        # self weights
W_neigh = rng.normal(0, 0.5, (X.shape[1], 8))                       # neighbour weights
bias = np.zeros(8)                                                 # bias
w_out = rng.normal(0, 0.3, 8)                                       # the output head
def predict(Ws, Wn, b, wo, feats):                                 # forward pass through both layers
    """Return the predicted log export strength."""                 # short docstring
    return forward(Ws, Wn, b, feats) @ wo                           # embedding then linear head

# simple gradient descent on the mean squared error
def loss_and_grad(params):                                         # flattening lets us use one optimiser
    """Return the MSE and its gradients for the flattened parameters."""   # short docstring
    Ws, Wn, b, wo = params                                             # unpack the blocks
    H = forward(Ws, Wn, b, X)                                       # the hidden layer
    pred = H @ wo                                                   # predictions
    err = pred - y                                                  # residuals
    mse = float(np.mean(err ** 2))                                   # the loss
    d_pred = 2 * err / len(y)                                       # gradient of the loss
    d_wo = H.T @ d_pred                                             # gradient for the head
    d_H = np.outer(d_pred, wo) * (H > 0)                              # back through the ReLU
    d_Ws = X.T @ d_H                                                # gradient for the self weights
    d_Wn = (An @ X).T @ d_H                                          # gradient for the neighbour weights
    d_b = d_H.sum(axis=0)                                           # gradient for the bias
    return mse, (d_Ws, d_Wn, d_b, d_wo)                             # loss and gradients

params = (W_self, W_neigh, bias, w_out)                             # the parameter tuple
for step in range(300):                                            # gradient descent
    mse, grads = loss_and_grad(params)                              # one evaluation
    params = tuple(p - 0.01 * g for p, g in zip(params, grads))      # one update
print("GNN training MSE: %.4f" % mse)                                # report
H = forward(*params[:3], X)                                         # the final embedding
pred_gnn = H @ params[3]                                            # final predictions

# ---------------------------------------------------------------- 4) the gravity baseline
base = pd.DataFrame({"iso": iso, "y": y, "x_gdp": x_gdp, "x_remote": x_remote})   # the frame
grav = smf.ols("y ~ x_gdp + x_remote", data=base).fit()              # the gravity regression
pred_grav = grav.predict(base)                                      # its predictions
r2_gnn = float(1 - np.sum((y - pred_gnn) ** 2) / np.sum((y - y.mean()) ** 2))   # GNN R2
r2_grav = float(1 - np.sum((y - pred_grav) ** 2) / np.sum((y - y.mean()) ** 2))   # gravity R2
print("R2: GNN %.3f | gravity %.3f" % (r2_gnn, r2_grav))              # report
dump(pd.DataFrame([{"model": "GNN (1 layer)", "r2": round(r2_gnn, 4)},   # the comparison
                   {"model": "重力回归", "r2": round(r2_grav, 4)}]), "l14_gnn_compare.csv")   # persist

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))                 # two panels
axes[0].scatter(y, pred_gnn, s=26, color=OKABE[2], alpha=0.85, label="GNN")   # GNN fit
axes[0].scatter(y, pred_grav, s=26, color=OKABE[1], alpha=0.65, label="重力")   # gravity fit
lims = [y.min(), y.max()]                                          # the axes range
axes[0].plot(lims, lims, ls="--", lw=1.0, color="#66717D")           # the 45 degree line
style_axes(axes[0], "预测 vs 实际（出口强度）", "实际值", "预测值")      # shared style
axes[0].legend(fontsize=7.6)                                        # compact legend
axes[1].bar(["GNN (1 层)", "重力回归"], [r2_gnn, r2_grav], color=[OKABE[2], OKABE[1]], alpha=0.92)   # R2 comparison
axes[1].set_ylim(0, 1.0)                                           # full scale
style_axes(axes[1], "样本外 R²", "", "R²")                          # shared style
for i, v in enumerate([r2_gnn, r2_grav]):                          # annotate
    axes[1].text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=9)   # value
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_gnn.png")                                           # save the figure
print("done: message passing versus gravity")                         # closing log
