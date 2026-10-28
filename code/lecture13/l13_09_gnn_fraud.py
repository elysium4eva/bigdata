"""l13_09_gnn_fraud.py -- graph-based fraud detection with hand-written message passing.

Every line carries an English comment. The script builds a transaction graph with a few dense fraud   # continuation of the call
rings, trains a one-layer graph network in numpy, and compares its recall at a fixed review budget   # continuation of the call
with a feature-only logistic regression so the deck can argue why fraud is a graph problem.   # continuation of the call
"""
import numpy as np                                                 # arrays and message passing
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.linear_model import LogisticRegression                 # the feature-only baseline
from sklearn.model_selection import train_test_split                # the split
from sklearn.metrics import roc_auc_score, precision_recall_curve    # the metrics
from _common_l13 import (make_fraud_graph, RNG, save, dump,         # shared helpers
                         style_axes, annotated, OKABE)                  # continuation of the call

# ---------------------------------------------------------------- 1) the graph
nodes, edges = make_fraud_graph(900, 6)                             # 900 nodes, six fraud rings
dump(nodes, "l13_fraud_nodes.csv")                                  # persist
dump(edges.head(2000), "l13_fraud_edges.csv")                       # persist a sample of the edges
n = len(nodes)                                                      # number of nodes
A = np.zeros((n, n))                                                # adjacency matrix
for src, dst in zip(edges["src"], edges["dst"]):                     # fill the edges
    A[src, dst] = 1.0                                                # one for an edge
A = np.maximum(A, A.T)                                              # treat the graph as undirected
A_bin = (A > 0).astype(float)                                       # the binary structure
deg = A_bin.sum(axis=1, keepdims=True)                              # node degrees
An = (A_bin + np.eye(n)) / np.maximum(deg + 1.0, 1.0)                # row-normalised with self loops
labels = nodes["label"].values.astype(float)                         # the fraud labels
print("nodes:", n, "| edges:", int(A_bin.sum()), "| fraud rate %.3f" % labels.mean())   # report

# ---------------------------------------------------------------- 2) node features
# A simple degree-based feature: fraud rings are dense, so degree is informative but not enough.
# A weak node-level signal (think "average transaction size"): it points at fraud but is noisy,
# so smoothing it over the neighbourhood is exactly where message passing helps.
weak = (labels * 2 - 1) * 0.9 + RNG.normal(0, 1.3, n)               # the noisy signal
X = np.column_stack([weak, np.log1p(deg.ravel())])                  # the feature matrix
X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)          # standardise

# ---------------------------------------------------------------- 3) one message-passing layer
def forward(W_self, W_neigh, bias, feats):                         # the GCN layer
    """Return the node embedding after one message-passing step."""   # short docstring
    agg = An @ feats                                               # aggregate the neighbours
    return np.maximum(feats @ W_self + agg @ W_neigh + bias, 0.0)   # linear plus ReLU

params = (RNG.normal(0, 0.3, (2, 8)), RNG.normal(0, 0.3, (2, 8)),    # self and neighbour weights
          np.zeros(8), RNG.normal(0, 0.2, 8))                        # bias and output weights
def loss_and_grad(p):                                              # loss and gradients
    """Return the weighted MSE and the gradients for the parameter tuple."""   # short docstring
    Ws, Wn, b, wo = p                                              # unpack
    H = forward(Ws, Wn, b, X)                                      # the hidden layer
    pred = H @ wo                                                  # the scores
    # class weights: fraud is rare, so we up-weight it in the loss
    w = np.where(labels > 0.5, 8.0, 1.0)                           # the sample weights
    err = (pred - (labels * 2 - 1)) * w / len(labels)               # weighted residual on the +/-1 target
    loss = float(np.mean(err ** 2))                                 # the loss
    d_pred = 2 * err / len(labels)                                  # the gradient
    d_wo = H.T @ d_pred                                            # for the output layer
    d_H = np.outer(d_pred, wo) * (H > 0)                            # through the ReLU
    return loss, (X.T @ d_H, (An @ X).T @ d_H, d_H.sum(axis=0), d_wo)   # the gradients

for step in range(500):                                            # gradient descent
    loss, grads = loss_and_grad(params)                             # one evaluation
    params = tuple(p - 0.15 * g for p, g in zip(params, grads))      # one update
print("GNN loss: %.4f" % loss)                                       # report
score_gnn = forward(*params[:3], X) @ params[3]                     # the fraud score

# ---------------------------------------------------------------- 4) the feature-only baseline
idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.3, random_state=42, stratify=labels)   # split nodes
base = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[idx_tr], labels[idx_tr])   # features only
score_base = base.decision_function(X)                              # the baseline score

# ---------------------------------------------------------------- 5) recall at a fixed review budget
# evaluate on the held-out nodes only, otherwise the AUC is inflated by in-sample predictions
y_te = labels[idx_te]                                              # the test labels
budget = max(1, int(0.05 * len(idx_te)))                            # review 5% of the test nodes
def recall_at_budget(score):                                       # top-k recall on the test set
    """Return the share of frauds caught when reviewing the top-k test nodes."""   # short docstring
    order = np.argsort(score)[::-1][:budget]                        # the top-k nodes
    return float(y_te[order].sum() / max(y_te.sum(), 1))             # recall
rec_gnn = recall_at_budget(score_gnn[idx_te])                       # the GNN recall
rec_base = recall_at_budget(score_base[idx_te])                     # the baseline recall
auc_gnn = float(roc_auc_score(y_te, score_gnn[idx_te]))              # the GNN AUC
auc_base = float(roc_auc_score(y_te, score_base[idx_te]))            # the baseline AUC
print("recall@5%%: GNN %.3f | features %.3f" % (rec_gnn, rec_base))    # report
print("AUC: GNN %.3f | features %.3f" % (auc_gnn, auc_base))          # report
dump(pd.DataFrame([{"model": "GNN (1 layer)", "auc": round(auc_gnn, 4), "recall_at_5pct": round(rec_gnn, 4)},   # assemble the frame
                   {"model": "features only", "auc": round(auc_base, 4), "recall_at_5pct": round(rec_base, 4)}]),   # continuation of the call
     "l13_fraud_compare.csv")                                       # persist

# ---------------------------------------------------------------- 6) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                  # three panels
deg_fraud = np.log1p(deg.ravel())[labels > 0.5]                      # degrees of fraud nodes
deg_ok = np.log1p(deg.ravel())[labels < 0.5]                         # degrees of normal nodes
axes[0].hist(deg_ok, bins=30, alpha=0.6, color=OKABE[2], label="正常节点")   # the normal nodes
axes[0].hist(deg_fraud, bins=30, alpha=0.7, color=OKABE[5], label="欺诈节点")   # the fraud nodes
axes[0].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[0], "欺诈节点的度明显更高", "log(1+度)", "节点数")        # shared style
for name, s, colour in [("特征模型", score_base, OKABE[1]), ("GNN", score_gnn, OKABE[5])]:   # two scores
    prec, rec, _ = precision_recall_curve(y_te, s[idx_te])           # the PR curve on the test set
    axes[1].plot(rec, prec, lw=1.8, color=colour, label=name)         # the curve
axes[1].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[1], "精确率—召回率曲线", "召回率", "精确率")             # shared style
axes[2].bar(["特征模型", "GNN"], [rec_base, rec_gnn], color=[OKABE[1], OKABE[5]], alpha=0.92)   # the budget comparison
style_axes(axes[2], "复核 5%% 节点能抓到多少欺诈", "", "召回率")          # shared style
for i, v in enumerate([rec_base, rec_gnn]):                          # annotate
    axes[2].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=9)   # the value
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_fraud_gnn.png")                                      # save the figure
print("done: graph fraud detection versus a feature-only model")       # closing log
