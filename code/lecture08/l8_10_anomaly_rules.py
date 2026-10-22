"""l8_10_anomaly_rules.py -- anomaly detection and association rules without labels.

Every line carries an English comment. The script contrast two anomaly detectors on a
contaminated sample, then mines association rules from a synthetic market-basket set
with a hand-written Apriori so the support, confidence and lift definitions become
concrete numbers rather than symbols.
"""
import itertools                                                   # enumerate the itemsets
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # two panels
from sklearn.ensemble import IsolationForest                       # trees based anomaly detector
from sklearn.neighbors import LocalOutlierFactor                   # density based anomaly detector
from itertools import combinations                                 # item pairs for the rules
from _common_l8 import (RNG, make_transactions, save, dump,        # shared helpers
                        style_axes, annotated, OKABE)                   # shared plotting helpers

# ---------------------------------------------------------------- 1) a contaminated sample
n_normal, n_anomaly = 500, 25                                      # most points are normal
normal = RNG.normal(0, 1, (n_normal, 2))                           # the normal cloud
anomaly = RNG.uniform(-6, 6, (n_anomaly, 2))                       # scattered anomalies
X = np.vstack([normal, anomaly])                                   # stack them
truth = np.r_[np.zeros(n_normal), np.ones(n_anomaly)].astype(int)   # 1 marks a true anomaly
frame = pd.DataFrame({"x1": X[:, 0].round(3), "x2": X[:, 1].round(3), "anomaly": truth})   # tabulate
dump(frame, "l8_anomaly.csv")                                      # persist it
print("rows:", len(X), "| anomalies:", int(truth.sum()))            # report

# ---------------------------------------------------------------- 2) two detectors
iso = IsolationForest(contamination=0.05, random_state=42).fit(X)   # isolation forest
lof = LocalOutlierFactor(contamination=0.05, novelty=False).fit(X)  # local outlier factor
iso_flag = (iso.predict(X) == -1).astype(int)                      # flag from the forest
lof_flag = (lof.fit_predict(X) == -1).astype(int)                  # flag from LOF
def precision_recall(flag, truth_labels):                          # quick precision and recall
    """Return precision and recall of a boolean flag against the truth."""   # short docstring
    tp = int(((flag == 1) & (truth_labels == 1)).sum())            # true positives
    fp = int(((flag == 1) & (truth_labels == 0)).sum())            # false positives
    fn = int(((flag == 0) & (truth_labels == 1)).sum())            # false negatives
    return (tp / max(tp + fp, 1), tp / max(tp + fn, 1))            # precision and recall
p_iso, r_iso = precision_recall(iso_flag, truth)                   # forest scores
p_lof, r_lof = precision_recall(lof_flag, truth)                   # LOF scores
print("isolation forest: precision %.2f | recall %.2f" % (p_iso, r_iso))   # report
print("LOF:              precision %.2f | recall %.2f" % (p_lof, r_lof))   # report
score_table = pd.DataFrame({"detector": ["isolation forest", "LOF"],   # detector names
                            "precision": [round(p_iso, 3), round(p_lof, 3)],   # precision
                            "recall": [round(r_iso, 3), round(r_lof, 3)]})   # recall
dump(score_table, "l8_anomaly_scores.csv")                         # persist

# ---------------------------------------------------------------- 3) association rules with Apriori
baskets = make_transactions(600, 12)                               # synthetic transactions
dump(pd.DataFrame({"basket": ["|".join(b) for b in baskets]}), "l8_baskets.csv")   # persist
min_support = 0.06                                                 # keep itemsets in at least six per cent of baskets
def support(itemset):                                              # support of an itemset
    """Return the share of baskets that contain the itemset."""     # short docstring
    target = set(itemset)                                          # the set we test
    return sum(1 for b in baskets if target.issubset(set(b))) / len(baskets)   # share of baskets
items = sorted({item for basket in baskets for item in basket})     # all distinct items
frequent = []                                                      # frequent itemsets of size 1 and 2
for size in (1, 2):                                                # single items and pairs
    for combo in combinations(items, size):                        # enumerate the combinations
        s = support(combo)                                         # compute the support
        if s >= min_support:                                       # keep it when frequent enough
            frequent.append((combo, s))                            # store it
print("frequent itemsets:", len(frequent))                          # report
rule_rows = []                                                     # one row per rule
for (itemset, s_itemset) in frequent:                              # candidate rule bodies
    if len(itemset) < 2:                                           # only pairs can produce a rule
        continue                                                   # skip singletons
    for a, b in [(itemset[0], itemset[1]), (itemset[1], itemset[0])]:   # both directions
        s_a = support([a])                                         # support of the antecedent
        confidence = s_itemset / s_a                               # confidence of a -> b
        lift = confidence / support([b])                           # lift of the rule
        if confidence >= 0.3 and lift > 1.1:                       # keep interesting rules
            rule_rows.append({"antecedent": a, "consequent": b,     # the rule
                              "support": round(s_itemset, 3),       # support of the pair
                              "confidence": round(confidence, 3),   # conditional probability
                              "lift": round(lift, 3)})              # lift above independence
rules = pd.DataFrame(rule_rows).sort_values("lift", ascending=False)   # strongest first
dump(rules, "l8_rules.csv")                                        # persist it
print(rules.head(8).to_string(index=False))                        # report the top rules

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                # three panels
axes[0].scatter(X[truth == 0, 0], X[truth == 0, 1], s=12, color="#B9B3B6", label="normal")   # normal points
axes[0].scatter(X[truth == 1, 0], X[truth == 1, 1], s=60, marker="X", color=OKABE[5], label="true anomaly")   # truth
style_axes(axes[0], "The contaminated sample", "x1", "x2")          # shared style
axes[0].legend(fontsize=7.2, loc="upper left")                     # compact legend
scores = iso.score_samples(X)                                      # lower means more abnormal
sc = axes[1].scatter(X[:, 0], X[:, 1], c=scores, cmap="viridis", s=16)   # colour by score
style_axes(axes[1], "Isolation forest scores", "x1", "x2")          # shared style
plt.colorbar(sc, ax=axes[1], fraction=0.046, label="score")        # colour bar
top_rules = rules.head(8)                                          # the strongest rules
axes[2].barh(top_rules["antecedent"] + " -> " + top_rules["consequent"],   # rule labels
             top_rules["lift"], color=OKABE[2], alpha=0.9)         # lift values
axes[2].axvline(1.0, ls="--", lw=1.0, color="#66717D")              # independence reference
style_axes(axes[2], "Top rules by lift", "lift", "")                # shared style
axes[2].tick_params(axis="y", labelsize=7.0)                        # small labels
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_anomaly.png")                                       # save the figure
print("done: anomaly detection and association rules")              # closing log
