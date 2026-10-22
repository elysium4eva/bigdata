"""l7_11_adaboost_scratch.py -- AdaBoost from scratch, one weak learner at a time.

Every line carries an English comment. The script implements AdaBoost with decision
stumps in NumPy, records the weighted error and the stagewise accuracy, and draws the
error curve so the "boosting corrects its own mistakes" story becomes visible.
"""
import numpy as np                                                 # arrays for the weights
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # the staged error curve
from sklearn.tree import DecisionTreeClassifier                    # the weak learner
from sklearn.datasets import make_moons                            # a non-linear two-class sample
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.metrics import accuracy_score                         # accuracy per stage
from _common_l7 import (SEED, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) the sample
X, y = make_moons(n_samples=300, noise=0.22, random_state=SEED)     # two intertwined crescents
y = np.where(y == 0, -1, 1)                                        # AdaBoost wants labels in {-1, +1}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)   # split
print("rows:", len(X), "| classes:", np.bincount(y + 1).tolist())    # report the sample

# ---------------------------------------------------------------- 2) AdaBoost
n_train = len(y_train)                                             # number of training rows
w = np.full(n_train, 1.0 / n_train)                                # start with uniform sample weights
rounds = 60                                                        # number of boosting rounds
history = []                                                       # per-round records
alphas = []                                                        # the weight of each weak learner
learners = []                                                      # the fitted stumps
for t in range(rounds):                                            # loop over the rounds
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)   # a decision stump
    stump.fit(X_train, y_train, sample_weight=w)                   # fit with the current weights
    pred = stump.predict(X_train)                                  # training predictions
    wrong = pred != y_train                                        # misclassified rows
    err = float(np.sum(w[wrong]) / np.sum(w))                      # weighted error
    err = min(max(err, 1e-10), 1 - 1e-10)                          # clip to keep the log finite
    alpha = 0.5 * np.log((1 - err) / err)                          # the learner's weight
    w = w * np.exp(-alpha * y_train * pred)                        # upweight the mistakes
    w = w / w.sum()                                                # renormalise
    alphas.append(alpha)                                           # remember the weight
    learners.append(stump)                                         # remember the learner
    train_acc = accuracy_score(y_train, np.sign(sum(a * m.predict(X_train) for a, m in zip(alphas, learners))))   # staged accuracy
    test_acc = accuracy_score(y_test, np.sign(sum(a * m.predict(X_test) for a, m in zip(alphas, learners))))     # test accuracy
    history.append({"round": t + 1, "weighted_error": round(err, 5),   # the error the stump made
                    "alpha": round(float(alpha), 5), "train_acc": round(float(train_acc), 4),   # learner weight and fit
                    "test_acc": round(float(test_acc), 4)})        # generalisation
    if (t + 1) % 10 == 0:                                          # report every ten rounds
        print("round %2d | err %.4f | alpha %.3f | test acc %.3f" % (t + 1, err, alpha, test_acc))   # log line
table = pd.DataFrame(history)                                      # assemble the table
dump(table, "l7_adaboost_stages.csv")                              # persist it
single = accuracy_score(y_test, learners[0].predict(X_test))       # the first stump alone
final = float(table["test_acc"].iloc[-1])                          # the full ensemble
print("single stump test accuracy %.3f -> ensemble %.3f" % (single, final))   # report the gain

# ---------------------------------------------------------------- 3) figure
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))                # three panels
axes[0].plot(table["round"], table["weighted_error"], "-o", ms=3.5, color=OKABE[5])   # weighted error per round
style_axes(axes[0], "Each stump's weighted error", "round", "weighted error")   # shared style
axes[0].set_yscale("log")                                          # errors span orders of magnitude
axes[1].plot(table["round"], table["train_acc"], "-", lw=1.9, color=OKABE[2], label="train")   # staged train accuracy
axes[1].plot(table["round"], table["test_acc"], "-", lw=1.9, color=OKABE[5], label="test")     # staged test accuracy
axes[1].axhline(single, ls="--", lw=1.0, color="#66717D", label="single stump")   # the baseline
style_axes(axes[1], "Boosting improves stage by stage", "round", "accuracy")   # shared style
axes[1].legend(fontsize=7.2, loc="lower right")                     # compact legend
axes[2].plot(table["round"], table["alpha"], "-o", ms=3.0, color=OKABE[3])   # learner weights
style_axes(axes[2], "Learner weights alpha", "round", "alpha")      # shared style
annotated(axes[2], "later stumps get\nsmaller weights", "upper right")   # annotate the shrinkage
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_adaboost_staged.png")                               # save the figure
print("done: AdaBoost from scratch over %d rounds" % rounds)        # closing log
