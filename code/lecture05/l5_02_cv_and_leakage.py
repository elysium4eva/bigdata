"""l5_02_cv_and_leakage.py -- cross-validation done right, and three ways to leak.

Every line carries an English comment. The script runs the same model under four
protocols (clean CV, scaling before the split, a leaked feature, duplicated rows)
and shows how optimistic the reported score becomes when the protocol is wrong.
"""
import numpy as np                                                 # numeric helpers for the leakage demo
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # figure with the protocol comparison
from sklearn.linear_model import LogisticRegression                # the model used in every protocol
from sklearn.metrics import roc_auc_score                          # metric reported for each protocol
from sklearn.model_selection import (StratifiedKFold,              # stratified k-fold splitter
                                     cross_val_score,              # single-metric cross-validation
                                     train_test_split)             # plain train/test split
from sklearn.pipeline import Pipeline                              # keep scaling inside the pipeline
from sklearn.preprocessing import StandardScaler                   # the transformer that is easy to misuse
from _common_l5 import DAT, save, dump, style_axes, OKABE          # shared helpers

# ---------------------------------------------------------------- 1) load the sample from lesson 5.1
credit = pd.read_csv(DAT / "l5_credit_default.csv")                # read the synthetic credit dataset
num_cols = credit.select_dtypes(include=[np.number]).columns.drop("default").tolist()   # numeric predictors
X = credit[num_cols]                                               # feature matrix with numeric columns only
y = credit["default"].values                                       # binary target
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)    # five stratified folds, fixed seed
print("rows:", len(credit), "| predictors:", len(num_cols), "| default rate: %.3f" % y.mean())   # quick sanity check

# ---------------------------------------------------------------- 2) protocol A: clean pipeline inside CV
clean = Pipeline(steps=[("scale", StandardScaler()),               # scaling is fitted on each training fold only
                        ("model", LogisticRegression(max_iter=2000, random_state=42))])   # classifier
scores_clean = cross_val_score(clean, X, y, cv=cv, scoring="roc_auc")   # honest cross-validated ROC-AUC
print("A clean pipeline   AUC = %.4f (+-%.4f)" % (scores_clean.mean(), scores_clean.std()))   # report mean and spread
model_only = LogisticRegression(max_iter=2000, random_state=42)    # a bare estimator for protocols that pre-process outside CV

# ---------------------------------------------------------------- 3) protocol B: feature selection before splitting (leak)
from sklearn.feature_selection import SelectKBest, f_classif      # leaky feature selection helper
selector = SelectKBest(score_func=f_classif, k=5).fit(X, y)        # WRONG: the selector sees every row, including the test rows
X_selected = pd.DataFrame(selector.transform(X),                   # keep only the five "best" columns
                          columns=X.columns[selector.get_support()])   # reuse the original column names
scores_scaled = cross_val_score(clean, X_selected, y, cv=cv, scoring="roc_auc")   # CV on the pre-selected frame
print("B selection leak   AUC = %.4f (+-%.4f)" % (scores_scaled.mean(), scores_scaled.std()))   # clears the honest score

# ---------------------------------------------------------------- 4) protocol C: a feature derived from the target
leak_feature = credit["pay_status_1"] * 0.0 + y * 0.9 + np.random.default_rng(7).normal(0, 0.1, len(credit))   # target leakage
X_target = X.copy()                                                # copy the honest feature matrix
X_target["next_month_overdue"] = leak_feature                      # WRONG: information only known after the fact
scores_target = cross_val_score(model_only, X_target, y, cv=cv, scoring="roc_auc")   # CV with the leaked column
print("C target leak      AUC = %.4f (+-%.4f)" % (scores_target.mean(), scores_target.std()))   # almost perfect

# ---------------------------------------------------------------- 5) protocol D: duplicated rows across folds
dup_idx = np.random.default_rng(11).choice(len(credit), 600, replace=False)   # pick 600 rows to duplicate
credit_dup = pd.concat([credit, credit.iloc[dup_idx]], ignore_index=True)   # WRONG: copies inflate the sample
X_dup = credit_dup[num_cols]                                       # features of the inflated dataset
y_dup = credit_dup["default"].values                               # targets of the inflated dataset
scores_dup = cross_val_score(clean, X_dup, y_dup, cv=cv, scoring="roc_auc")   # CV where copies span folds
print("D duplicate rows   AUC = %.4f (+-%.4f)" % (scores_dup.mean(), scores_dup.std()))   # optimistically stable

# ---------------------------------------------------------------- 6) single hold-out comparison for the report
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   # one clean split
model_only.fit(X_tr, y_tr)                                         # fit the bare model on raw features
holdout_raw = roc_auc_score(y_te, model_only.predict_proba(X_te)[:, 1])   # hold-out AUC without scaling
clean.fit(X_tr, y_tr)                                              # fit the pipeline on the same split
holdout_pipe = roc_auc_score(y_te, clean.predict_proba(X_te)[:, 1])   # hold-out AUC with scaling inside the pipeline
print("hold-out AUC: raw %.4f | pipeline %.4f" % (holdout_raw, holdout_pipe))   # scaling matters for linear models

protocols = pd.DataFrame({"protocol": ["A clean pipeline", "B selection leak", "C target leak", "D duplicate rows"],   # names
                          "cv_auc_mean": [scores_clean.mean(), scores_scaled.mean(),   # mean cross-validated AUC
                                          scores_target.mean(), scores_dup.mean()],   # for all four protocols
                          "cv_auc_std": [scores_clean.std(), scores_scaled.std(),   # spread across folds
                                         scores_target.std(), scores_dup.std()],   # for all four protocols
                          "holdout_auc": [holdout_pipe, np.nan, np.nan, np.nan]})   # one honest hold-out reference
protocols = protocols.round(4)                                     # keep four decimals for readability
dump(protocols, "l5_cv_results.csv")                               # persist the comparison table
print(protocols.to_string(index=False))                            # show the table in the console

# ---------------------------------------------------------------- 7) figure: what the mistake costs
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))                # three panels: folds, learning curve, duplicates
axes[0].boxplot([scores_clean, scores_scaled, scores_target, scores_dup],   # one box per protocol
                patch_artist=True,                                 # allow colouring the boxes
                tick_labels=["A clean", "B selection\nbefore split", "C target\nleak", "D duplicate\nrows"])   # labels
for patch, colour in zip(axes[0].artists if hasattr(axes[0], "artists") else axes[0].patches,   # colour the boxes
                         [OKABE[2], OKABE[0], OKABE[5], OKABE[1]]):   # one colour per protocol
    patch.set_facecolor(colour)                                    # apply the fill colour
    patch.set_alpha(0.6)                                           # keep it light
axes[0].axhline(0.5, color="#66717D", ls="--", lw=1.0)             # reference line: random ranking
style_axes(axes[0], "Cross-validated ROC-AUC by protocol", "", "ROC-AUC (5 folds)")   # shared style
sizes = np.linspace(0.15, 1.0, 8)                                  # growing fractions of the training data
curve_clean, curve_leak = [], []                                   # collect the learning curves
for frac in sizes:                                                 # loop over the training sizes
    n = max(60, int(len(X) * 0.8 * frac))                          # number of training rows for this step
    Xs, ys = X.iloc[:n], y[:n]                                     # take a prefix as the training sample
    curve_clean.append(cross_val_score(clean, Xs, ys, cv=3, scoring="roc_auc").mean())   # honest protocol
    Xs_leak = X_target.iloc[:n]                                    # same rows, leaked feature included
    curve_leak.append(cross_val_score(model_only, Xs_leak, ys.iloc[:n] if hasattr(ys, "iloc") else ys[:n],   # leaked protocol
                                      cv=3, scoring="roc_auc").mean())   # mean over three folds
axes[1].plot(sizes * 80, curve_clean, "-o", ms=4, color=OKABE[2], label="A clean pipeline")   # honest curve
axes[1].plot(sizes * 80, curve_leak, "-s", ms=4, color=OKABE[5], label="C target leak")   # inflated curve
style_axes(axes[1], "Leakage looks great until deployment", "Training rows used (%)", "ROC-AUC")   # shared style
axes[1].legend(fontsize=8.0, loc="lower right")                    # legend inside the panel
counts = pd.Series({"unique rows": len(credit), "after duplication": len(credit_dup)})   # row counts to compare
axes[2].bar(counts.index, counts.values, color=[OKABE[2], OKABE[5]], alpha=0.85)   # bar chart of the row counts
axes[2].set_ylim(0, len(credit_dup) * 1.15)                        # leave room for the labels
for i, v in enumerate(counts.values):                              # annotate each bar with its value
    axes[2].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9.5)   # count label above the bar
style_axes(axes[2], "Duplicate rows inflate the sample", "", "Number of rows")   # shared style
axes[2].text(0.03, 0.9, f"AUC {scores_clean.mean():.3f} -> {scores_dup.mean():.3f}",   # annotate the AUC shift
             transform=axes[2].transAxes, fontsize=8.6, color="#1E2630")   # small note inside the panel
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_cv_leakage.png")                                    # save the leakage comparison figure
