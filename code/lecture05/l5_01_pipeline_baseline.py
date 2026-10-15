"""l5_01_pipeline_baseline.py -- an end-to-end scikit-learn pipeline and a baseline.

Every line carries an English comment. The script builds a synthetic credit-card
default dataset, splits it into train/validation/test, fits a dummy baseline and
a first real model, and records the scores in a lightweight experiment log.
"""
import time                                                        # measure training time for the cost discussion
import numpy as np                                                 # numeric arrays for the feature construction
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # figure with the workflow and the scores
from sklearn.compose import ColumnTransformer                      # apply different preprocessing per column type
from sklearn.dummy import DummyClassifier                          # majority-class baseline
from sklearn.linear_model import LogisticRegression                # first interpretable model
from sklearn.metrics import (accuracy_score, roc_auc_score,        # scoring functions used for the log
                             confusion_matrix, classification_report)   # matrix and per-class report
from sklearn.model_selection import train_test_split               # train/validation/test splitting
from sklearn.pipeline import Pipeline                              # bind preprocessing and model together
from sklearn.preprocessing import OneHotEncoder, StandardScaler    # scaling for numbers, one-hot for categories
from _common_l5 import (RNG, save, dump, write_json,               # shared helpers from the lecture module
                        style_axes, OKABE)                         # plotting helper and the safe palette

# ---------------------------------------------------------------- 1) build a synthetic credit-default dataset
N = 3000                                                           # number of credit-card accounts
age = np.clip(RNG.normal(38, 11, N), 21, 75)                       # customer age in years
limit = np.round(np.exp(RNG.normal(10.6, 0.7, N)), -2)             # credit limit, log-normal keeps it right skewed
education = RNG.choice(["high school", "bachelor", "master", "phd"], N,   # education level, a categorical feature
                       p=[0.32, 0.46, 0.18, 0.04])                 # sampling probabilities per level
pay_status = RNG.integers(-1, 6, size=(N, 3))                      # repayment status for the last three months
bill_amt = np.round(limit[:, None] * RNG.uniform(0.05, 1.0, (N, 3)), 0)   # billed amount per month
pay_amt = np.round(bill_amt * RNG.uniform(0.0, 1.1, (N, 3)), 0)    # amount actually paid per month
risk = (-0.030 * (age - 38)                                        # older customers default slightly less often
        + 0.55 * pay_status.sum(axis=1)                            # more months in arrears raises risk sharply
        - 0.90 * (pay_amt.sum(axis=1) / (bill_amt.sum(axis=1) + 1))   # paying a larger share lowers risk
        - 0.12 * np.log(limit)                                     # larger limits lower risk a little
        + 0.8 * (education == "high school")                       # education acts as a weak proxy variable
        + RNG.normal(0, 1.0, N))                                   # idiosyncratic noise
proba = 1 / (1 + np.exp(-(risk - np.quantile(risk, 0.78))))        # shift so that roughly 22 per cent default
default = (RNG.random(N) < proba).astype(int)                      # draw the binary label from that probability
credit = pd.DataFrame({"age": age.round(0), "limit": limit,        # assemble the first two columns
                       "education": education,                     # categorical column
                       "pay_status_1": pay_status[:, 0],           # repayment status, most recent month
                       "pay_status_2": pay_status[:, 1],           # repayment status, two months ago
                       "pay_status_3": pay_status[:, 2],           # repayment status, three months ago
                       "bill_1": bill_amt[:, 0], "bill_2": bill_amt[:, 1], "bill_3": bill_amt[:, 2],   # billed amounts
                       "pay_1": pay_amt[:, 0], "pay_2": pay_amt[:, 1], "pay_3": pay_amt[:, 2],   # paid amounts
                       "default": default})                        # target label: 1 = defaulted next month
credit["pay_ratio"] = (credit[["pay_1", "pay_2", "pay_3"]].sum(axis=1)   # engineered feature: share of bill repaid
                       / (credit[["bill_1", "bill_2", "bill_3"]].sum(axis=1) + 1))   # add one to avoid a zero divisor
dump(credit, "l5_credit_default.csv")                              # persist the sample for the later scripts
print("credit rows:", len(credit), "| default rate: %.3f" % credit["default"].mean())   # report class balance

# ---------------------------------------------------------------- 2) split features and target
X = credit.drop(columns=["default"])                               # feature matrix, one row per account
y = credit["default"].values                                       # target vector
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()   # numeric feature names
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()   # categorical feature names
X_train, X_tmp, y_train, y_tmp = train_test_split(                 # first split: training versus the rest
    X, y, test_size=0.4, random_state=42, stratify=y)              # stratify keeps the default rate identical
X_val, X_test, y_val, y_test = train_test_split(                   # second split: validation versus test
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)  # both halves keep the same positive rate
print("shapes:", X_train.shape, X_val.shape, X_test.shape)         # confirm the 60/20/20 partition

# ---------------------------------------------------------------- 3) shared preprocessing inside a pipeline
prep = ColumnTransformer(                                          # route each column type to its own transformer
    transformers=[("num", StandardScaler(), num_cols),             # numeric columns are standardised
                  ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],   # categories become dummy columns
    remainder="drop")                                              # drop anything unexpected
def make_pipe(model):                                              # build a fresh pipeline for a given estimator
    """Return a pipeline that preprocesses the features and fits the model."""   # short docstring
    return Pipeline(steps=[("prep", prep), ("model", model)])      # preprocessing then estimator

# ---------------------------------------------------------------- 4) dummy baseline versus logistic regression
results = []                                                       # collect one row of metrics per model and split
fitted = {}                                                        # keep the fitted pipelines for later reuse
for name, model in [("dummy (majority)", DummyClassifier(strategy="most_frequent")),   # baseline to beat
                    ("logistic regression", LogisticRegression(max_iter=2000, random_state=42))]:   # first real model
    pipe = make_pipe(model)                                        # attach the shared preprocessing
    t0 = time.time()                                               # start the clock before fitting
    pipe.fit(X_train, y_train)                                     # train on the training split only
    train_seconds = time.time() - t0                               # measure the training cost
    fitted[name] = pipe                                            # remember the fitted pipeline
    for split, Xs, ys in [("train", X_train, y_train),             # evaluate on the training split
                          ("validation", X_val, y_val),            # evaluate on the validation split
                          ("test", X_test, y_test)]:               # evaluate once on the held-out test split
        proba = pipe.predict_proba(Xs)[:, 1]                       # predicted default probability
        pred = (proba >= 0.5).astype(int)                          # hard label at the default threshold
        results.append({"model": name, "split": split,             # model and split identifiers
                        "accuracy": round(accuracy_score(ys, pred), 4),   # share of correct decisions
                        "roc_auc": round(roc_auc_score(ys, proba), 4),    # ranking quality across thresholds
                        "flag_rate": round(pred.mean(), 4),        # share flagged as defaulters
                        "train_seconds": round(train_seconds, 3)})   # training cost
scores = pd.DataFrame(results)                                     # assemble the metrics table
dump(scores, "l5_baseline_scores.csv")                             # persist the table for the report
print(scores.to_string(index=False))                               # show the comparison in the console

# ---------------------------------------------------------------- 5) lightweight experiment log (instead of MLflow)
best = scores[(scores["model"] == "logistic regression") & (scores["split"] == "test")].iloc[0]   # test row of the model
write_json({"lecture": "L5",                                       # experiment metadata: which lecture
            "script": "l5_01_pipeline_baseline.py",                # which script produced the run
            "seed": 20260506,                                      # seed used for reproducibility
            "dataset": {"file": "l5_credit_default.csv", "rows": int(len(credit)),   # dataset description
                        "positive_rate": round(float(credit["default"].mean()), 4)},   # class balance
            "metrics": {"test_roc_auc": float(best["roc_auc"]),    # test ROC-AUC of the first real model
                        "test_accuracy": float(best["accuracy"]),  # test accuracy of the first real model
                        "train_seconds": float(best["train_seconds"])},   # training cost of that model
            "artifacts": ["l5_baseline_scores.csv"]}, "l5_run_log_01.json")   # related artefacts

# ---------------------------------------------------------------- 6) figure: workflow, scores and confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels in one row
axes[0].axis("off")                                                # the first panel is a schematic
boxes = [("Raw data\n3,000 accounts", 0.02, 0.64, OKABE[0]),       # describe the dataset node
         ("Split 60/20/20\nstratified", 0.36, 0.64, OKABE[1]),     # describe the split node
         ("Pipeline\nprep + model", 0.70, 0.64, OKABE[2]),         # describe the pipeline node
         ("Baseline\nmajority class", 0.02, 0.16, OKABE[3]),       # describe the baseline node
         ("Evaluate\ntrain / val / test", 0.36, 0.16, OKABE[4]),   # describe the evaluation node
         ("Model card\n+ run log", 0.70, 0.16, OKABE[5])]          # describe the documentation node
for label, x, y, colour in boxes:                                  # draw one rounded box per node
    axes[0].add_patch(plt.Rectangle((x, y), 0.27, 0.21,            # rectangle position and size in axes units
                                    transform=axes[0].transAxes, fc=colour, alpha=0.16,   # pale fill colour
                                    ec=colour, lw=1.2, zorder=2))  # coloured border drawn on top
    axes[0].text(x + 0.135, y + 0.105, label, transform=axes[0].transAxes,   # centre the label inside the box
                 ha="center", va="center", fontsize=8.3, zorder=3)   # readable font size
for x0, y0, x1, y1 in [(0.29, 0.745, 0.36, 0.745), (0.63, 0.745, 0.70, 0.745),   # arrows along the top row
                       (0.29, 0.265, 0.36, 0.265), (0.63, 0.265, 0.70, 0.265),   # arrows along the bottom row
                       (0.155, 0.64, 0.155, 0.37), (0.495, 0.37, 0.495, 0.64),   # arrows between the rows
                       (0.835, 0.37, 0.835, 0.64)]:                     # last vertical arrow
    axes[0].annotate("", xy=(x1, y1), xytext=(x0, y0), transform=axes[0].transAxes,   # draw the arrow head
                     arrowprops=dict(arrowstyle="->", color="#66717D", lw=1.1))   # thin grey arrow
axes[0].set_title("Typical supervised workflow")                   # title of the schematic panel
plot_scores = scores.pivot(index="model", columns="split", values="roc_auc")   # reshape metrics for plotting
plot_scores = plot_scores[["train", "validation", "test"]]         # keep a stable column order
plot_scores.plot(kind="bar", ax=axes[1], rot=10, width=0.78,       # grouped bar chart per model and split
                 color=[OKABE[0], OKABE[1], OKABE[2]])             # one colour per split
axes[1].set_ylim(0.4, 1.02)                                        # zoom on the informative range
style_axes(axes[1], "ROC-AUC by split: beat a baseline first", "", "ROC-AUC")   # shared style
axes[1].legend(fontsize=7.6, loc="lower right")                    # compact legend inside the panel
cm = confusion_matrix(y_test, fitted["logistic regression"].predict(X_test))   # confusion matrix on the test split
axes[2].imshow(cm, cmap="RdPu")                                    # render the matrix as a heatmap
axes[2].set_xticks([0, 1], ["pred no", "pred yes"], fontsize=8.6)   # x axis: predicted labels
axes[2].set_yticks([0, 1], ["true no", "true yes"], fontsize=8.6)   # y axis: actual labels
axes[2].grid(False)                                                # a heatmap needs no grid lines
for i in range(2):                                                 # annotate each row of the matrix
    for j in range(2):                                             # loop over the two columns
        axes[2].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10,   # print the cell count
                     color="white" if cm[i, j] > cm.max() * 0.6 else "#1E2630")   # contrast against the fill
axes[2].set_title("Test confusion matrix (threshold 0.5)")         # title of the third panel
plt.tight_layout()                                                 # tidy the layout before saving
save(fig, "fig_ml_pipeline.png")                                   # save the workflow and score figure
print(classification_report(y_test, fitted["logistic regression"].predict(X_test),   # per-class metrics
                            target_names=["no default", "default"], digits=3))   # readable class names
