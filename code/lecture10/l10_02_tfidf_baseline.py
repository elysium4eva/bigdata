"""l10_02_tfidf_baseline.py -- bag of words, TF-IDF and the classification baseline.

Every line carries an English comment. The script builds word-frequency and TF-IDF
matrices from the bilingual review corpus, compares them, trains a logistic-regression
baseline, and draws the vocabulary sizes and the confusion matrix.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the figures
import seaborn as sns                                             # heatmap
import jieba                                                       # Chinese segmentation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer   # the two representations
from sklearn.linear_model import LogisticRegression               # the baseline classifier
from sklearn.model_selection import train_test_split               # the split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix   # the metrics
from _common_l10 import (DAT, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) the corpus
reviews = pd.read_csv(DAT / "l10_reviews.csv")                     # the bilingual reviews
labels = reviews["label"].values                                   # sentiment labels
print("reviews:", len(reviews), "| labels:", pd.Series(labels).value_counts().to_dict())   # report

def cut(text):                                                     # tokenise for the vectoriser
    """Return the space-joined tokens of a review (jieba for Chinese, split for English)."""   # short docstring
    if any("\u4e00" <= ch <= "\u9fff" for ch in str(text)):         # contains Chinese characters
        return " ".join(t for t in jieba.lcut(str(text)) if t.strip())   # segment with jieba
    return " ".join(str(text).lower().split())                     # otherwise split on whitespace

tokens = reviews["text"].map(cut)                                  # pre-tokenised text
X_tr, X_te, y_tr, y_te = train_test_split(tokens, labels, test_size=0.3, random_state=42, stratify=labels)   # split

# ---------------------------------------------------------------- 2) bag of words versus TF-IDF
rows = []                                                          # one row per representation
matrices = {}                                                      # keep the matrices for the figure
for name, vec in [("词袋 BoW", CountVectorizer(max_features=2000)),   # raw counts
                  ("TF-IDF", TfidfVectorizer(max_features=2000))]:   # tf-idf weights
    vec.fit(X_tr)                                                  # fit on the training split only
    Atr = vec.transform(X_tr)                                      # training matrix
    Ate = vec.transform(X_te)                                      # test matrix
    model = LogisticRegression(max_iter=2000).fit(Atr, y_tr)        # the baseline classifier
    pred = model.predict(Ate)                                      # test predictions
    acc = float(accuracy_score(y_te, pred))                        # accuracy
    f1 = float(f1_score(y_te, pred, average="macro"))               # macro F1
    rows.append({"representation": name, "vocab": len(vec.vocabulary_),   # the row
                 "nnz_per_doc": round(float(Atr.nnz / Atr.shape[0]), 1),   # sparsity
                 "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})    # the scores
    matrices[name] = (vec, model, Ate, pred)                        # remember the artefacts
    print("%-10s | vocab %4d | acc %.3f | F1 %.3f" % (name, len(vec.vocabulary_), acc, f1))   # report
table = pd.DataFrame(rows)                                         # assemble the table
dump(table, "l10_baseline_scores.csv")                             # persist

# ---------------------------------------------------------------- 3) top features by weight
vec, model, Ate, pred = matrices["TF-IDF"]                         # use the TF-IDF model
names = np.array(vec.get_feature_names_out())                       # the vocabulary
coefs = model.coef_                                                # one row per class
top_rows = []                                                      # the strongest features per class
for ci, cls in enumerate(model.classes_):                          # loop over the classes
    order = np.argsort(coefs[ci])[::-1][:8]                         # the eight most positive weights
    top_rows.append({"class": cls, "top_features": " / ".join(names[order].tolist())})   # record them
    print("%s: %s" % (cls, " / ".join(names[order].tolist())))      # report
dump(pd.DataFrame(top_rows), "l10_top_features.csv")               # persist

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
axes[0].bar(table["representation"], table["vocab"], color=OKABE[2], alpha=0.9)   # vocabulary size
style_axes(axes[0], "Vocabulary size (capped at 2000)", "", "terms")   # shared style
axes[1].bar(table["representation"], table["accuracy"], color=OKABE[5], alpha=0.9)   # accuracy
axes[1].set_ylim(0.0, 1.02)                                        # full scale
style_axes(axes[1], "Accuracy of the two representations", "", "accuracy")   # shared style
for i, row in table.iterrows():                                    # annotate the bars
    axes[1].text(i, row["accuracy"], "%.3f" % row["accuracy"], ha="center", va="bottom", fontsize=9)   # value
cm = confusion_matrix(y_te, pred)                                   # confusion matrix of the TF-IDF model
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", cbar=False, ax=axes[2],   # the heatmap of the confusion matrix
            xticklabels=model.classes_, yticklabels=model.classes_)   # the heatmap
axes[2].set_title("Confusion matrix (TF-IDF baseline)")             # panel title
axes[2].tick_params(labelsize=8)                                    # small labels
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_bow_tfidf.png")                                     # save the representation figure

# ---------------------------------------------------------------- 5) the three paradigms figure
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6))                # two panels
paradigms = ["规则/词典", "TF-IDF + LR", "预训练模型 + 微调"]         # three paradigms
cost = [0.2, 1.0, 6.0]                                             # relative cost (schematic)
quality = [0.62, 0.78, 0.88]                                       # relative quality (schematic)
axes[0].scatter(cost, quality, s=180, color=[OKABE[1], OKABE[2], OKABE[5]])   # the three points
for i, name in enumerate(paradigms):                               # label the points
    axes[0].annotate(name, (cost[i], quality[i]), xytext=(6, 6), textcoords="offset points", fontsize=8)   # label
style_axes(axes[0], "Cost versus quality (schematic)", "relative cost", "relative quality")   # shared style
axes[1].bar(paradigms, cost, color=[OKABE[1], OKABE[2], OKABE[5]], alpha=0.9)   # cost bars
axes[1].tick_params(axis="x", labelsize=7.2)                        # small labels
style_axes(axes[1], "The same task, three costs", "", "relative cost")   # shared style
annotated(axes[1], "start cheap, escalate", "upper left")           # annotate the strategy
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_paradigms.png")                                     # save the paradigm figure
print("done: BoW versus TF-IDF and the baseline")                    # closing log
