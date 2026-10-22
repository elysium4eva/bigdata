"""l10_05_sentiment.py -- sentiment classification with two representations.

Every line carries an English comment. The script compares a TF-IDF logistic-regression
baseline with a word-vector-average plus MLP model, reports accuracy/F1, and draws the
confusion matrix and the confidence calibration so the deck can discuss what "better" means.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the figures
import seaborn as sns                                             # heatmap
import jieba                                                       # Chinese segmentation
from gensim.models import Word2Vec                                 # word vectors
from sklearn.feature_extraction.text import TfidfVectorizer        # the sparse representation
from sklearn.linear_model import LogisticRegression               # the linear model
from sklearn.neural_network import MLPClassifier                  # the non-linear model
from sklearn.model_selection import train_test_split               # the split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix   # the metrics
from _common_l10 import (DAT, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) the corpus
reviews = pd.read_csv(DAT / "l10_reviews.csv")                     # the bilingual reviews
def cut(text):                                                     # tokenise one document
    """Return a list of tokens for a review."""                     # short docstring
    if any("\u4e00" <= ch <= "\u9fff" for ch in str(text)):         # Chinese text
        return [t for t in jieba.lcut(str(text)) if t.strip()]      # segment with jieba
    return str(text).lower().split()                               # otherwise split on whitespace

tokens = reviews["text"].map(cut)                                  # token lists
labels = reviews["label"].values                                   # the three classes
X_tr, X_te, y_tr, y_te = train_test_split(tokens, labels, test_size=0.3, random_state=42, stratify=labels)   # split
print("train/test:", len(X_tr), len(X_te))                          # report

# ---------------------------------------------------------------- 2) representation A: TF-IDF
vec = TfidfVectorizer(analyzer=lambda toks: toks, max_features=2000)   # tokens are already lists
Atr = vec.fit_transform(X_tr)                                      # training matrix
Ate = vec.transform(X_te)                                          # test matrix
lr = LogisticRegression(max_iter=2000).fit(Atr, y_tr)               # the baseline
pred_lr = lr.predict(Ate)                                          # test predictions

# ---------------------------------------------------------------- 3) representation B: averaged word vectors + MLP
w2v = Word2Vec(sentences=list(X_tr), vector_size=40, window=4, min_count=1, sg=1, epochs=30, seed=20261111, workers=1)   # train on the training split only
def average_vector(toks):                                          # average the word vectors of a document
    """Return the mean word vector of a token list (zeros when no token is known)."""   # short docstring
    vs = [w2v.wv[t] for t in toks if t in w2v.wv]                   # the known vectors
    return np.mean(vs, axis=0) if vs else np.zeros(w2v.vector_size)   # the average or zeros
Btr = np.vstack([average_vector(t) for t in X_tr])                  # training matrix
Bte = np.vstack([average_vector(t) for t in X_te])                  # test matrix
mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=42).fit(Btr, y_tr)   # the MLP
pred_mlp = mlp.predict(Bte)                                        # test predictions
pred_mlp_proba = mlp.predict_proba(Bte)                             # probabilities for the calibration panel

# ---------------------------------------------------------------- 4) the comparison table
rows = []                                                          # one row per representation
for name, pred in [("TF-IDF + LR", pred_lr), ("w2v 平均 + MLP", pred_mlp)]:   # the two models
    rows.append({"model": name, "accuracy": round(float(accuracy_score(y_te, pred)), 4),   # accuracy
                 "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4)})   # macro F1
    print("%-14s | acc %.3f | F1 %.3f" % (name, accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")))   # report
table = pd.DataFrame(rows)                                         # assemble the table
dump(table, "l10_sentiment_scores.csv")                            # persist

# ---------------------------------------------------------------- 5) error samples
errors = reviews.iloc[np.where(np.zeros(0) == 0)[0]] if False else None   # placeholder removed below
test_index = np.arange(len(reviews))[np.isin(reviews["text"], X_te)] if False else None   # placeholder removed below
wrong = [(t, y, p) for t, y, p in zip(X_te, y_te, pred_lr) if y != p][:5]   # five misclassified examples
dump(pd.DataFrame({"text": [" ".join(t) for t, _, _ in wrong], "true": [y for _, y, _ in wrong],   # the error table
                   "pred": [p for _, _, p in wrong]}), "l10_sentiment_errors.csv")   # persist
print("misclassified examples:", len(wrong))                        # report

# ---------------------------------------------------------------- 6) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
axes[0].bar(table["model"], table["accuracy"], color=[OKABE[1], OKABE[2]], alpha=0.9)   # accuracy bars
axes[0].set_ylim(0.0, 1.02)                                        # full scale
style_axes(axes[0], "Accuracy of the two representations", "", "accuracy")   # shared style
for i, row in table.iterrows():                                    # annotate the bars
    axes[0].text(i, row["accuracy"], "%.3f" % row["accuracy"], ha="center", va="bottom", fontsize=9)   # value
cm = confusion_matrix(y_te, pred_lr)                               # confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", cbar=False, ax=axes[1],   # the heatmap of the confusion matrix
            xticklabels=lr.classes_, yticklabels=lr.classes_)      # heatmap
axes[1].set_title("Confusion matrix (TF-IDF)")                     # panel title
axes[1].tick_params(labelsize=7.6)                                 # small labels
confidence = pred_mlp_proba.max(axis=1)                            # confidence of the MLP
correct = pred_mlp == y_te                                         # whether the prediction is right
axes[2].hist(confidence[correct], bins=12, alpha=0.7, color=OKABE[2], label="correct")   # correct confidences
axes[2].hist(confidence[~correct], bins=12, alpha=0.7, color=OKABE[5], label="wrong")   # wrong confidences
style_axes(axes[2], "Confidence: right versus wrong", "max probability", "count")   # shared style
axes[2].legend(fontsize=7.4)                                       # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_sentiment.png")                                     # save the sentiment figure
print("done: two sentiment representations compared")               # closing log
