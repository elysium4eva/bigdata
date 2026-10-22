"""l10_09_rag_citation.py -- retrieval-augmented answers with citation tracing.

Every line carries an English comment. The script builds a tiny document store from the
contract snippets, retrieves the passages that answer a question, assembles an answer with
inline citations, and draws the retrieval scores so the deck can show what "grounding" means.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the figure
import jieba                                                       # Chinese segmentation
from sklearn.feature_extraction.text import TfidfVectorizer        # the retrieval representation
from sklearn.metrics.pairwise import cosine_similarity              # the similarity
from _common_l10 import (DAT, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) the document store
contracts = pd.read_csv(DAT / "l10_contracts.csv")                 # the contract snippets
docs = contracts["text"].tolist()                                  # the passages
doc_ids = contracts["doc_id"].tolist()                             # stable identifiers
print("passages:", len(docs))                                       # report
def cut(text):                                                     # tokenise one passage
    """Return the space-joined tokens of a passage."""             # short docstring
    return " ".join(t for t in jieba.lcut(str(text)) if t.strip())   # segment with jieba
corpus = [cut(d) for d in docs]                                    # the tokenised corpus
vec = TfidfVectorizer().fit(corpus)                                # fit the retrieval model
matrix = vec.transform(corpus)                                     # the document matrix

# ---------------------------------------------------------------- 2) retrieve and answer with citations
questions = ["哪家公司签署了金额为 3,450 万元的协议？",              # a question about an amount
             "2026 年 3 月 15 日签订的协议双方是谁？",               # a question about a date
             "涉及收购子公司股权的公告是哪一条？"]                   # a question about an acquisition
rows = []                                                          # one row per question
for qi, question in enumerate(questions):                          # loop over the questions
    q = vec.transform([cut(question)])                             # the query vector
    scores = cosine_similarity(q, matrix).ravel()                   # similarity to every passage
    order = np.argsort(scores)[::-1][:3]                           # the three best passages
    answer = " ".join("[%s] %s" % (doc_ids[i], docs[i]) for i in order if scores[i] > 0.05)   # answer with citations
    rows.append({"question": question, "answer": answer,           # the grounded answer
                 "top_id": doc_ids[order[0]], "top_score": round(float(scores[order[0]]), 4),   # the best hit
                 "scores": [round(float(scores[i]), 3) for i in order]})   # the scores
    print("Q%d:" % (qi + 1), question)                              # report the question
    print("  A:", answer[:150])                                    # report the answer (truncated)
grounded = pd.DataFrame(rows)                                      # assemble the table
dump(grounded, "l10_rag_answers.csv")                              # persist

# ---------------------------------------------------------------- 3) the retrieval-score figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
for ax, row, question in zip(axes, grounded.itertuples(), questions):   # one panel per question
    q = vec.transform([cut(question)])                             # the query vector
    scores = cosine_similarity(q, matrix).ravel()                   # the scores
    order = np.argsort(scores)[::-1][:8]                           # the eight best passages
    ax.barh([doc_ids[i] for i in order][::-1], scores[order][::-1], color=OKABE[4], alpha=0.9)   # the bars
    ax.axvline(0.05, ls="--", lw=1.0, color="#66717D")              # the citation threshold
    style_axes(ax, question[:16] + "…", "cosine similarity", "")    # shared style
    ax.tick_params(axis="y", labelsize=7.0)                         # small labels
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_rag.png")                                           # save the RAG figure

# ---------------------------------------------------------------- 4) a pipeline sketch
fig, axes = plt.subplots(1, 1, figsize=(9.6, 2.6))                 # one wide panel
axes.axis("off")                                                  # sketch only
steps = ["问题 query", "检索 retrieve", "拼接 context", "生成 answer", "标注 citation"]   # the steps
for i, step in enumerate(steps):                                   # draw the boxes
    axes.text(0.09 + i * 0.2, 0.5, step, ha="center", va="center", fontsize=10,   # the label
              bbox=dict(boxstyle="round,pad=0.4", fc="#F7F3F2", ec=OKABE[4], lw=1.2))   # the box
    if i < len(steps) - 1:                                         # the arrow to the next step
        axes.annotate("", xy=(0.09 + (i + 1) * 0.2 - 0.055, 0.5), xytext=(0.09 + i * 0.2 + 0.055, 0.5),   # the arrow to the next step
                      arrowprops=dict(arrowstyle="->", color=OKABE[5], lw=1.4))   # arrow style
axes.set_title("RAG: retrieve, ground, cite")                       # one title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_rag_pipeline.png")                                  # save the sketch
print("done: retrieval-augmented answers with citations")           # closing log
