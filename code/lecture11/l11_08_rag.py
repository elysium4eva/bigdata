"""l11_08_rag.py -- a complete retrieval-augmented generation pipeline with citations.

Every line carries an English comment. The script chunks a document library, builds a TF-IDF index,   # continuation of the call
retrieves for each question, answers with a template that must cite its source, and measures   # continuation of the call
retrieval hit rate against answer accuracy - the two numbers that decide whether a RAG system works.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.feature_extraction.text import TfidfVectorizer         # the retrieval index
from sklearn.metrics.pairwise import cosine_similarity              # the similarity
from _common_l11 import (make_documents, make_qa_set, tokenize,      # shared helpers
                         save, dump, style_axes, annotated, OKABE)      # continuation of the call

# ---------------------------------------------------------------- 1) documents, chunks and the index
docs = make_documents(40)                                          # the library
qa = make_qa_set(docs, 60)                                          # the questions
def chunk(text, size=24, overlap=6):                               # split a body into chunks
    """Return character chunks with a fixed overlap."""             # short docstring
    out, start = [], 0                                             # the chunks and the cursor
    while start < len(text):                                       # walk the text
        out.append(text[start:start + size])                        # take a window
        start += size - overlap                                    # step forward with overlap
    return out                                                     # the chunks
chunks = []                                                        # one row per chunk
for _, row in docs.iterrows():                                     # loop over documents
    for k, piece in enumerate(chunk(row["body"])):                  # loop over its chunks
        chunks.append({"doc_id": row["doc_id"], "chunk_id": k, "text": piece})   # the chunk row
chunk_df = pd.DataFrame(chunks)                                    # assemble
dump(chunk_df.head(200), "l11_chunks.csv")                         # persist a sample
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1)   # character n-grams handle Chinese
matrix = vectorizer.fit_transform(chunk_df["text"])                 # the chunk matrix
print("chunks:", len(chunk_df), "| vocabulary:", matrix.shape[1])    # report

# ---------------------------------------------------------------- 2) retrieve, answer, cite
def retrieve(question, k=3):                                       # top-k retrieval
    """Return the top-k chunks with their scores for a question."""   # short docstring
    q = vectorizer.transform([question])                            # the query vector
    scores = cosine_similarity(q, matrix).ravel()                    # the similarities
    top = np.argsort(scores)[::-1][:k]                              # the top indices
    return [(int(i), float(scores[i])) for i in top]                 # index and score

def answer(question, k=3, threshold=0.02):                         # retrieval + citation
    """Return (answer text, cited doc id, hit) for a question."""    # short docstring
    hits = retrieve(question, k)                                    # the retrieved chunks
    if not hits or hits[0][1] < threshold:                          # nothing relevant enough
        return "资料中没有找到相关内容。", "", False                   # refuse instead of guessing
    idx, score = hits[0]                                            # the best chunk
    row = chunk_df.iloc[idx]                                        # its metadata
    text = row["text"]                                              # the chunk text
    return "%s [%s]" % (text[:40], row["doc_id"]), row["doc_id"], True   # the cited answer

rows = []                                                          # one row per question
for _, q in qa.iterrows():                                         # loop over the questions
    text, cited, hit = answer(q["question"])                        # answer with citation
    rows.append({"q_id": q["q_id"], "gold_doc": q["gold_doc"],       # the record
                 "cited_doc": cited, "retrieval_hit": int(cited == q["gold_doc"]),   # retrieval quality
                 "answered": int(hit), "text": text})                # answer quality
out = pd.DataFrame(rows)                                           # assemble
dump(out, "l11_rag_results.csv")                                   # persist
hit_rate = float(out["retrieval_hit"].mean())                       # retrieval hit rate
answer_rate = float(out["answered"].mean())                         # answer rate
print("retrieval hit@1 %.3f | answer rate %.3f" % (hit_rate, answer_rate))   # report

# ---------------------------------------------------------------- 3) how retrieval quality drives accuracy
rows = []                                                          # one row per k
for k in [1, 2, 3, 5, 8]:                                          # increasing k
    hits = 0                                                       # reset
    for _, q in qa.iterrows():                                     # loop over the questions
        found = retrieve(q["question"], k)                          # retrieve k chunks
        docs_k = {chunk_df.iloc[i]["doc_id"] for i, _ in found}      # the documents touched
        hits += int(q["gold_doc"] in docs_k)                        # a hit if the gold doc is in the set
    rows.append({"k": k, "hit_rate": round(hits / len(qa), 3)})      # the row
curve = pd.DataFrame(rows)                                         # assemble
dump(curve, "l11_rag_curve.csv")                                   # persist
print(curve.to_string(index=False))                                # report

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].plot(curve["k"], curve["hit_rate"], "-o", ms=5, color=OKABE[5])   # hit rate vs k
axes[0].set_xticks(curve["k"])                                      # tick every k
style_axes(axes[0], "检索命中率随 k 上升，但上下文也变长", "召回条数 k", "命中率")   # shared style
annotated(axes[0], "hit@1 = %.2f" % hit_rate, "lower right")           # annotate
# the joint picture: retrieval hit vs final answer correctness
axes[1].scatter(out["retrieval_hit"] + np.random.default_rng(0).normal(0, 0.02, len(out)),   # draw the panel
                out["answered"] + np.random.default_rng(1).normal(0, 0.02, len(out)),   # continuation of the call
                s=14, alpha=0.5, color=OKABE[4])                     # jittered counts
style_axes(axes[1], "检索错了，回答不可能对", "检索是否命中（抖动）", "是否给出回答（抖动）")   # shared style
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_rag_quality.png")                                    # save the figure
print("done: chunking, TF-IDF retrieval, cited answers")              # closing log
