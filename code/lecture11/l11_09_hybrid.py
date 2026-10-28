"""l11_09_hybrid.py -- keyword versus vector retrieval, cost and model choice.

Every line carries an English comment. The script compares a keyword scorer with the TF-IDF vector   # continuation of the call
scorer on question types where they disagree, blends the two, and draws the cost-latency frontier   # continuation of the call
and the model-choice matrix that the deck uses to close the engineering part.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.feature_extraction.text import TfidfVectorizer         # semantic-ish scoring
from sklearn.metrics.pairwise import cosine_similarity              # similarity
from _common_l11 import (make_documents, make_qa_set, tokenize,      # shared helpers
                         save, dump, style_axes, annotated, OKABE)      # continuation of the call

docs = make_documents(40)                                          # the library
qa = make_qa_set(docs, 60)                                          # the questions
texts = docs["body"].tolist()                                      # the searchable text
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2)).fit(texts)   # the index
matrix = vectorizer.transform(texts)                               # the document matrix

# ---------------------------------------------------------------- 1) two scorers
def keyword_score(question, doc_text):                             # exact-match scoring
    """Return the share of question tokens that appear in the document."""   # short docstring
    qs = set(tokenize(question))                                    # the question tokens
    ds = set(tokenize(doc_text))                                    # the document tokens
    return float(len(qs & ds) / max(len(qs), 1))                     # the overlap ratio

def vector_score(question, i):                                     # semantic scoring
    """Return the cosine similarity between the question and document i."""   # short docstring
    q = vectorizer.transform([question])                            # the query vector
    return float(cosine_similarity(q, matrix[i])[0, 0])              # the similarity

rows = []                                                          # one row per question
for _, q in qa.iterrows():                                         # loop over the questions
    k_scores = [keyword_score(q["question"], t) for t in texts]      # keyword scores
    v_scores = [vector_score(q["question"], i) for i in range(len(texts))]   # vector scores
    k_top = int(np.argmax(k_scores))                                # keyword winner
    v_top = int(np.argmax(v_scores))                                # vector winner
    blend = [0.4 * a + 0.6 * b for a, b in zip(k_scores, v_scores)]   # the hybrid score
    h_top = int(np.argmax(blend))                                   # hybrid winner
    rows.append({"q_id": q["q_id"], "topic": q["topic"],            # the record
                 "kw_hit": int(docs["doc_id"].iloc[k_top] == q["gold_doc"]),   # keyword hit
                 "vec_hit": int(docs["doc_id"].iloc[v_top] == q["gold_doc"]),   # vector hit
                 "hyb_hit": int(docs["doc_id"].iloc[h_top] == q["gold_doc"])})   # hybrid hit
res = pd.DataFrame(rows)                                           # assemble
dump(res, "l11_retrieval_compare.csv")                             # persist
summary = pd.DataFrame([{"scorer": "关键词", "hit_rate": round(float(res["kw_hit"].mean()), 3)},   # assemble the frame
                        {"scorer": "向量 TF-IDF", "hit_rate": round(float(res["vec_hit"].mean()), 3)},   # continuation of the call
                        {"scorer": "混合 0.4/0.6", "hit_rate": round(float(res["hyb_hit"].mean()), 3)}])   # the summary
dump(summary, "l11_retrieval_summary.csv")                         # persist
print(summary.to_string(index=False))                              # report

# ---------------------------------------------------------------- 2) cost and latency
models = pd.DataFrame([                                                 # assemble the frame
    {"model": "小模型", "cost_per_1k": 0.0005, "latency_ms": 180, "quality": 0.62},   # cheap and fast
    {"model": "中模型", "cost_per_1k": 0.0020, "latency_ms": 420, "quality": 0.78},   # balanced
    {"model": "大模型", "cost_per_1k": 0.0100, "latency_ms": 900, "quality": 0.88},   # slow and dear
    {"model": "推理模型", "cost_per_1k": 0.0300, "latency_ms": 3200, "quality": 0.93},   # thinks first
])                                                                 # the four options
models["cost_per_1k_calls"] = (models["cost_per_1k"] * 0.8).round(3)   # 800 tokens per call
dump(models, "l11_cost.csv")                                       # persist
print(models.to_string(index=False))                               # report

# ---------------------------------------------------------------- 3) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
labels = ["关键词", "向量", "混合"]                                   # the three scorers
vals = [float(res["kw_hit"].mean()), float(res["vec_hit"].mean()), float(res["hyb_hit"].mean())]   # hit rates
axes[0].bar(labels, vals, color=[OKABE[1], OKABE[2], OKABE[5]], alpha=0.92)   # the comparison
for i, v in enumerate(vals):                                       # annotate
    axes[0].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=9)   # the value
style_axes(axes[0], "两种检索各有胜负，混起来更稳", "", "hit@1")           # shared style
axes[1].scatter(models["latency_ms"], models["quality"], s=90, color=OKABE[4], zorder=3)   # the frontier
for _, r in models.iterrows():                                     # label each point
    axes[1].annotate(r["model"], (r["latency_ms"], r["quality"]),       # draw the panel
                     textcoords="offset points", xytext=(8, 4), fontsize=8.4)   # the label
axes[1].set_xscale("log")                                         # latency spans an order of magnitude
style_axes(axes[1], "延迟与效果：没有免费的午餐", "首字延迟（ms，对数）", "质量分")   # shared style
annotated(axes[1], "选型 = 在约束下找最便宜的够用", "upper left")          # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_retrieval.png")                                     # save the figure

fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].bar(models["model"], models["cost_per_1k_calls"], color=OKABE[5], alpha=0.9)   # cost
axes[0].set_xticks(range(len(models)), models["model"], rotation=15, ha="right")   # labels
style_axes(axes[0], "每千次调用的成本（元）", "", "元")                   # shared style
for i, v in enumerate(models["cost_per_1k_calls"]):                 # annotate
    axes[0].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=8.4)   # the value
axes[1].axis("off")                                                # the matrix is text
matrix_rows = [("结构化抽取", "小模型 + schema 校验"),                 # a task-to-model mapping
               ("客服问答", "中模型 + RAG + 护栏"),                              # continuation of the call
               ("长文摘要", "大模型 + 分块"),                                    # continuation of the call
               ("数学/代码", "推理模型 + 工具"),                                  # continuation of the call
               ("离线批处理", "小模型 + 蒸馏")]                       # the five rows
for i, (task, choice) in enumerate(matrix_rows):                    # draw the rows
    y = 0.86 - i * 0.17                                            # vertical position
    axes[1].text(0.02, y, task, fontsize=9.2, fontweight="bold")     # the task
    axes[1].text(0.42, y, "→ " + choice, fontsize=9.0, color=OKABE[4])   # the choice
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)                       # the sketch bounds
axes[1].set_title("选型矩阵：任务决定模型，而不是反过来")                 # the title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_cost.png")                                          # save the figure
dump(models, "l11_model_choice.csv")                               # persist the matrix source
print("done: hybrid retrieval, cost and model choice")                 # closing log
