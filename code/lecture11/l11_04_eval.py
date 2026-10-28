"""l11_04_eval.py -- a minimal evaluation harness for a chat model.

Every line carries an English comment. The script scores the local model on three metrics that   # continuation of the call
matter in practice - answer accuracy, refusal rate and format compliance - and shows why a single   # continuation of the call
"accuracy" number hides the failures that actually break a product.     # continuation of the call
"""
import json                                                        # the JSON schema check
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l11 import (get_model, save, dump, style_axes,          # shared helpers
                         annotated, OKABE, RNG)                         # continuation of the call

model = get_model()                                                # the local chat model

# ---------------------------------------------------------------- 1) a small labelled test set
cases = [                                                          # (question, key facts the answer must contain)
    ("跨境电商的物流时效影响什么？", ["复购率"]),                        # logistics
    ("模型评测要区分哪些指标？", ["准确率", "拒答", "格式"]),             # evaluation
    ("检索增强生成有什么好处？", ["幻觉", "引用"]),                      # RAG
    ("提示工程的核心是什么？", ["写清楚", "格式"]),                       # prompting
    ("数据质量决定了什么？", ["可信度"]),                               # data quality
    ("小样本场景下哪种提示更好？", ["few-shot"]),                        # few-shot
]                                                                  # the test cases

def score_answer(answer, keys):                                    # keyword recall
    """Return the share of required key facts present in the answer."""   # short docstring
    if not keys:                                                   # nothing to check
        return 0.0                                                 # avoid a division by zero
    return float(sum(1 for k in keys if k in answer) / len(keys))    # the recall

rows = []                                                          # one row per case
for question, keys in cases:                                       # loop over the cases
    reply = model.chat([{"role": "user", "content": question}], temperature=0.3,   # continuation of the call
                       max_tokens=16)["choices"][0]["message"]["content"]   # ask the local model
    recall = score_answer(reply, keys)                              # the key-fact recall
    rows.append({"question": question, "answer": reply,             # the record
                 "recall": round(recall, 2),                        # the score
                 "refused": int(len(reply.strip()) < 2),            # a trivial refusal proxy
                 "json_ok": int(reply.strip().startswith("{"))})     # a trivial format proxy
results = pd.DataFrame(rows)                                        # assemble
dump(results, "l11_eval.csv")                                       # persist
accuracy = float(results["recall"].mean())                          # mean key-fact recall
refusal = float(results["refused"].mean())                          # refusal rate
format_ok = float(results["json_ok"].mean())                        # format compliance
print("accuracy %.3f | refusal %.3f | format %.3f" % (accuracy, refusal, format_ok))   # report

# ---------------------------------------------------------------- 2) why one number is not enough
# Two hypothetical models with the same accuracy but very different failure profiles.
profiles = pd.DataFrame([                                               # assemble the frame
    {"model": "模型 A", "accuracy": 0.72, "refusal": 0.02, "format": 0.90, "hallucination": 0.18},   # eager
    {"model": "模型 B", "accuracy": 0.72, "refusal": 0.31, "format": 0.98, "hallucination": 0.06},   # cautious
])                                                                 # the comparison
dump(profiles, "l11_eval_profiles.csv")                            # persist
print(profiles.to_string(index=False))                             # report
# the decision rule: a cautious model is better for a compliance-heavy product
print("合规场景选 B：拒答可以接受，编造不行")                          # the takeaway

# ---------------------------------------------------------------- 3) the figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
metrics = ["准确率", "拒答率", "格式合规", "幻觉率"]                     # the four metrics
a_vals = [0.72, 0.02, 0.90, 0.18]                                   # model A
b_vals = [0.72, 0.31, 0.98, 0.06]                                   # model B
x = np.arange(len(metrics))                                         # positions
axes[0].bar(x - 0.19, a_vals, width=0.36, color=OKABE[1], alpha=0.92, label="模型 A（爱说）")   # A
axes[0].bar(x + 0.19, b_vals, width=0.36, color=OKABE[2], alpha=0.92, label="模型 B（谨慎）")   # B
axes[0].set_xticks(x, metrics)                                       # labels
axes[0].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[0], "同样的准确率，完全不同的风险画像", "", "比率")       # shared style
axes[1].bar(results["recall"].index, results["recall"], color=OKABE[4], alpha=0.9)   # per-case recall
axes[1].set_xticks(range(len(results)), ["Q%d" % (i + 1) for i in range(len(results))])   # labels
style_axes(axes[1], "逐题得分：平均值掩盖了短板", "题目", "关键事实召回")   # shared style
annotated(axes[1], "平均 %.2f" % accuracy, "lower right")             # annotate
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_eval.png")                                           # save the figure
print("done: a minimal evaluation harness")                           # closing log
