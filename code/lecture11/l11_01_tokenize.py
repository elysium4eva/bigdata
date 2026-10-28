"""l11_01_tokenize.py -- tokenisation, the GPT lineage and the scaling laws.

Every line carries an English comment. The script counts tokens for Chinese and English text,   # continuation of the call
draws the GPT family timeline and the scaling-law fit, and persists the token statistics that the   # continuation of the call
deck quotes.                                                            # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l11 import (get_model, tokenize, save, dump,           # shared helpers
                         style_axes, annotated, OKABE)                  # continuation of the call

# ---------------------------------------------------------------- 1) tokenisation
model = get_model()                                                # the local chat model
zh = "跨境电商的物流时效是影响复购率的关键因素之一"                    # a Chinese sentence
en = "Cross-border logistics speed affects repurchase rate."        # an English sentence
zh_tokens = tokenize(zh)                                           # Chinese tokenises to characters
en_tokens = tokenize(en)                                           # English tokenises to words
print("Chinese: %d chars -> %d tokens" % (len(zh), len(zh_tokens)))   # report
print("English: %d chars -> %d tokens" % (len(en), len(en_tokens)))   # report
rows = [{"language": "中文", "chars": len(zh), "tokens": len(zh_tokens),   # continuation of the call
         "chars_per_token": round(len(zh) / len(zh_tokens), 2)},     # the Chinese row
        {"language": "English", "chars": len(en), "tokens": len(en_tokens),   # continuation of the call
         "chars_per_token": round(len(en) / len(en_tokens), 2)}]     # the English row
counts = pd.DataFrame(rows)                                        # assemble
dump(counts, "l11_token_counts.csv")                               # persist
print(counts.to_string(index=False))                               # report
# a cost estimate: token counts decide the bill
price_per_1k = 0.002                                               # yuan per thousand tokens (illustrative)
daily_calls, tokens_per_call = 20000, 800                          # the assumed workload
monthly = daily_calls * tokens_per_call * 30 / 1000 * price_per_1k   # the monthly cost
print("20k calls/day x 800 tokens -> %.1f yuan per month" % monthly)   # report

# ---------------------------------------------------------------- 2) the GPT lineage
lineage = pd.DataFrame([                                                # assemble the frame
    {"model": "GPT-1", "year": 2018, "params_b": 0.117, "context_k": 0.5},   # the first GPT
    {"model": "GPT-2", "year": 2019, "params_b": 1.5, "context_k": 1.0},     # zero-shot beginnings
    {"model": "GPT-3", "year": 2020, "params_b": 175.0, "context_k": 2.0},   # in-context learning
    {"model": "GPT-3.5", "year": 2022, "params_b": 175.0, "context_k": 4.0}, # the chat product
    {"model": "GPT-4", "year": 2023, "params_b": 1800.0, "context_k": 32.0}, # multimodal rumours
    {"model": "推理模型", "year": 2025, "params_b": 1800.0, "context_k": 128.0},   # test-time compute
])                                                                  # the table
dump(lineage, "l11_gpt_lineage.csv")                               # persist
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].bar(lineage["model"], lineage["params_b"], color=OKABE[4], alpha=0.9)   # parameters
axes[0].set_yscale("log")                                         # three orders of magnitude
axes[0].set_xticks(range(len(lineage)), lineage["model"], rotation=20, ha="right")   # labels
style_axes(axes[0], "参数量（对数轴）：三次跃迁", "", "参数量（十亿）")     # shared style
axes[1].plot(lineage["year"], lineage["context_k"], "-o", ms=6, color=OKABE[5])   # context window
axes[1].set_yscale("log")                                         # log scale reads better
style_axes(axes[1], "上下文窗口：从 0.5K 到 128K", "年份", "上下文（千 token，对数）")   # shared style
annotated(axes[1], "窗口是它唯一的记忆", "upper left")                  # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_gpt_lineage.png")                                   # save the figure

# ---------------------------------------------------------------- 3) scaling laws
# A stylised fit: loss falls as a power law in parameters, data and compute.
params = np.logspace(8, 11.5, 40)                                  # 0.1B to 30B parameters
loss = 1.6 + 3.4 * (params / 1e8) ** -0.075                         # the power-law fit
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].plot(params / 1e9, loss, lw=2.0, color=OKABE[4])            # parameters axis
axes[0].set_xscale("log")                                         # log x
style_axes(axes[0], "缩放定律：参数越多，loss 越低（但边际递减）", "参数量（十亿，对数）", "验证集 loss")   # shared style
annotated(axes[0], "幂律 · 斜率约 −0.075", "upper right")              # annotate
# the compute-optimal split between parameters and data
budget = np.logspace(19, 25, 30)                                    # FLOPs budgets
optimal_params = (budget / 6.0) ** 0.5 / 1e9                         # a Chinchilla-style rule
axes[1].plot(budget, optimal_params, lw=2.0, color=OKABE[2])         # the allocation
axes[1].set_xscale("log"); axes[1].set_yscale("log")                 # log axes
style_axes(axes[1], "算力预算决定「参数与数据怎么分」", "训练算力（FLOPs，对数）", "最优参数量（十亿，对数）")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_scaling.png")                                       # save the figure
dump(pd.DataFrame({"params_b": (params / 1e9).round(3), "loss": loss.round(4)}), "l11_scaling.csv")   # persist
print("done: tokenisation, lineage and scaling laws")                 # closing log
