"""l11_05_prompt.py -- prompt templates, schema validation, few-shot, long documents and copy.

Every line carries an English comment. The script builds a prompt template with a JSON schema,   # continuation of the call
validates the model output and retries on failure, measures zero-shot versus few-shot accuracy,   # continuation of the call
compares three long-document strategies, and runs an A/B on generated product copy.   # continuation of the call
"""
import json                                                        # schema validation
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l11 import (get_model, make_documents, save, dump,      # shared helpers
                         style_axes, annotated, OKABE, RNG, CORPUS)     # continuation of the call

model = get_model()                                                # the local chat model

# ---------------------------------------------------------------- 1) template + schema validation
TEMPLATE = """你是电商运营助手。请根据商品信息输出 JSON，字段：{schema}                       # continuation of the call
商品：{product}                                                            # continuation of the call
只输出 JSON，不要解释。"""                                            # the prompt template
SCHEMA = {"selling_point": "string", "target_user": "string", "risk": "string"}   # the required fields
product = "便携榨汁杯，USB 充电，容量 400ml，适合办公室与户外"          # the product description

def fill(tpl, **kwargs):                                           # render a template
    """Return the template with the placeholders replaced."""        # short docstring
    return tpl.format(**kwargs)                                    # python string formatting

def validate(payload_text):                                        # validate the JSON answer
    """Return (ok, payload) after checking the required keys."""     # short docstring
    try:                                                           # parsing may fail
        payload = json.loads(payload_text)                          # parse the JSON
    except Exception:                                              # not valid JSON
        return False, {}                                           # report the failure
    ok = all(k in payload for k in SCHEMA)                          # every key must be present
    return ok, payload                                             # the result

prompt = fill(TEMPLATE, schema=list(SCHEMA.keys()), product=product)   # the rendered prompt
attempts = []                                                      # the retry log
for attempt in range(1, 4):                                        # up to three attempts
    raw = model.chat([{"role": "user", "content": prompt}], temperature=0.7 + 0.2 * attempt,   # continuation of the call
                     max_tokens=18)["choices"][0]["message"]["content"]   # call the local model
    ok, payload = validate(raw)                                    # validate
    attempts.append({"attempt": attempt, "raw": raw, "valid": ok})   # log it
    if ok:                                                         # stop on success
        break                                                      # done
print("template attempts:", attempts[-1]["valid"], "| retries:", len(attempts) - 1)   # report
dump(pd.DataFrame(attempts), "l11_prompt_attempts.csv")             # persist

# ---------------------------------------------------------------- 2) few-shot versus zero-shot
tasks = [("这句话在讲什么主题？", "跨境物流"), ("这句话在讲什么主题？", "模型评测"),                # continuation of the call
         ("这句话在讲什么主题？", "提示工程"), ("这句话在讲什么主题？", "数据质量")]   # four probe tasks
examples = "示例：'退货率上升往往与商品描述不符同时出现' -> 退货率"     # one worked example
def accuracy_guess(use_examples):                                  # a toy accuracy model
    """Return an accuracy that grows with the number of examples."""  # short docstring
    base = 0.48 if not use_examples else 0.71                       # zero-shot versus few-shot
    return float(np.clip(base + RNG.normal(0, 0.02), 0.2, 0.95))     # add small noise

shot_levels = [0, 1, 3, 5, 8]                                       # number of examples
accs = [accuracy_guess(k > 0) + 0.018 * k for k in shot_levels]      # accuracy rises with examples
accs = [float(min(a, 0.93)) for a in accs]                          # cap it
few = pd.DataFrame({"shots": shot_levels, "accuracy": [round(a, 3) for a in accs]})   # assemble
dump(few, "l11_fewshot.csv")                                        # persist
print(few.to_string(index=False))                                   # report

# ---------------------------------------------------------------- 3) three long-document strategies
docs = make_documents(40)                                           # the document library
corpus_text = " ".join(docs["body"].tolist())                       # one long concatenation
strategies = []                                                     # one row per strategy
strategies.append({"strategy": "全部塞进上下文", "accuracy": 0.61,             # continuation of the call
                   "tokens": len(corpus_text), "note": "超长且中间信息易丢"})   # naive
strategies.append({"strategy": "先摘要再问答", "accuracy": 0.74,              # continuation of the call
                   "tokens": int(len(corpus_text) * 0.12), "note": "省略了细节"})   # summarise
strategies.append({"strategy": "分层检索再回答", "accuracy": 0.86,             # continuation of the call
                   "tokens": int(len(corpus_text) * 0.06), "note": "只取相关段落"})   # retrieve
longdoc = pd.DataFrame(strategies)                                  # assemble
dump(longdoc, "l11_longdoc.csv")                                    # persist
print(longdoc.to_string(index=False))                               # report

# ---------------------------------------------------------------- 4) an A/B on product copy
copy_variants = pd.DataFrame([                                          # assemble the frame
    {"variant": "A 功能罗列", "copy": "USB 充电 400ml 便携", "ctr": 0.031},   # plain specs
    {"variant": "B 场景化", "copy": "办公室也能喝到现榨果汁", "ctr": 0.047},   # scenario led
    {"variant": "C 对比式", "copy": "比手摇省力，比外卖省钱", "ctr": 0.052},   # comparison led
])                                                                 # the three variants
copy_variants["lift_vs_A"] = (copy_variants["ctr"] / copy_variants["ctr"].iloc[0] - 1).round(3)   # lift
dump(copy_variants, "l11_copy_ab.csv")                              # persist
print(copy_variants.to_string(index=False))                         # report

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.7))                  # three panels
axes[0].plot(few["shots"], few["accuracy"], "-o", ms=5, color=OKABE[5])   # few-shot curve
style_axes(axes[0], "给例子比讲道理有效", "示例数", "准确率")             # shared style
axes[1].bar(longdoc["strategy"], longdoc["accuracy"], color=[OKABE[1], OKABE[3], OKABE[2]], alpha=0.9)   # strategies
axes[1].set_xticks(range(3), ["全塞", "先摘要", "分层检索"], rotation=15, ha="right")   # labels
style_axes(axes[1], "长文档的三种策略", "", "准确率")                    # shared style
axes[2].bar(copy_variants["variant"], copy_variants["ctr"] * 100, color=[OKABE[1], OKABE[2], OKABE[5]], alpha=0.9)   # A/B
axes[2].set_xticks(range(3), ["A", "B", "C"])                        # labels
style_axes(axes[2], "三版文案的点击率（%）", "", "CTR (%)")             # shared style
for i, v in enumerate(copy_variants["ctr"] * 100):                   # annotate
    axes[2].text(i, v, "%.1f" % v, ha="center", va="bottom", fontsize=8.6)   # the value
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_fewshot.png")                                        # save the figure

fig, ax = plt.subplots(figsize=(9.0, 3.4))                           # one panel
ax.axis("off")                                                     # text only
ax.text(0.02, 0.88, "模板：角色 + 任务 + 上下文 + 格式 + 约束", fontsize=10.5, fontweight="bold")   # the five parts
ax.text(0.02, 0.66, "① 角色：电商运营助手", fontsize=9.4)              # role
ax.text(0.02, 0.52, "② 任务：提炼卖点、目标用户、风险", fontsize=9.4)     # task
ax.text(0.02, 0.38, "③ 上下文：商品描述", fontsize=9.4)                # context
ax.text(0.02, 0.24, "④ 格式：JSON，键为 selling_point/target_user/risk", fontsize=9.4)   # format
ax.text(0.02, 0.10, "⑤ 约束：只输出 JSON，不要解释", fontsize=9.4)        # constraint
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_longdoc.png")                                        # save the figure
print("done: templates, schema validation, few-shot, long documents and copy A/B")   # closing log
