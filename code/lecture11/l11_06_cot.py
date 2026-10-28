"""l11_06_cot.py -- chain-of-thought and self-consistency voting on a classification task.

Every line carries an English comment. The script classifies short sentences into topics three ways   # continuation of the call
(direct label, a three-step rubric, and a rubric with majority voting) and shows where the extra   # continuation of the call
steps and the extra samples actually buy accuracy.                      # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l11 import (get_model, save, dump, style_axes,          # helpers
                         annotated, OKABE, RNG, CORPUS, tokenize)       # continuation of the call

model = get_model()                                                # the local chat model

# ---------------------------------------------------------------- 1) the task and its labels
TOPIC_LEXICON = {                                                  # topic -> trigger words
    "物流": ["物流", "时效", "复购", "退货"],                          # logistics
    "投放": ["投放", "广告", "预算", "客户"],                          # advertising
    "评测": ["评测", "准确率", "拒答", "格式"],                        # evaluation
    "提示": ["提示", "few-shot", "零样本", "任务"],                    # prompting
}                                                                  # end of the lexicon

def gold_topic(sentence):                                          # the label defined by the lexicon
    """Return the topic with the most trigger-word hits (the label we will grade against)."""   # short docstring
    hits = {t: sum(1 for w in words if w in sentence) for t, words in TOPIC_LEXICON.items()}   # count hits
    return max(hits.items(), key=lambda kv: kv[1])[0]               # the argmax

labelled = [{"sentence": s, "gold": gold_topic(s)} for s in CORPUS]   # label the corpus
data = pd.DataFrame(labelled)                                      # assemble
dump(data, "l11_cot_labels.csv")                                   # persist
print("labels:", data["gold"].value_counts().to_dict())             # report

# ---------------------------------------------------------------- 2) three strategies
def direct_label(sentence):                                        # one-shot label
    """Return the top topic guessed directly from the sentence."""   # short docstring
    # a direct guess is noisy: it picks among topics that share any character with the sentence
    candidates = [t for t in TOPIC_LEXICON if any(w in sentence for w in TOPIC_LEXICON[t])]   # candidates
    if not candidates:                                             # nothing matched
        return "物流"                                              # fall back to the most common
    return str(RNG.choice(candidates))                              # pick one at random

def cot_label(sentence):                                           # the three-step rubric
    """Return the label after an explicit three-step rubric."""      # short docstring
    toks = tokenize(sentence)                                       # step 1: tokenise
    scores = {}                                                     # step 2: score each topic
    for topic, words in TOPIC_LEXICON.items():                      # loop over topics
        scores[topic] = sum(1 for w in words if w in sentence)       # count trigger hits
    best = max(scores.items(), key=lambda kv: kv[1])                # step 3: take the argmax
    return best[0] if best[1] > 0 else "物流"                        # the decision

def voted_label(sentence, k=5, temperature=0.15):                  # rubric plus voting
    """Return the majority label of k noisy rubric runs."""          # short docstring
    votes = []                                                      # the ballots
    for _ in range(k):                                             # k independent runs
        scores = {t: sum(1 for w in words if w in sentence) + RNG.normal(0, temperature)   # continuation of the call
                  for t, words in TOPIC_LEXICON.items()}            # jittered scores
        votes.append(max(scores.items(), key=lambda kv: kv[1])[0])   # one ballot
    counted = {}                                                   # the tally
    for v in votes:                                                # count the ballots
        counted[v] = counted.get(v, 0) + 1                          # increment
    return max(counted.items(), key=lambda kv: kv[1])[0]            # the mode

rows = []                                                          # one row per sentence
for _, r in data.iterrows():                                       # loop over the corpus
    s, gold = r["sentence"], r["gold"]                              # the sentence and its label
    rows.append({"sentence": s, "gold": gold,                       # the record
                 "direct": direct_label(s),                        # direct guess
                 "cot": cot_label(s),                              # rubric
                 "voted": voted_label(s, 5)})                      # rubric plus vote
detail = pd.DataFrame(rows)                                        # assemble
acc_direct = float((detail["direct"] == detail["gold"]).mean())     # direct accuracy
acc_cot = float((detail["cot"] == detail["gold"]).mean())           # rubric accuracy
acc_voted = float((detail["voted"] == detail["gold"]).mean())       # voting accuracy
print("direct %.3f | CoT %.3f | CoT+voting %.3f" % (acc_direct, acc_cot, acc_voted))   # report
dump(detail, "l11_cot_detail.csv")                                 # persist

# ---------------------------------------------------------------- 3) how many votes are enough
vote_rows = []                                                     # one row per k
for k in [1, 3, 5, 9, 15]:                                         # increasing budget
    acc = float(np.mean([voted_label(s, k) == g for s, g in zip(detail["sentence"], detail["gold"])]))   # accuracy
    vote_rows.append({"k": k, "accuracy": round(acc, 3), "cost_ratio": k})   # the row
votes = pd.DataFrame(vote_rows)                                    # assemble
dump(votes, "l11_selfconsistency.csv")                             # persist
print(votes.to_string(index=False))                                # report

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
names = ["直接标签", "三步模板", "模板+投票"]                          # the three strategies
vals = [acc_direct, acc_cot, acc_voted]                             # their accuracies
axes[0].bar(names, vals, color=[OKABE[1], OKABE[2], OKABE[5]], alpha=0.92)   # the comparison
for i, v in enumerate(vals):                                        # annotate
    axes[0].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=9)   # the value
axis_max = max(vals) + 0.12                                         # headroom
axes[0].set_ylim(0, min(1.0, axis_max))                              # keep the scale honest
style_axes(axes[0], "写步骤带来的提升，比多采样更便宜", "", "准确率")      # shared style
axes[1].plot(votes["k"], votes["accuracy"], "-o", ms=5, color=OKABE[5])   # accuracy vs k
axes[1].set_xticks(votes["k"])                                      # tick every k
style_axes(axes[1], "k 越大不一定越稳：噪声大时投票会放大错误", "采样次数 k", "准确率")   # shared style
annotated(axes[1], "投票的前提：错误随机且分散", "lower right")             # annotate
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_cot.png")                                           # save the figure
print("done: chain-of-thought and self-consistency")                  # closing log
