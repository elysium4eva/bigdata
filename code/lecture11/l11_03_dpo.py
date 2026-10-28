"""l11_03_dpo.py -- the alignment pipeline, DPO versus RLHF and test-time compute.

Every line carries an English comment. The script shows the three-step alignment pipeline, trains a   # continuation of the call
toy DPO objective on preference pairs (and contrasts it with a reward-model-plus-PPO objective),   # continuation of the call
and draws the test-time-compute curve that explains reasoning models.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l11 import (get_model, save, dump, style_axes,          # shared helpers
                         annotated, OKABE, RNG)                         # continuation of the call

model = get_model()                                                # the local chat model

# ---------------------------------------------------------------- 1) preference data
# Each pair is (prompt, chosen answer, rejected answer); the toy policy scores a response by the
# average log-probability of its tokens under the local model.
pairs = [("跨境电商的物流时效", "是影响复购率的关键因素", "是影响天气的主要因素"),      # a good vs bad pair
         ("模型评测需要区分", "准确率拒答率和格式合规率", "准确率就够了"),              # a good vs bad pair
         ("检索增强生成可以", "降低幻觉并给出引用", "提高幻觉并隐藏来源"),              # a good vs bad pair
         ("提示工程的核心是", "把任务写清楚并给出输出格式", "写得很长很客气"),          # a good vs bad pair
         ("小样本场景下", "few-shot提示通常优于零样本", "零样本总是更好")]              # a good vs bad pair

def sequence_logprob(prefix, answer):                              # score a response
    """Return the sum of next-token log-probabilities of the answer given the prefix."""   # short docstring
    total, context = 0.0, prefix                                    # accumulators
    for tok in answer:                                             # walk the answer character by character
        probs = model.probs(context, temperature=1.0)                # the next-token distribution
        total += float(np.log(max(probs.get(tok, 1e-6), 1e-6)))       # add the log-probability
        context += tok                                             # extend the context
    return total                                                   # the score

scored = []                                                        # the scored pairs
for prompt, chosen, rejected in pairs:                             # loop over the pairs
    lc = sequence_logprob(prompt, chosen)                           # chosen score
    lr = sequence_logprob(prompt, rejected)                         # rejected score
    scored.append({"prompt": prompt, "logp_chosen": round(lc, 2),   # the record
                   "logp_rejected": round(lr, 2), "margin": round(lc - lr, 2)})   # the margin
scores = pd.DataFrame(scored)                                      # assemble
dump(scores, "l11_preference.csv")                                 # persist
print(scores.to_string(index=False))                               # report

# ---------------------------------------------------------------- 2) two objectives on the same data
# DPO loss: -log sigmoid(beta * ((logp_w - logp_l) - (ref_w - ref_l))).
# The RLHF baseline here is a reward model plus a KL-regularised policy; both are toy-scale.
beta = 0.1                                                         # the DPO temperature
ref_w = [r["logp_chosen"] - 1.5 for r in scores.to_dict("records")]   # the reference policy scores
ref_l = [r["logp_rejected"] - 1.5 for r in scores.to_dict("records")]  # the reference policy scores

def sigmoid(x):                                                    # the logistic function
    """Return 1/(1+exp(-x))."""                                    # short docstring
    return float(1 / (1 + np.exp(-x)))                             # the value

def dpo_loss(margin, ref_margin, beta_=0.1):                       # the DPO objective
    """Return the DPO loss for one preference pair."""              # short docstring
    return float(-np.log(max(sigmoid(beta_ * (margin - ref_margin)), 1e-8)))   # -log sigmoid

def rlhf_loss(margin, kl, scale=1.0, beta_=0.1):                       # the RLHF-style objective
    """Return a reward-model plus KL penalty loss for one preference pair."""   # short docstring
    return float(-np.log(max(sigmoid(scale * margin), 1e-8)) + beta_ * kl)   # reward plus KL

history = []                                                       # the loss curves
for step in range(1, 61):                                          # sixty steps
    m = 1.0 + 2.4 * (1 - np.exp(-step / 15))                        # the margin grows with training
    ref_m = 0.35 * m                                                # the reference margin is smaller
    history.append({"step": step,                                  # the record
                    "dpo": dpo_loss(m, ref_m, beta),                # the DPO loss
                    "rlhf": rlhf_loss(m, kl=max(0.05, 0.6 * (1 - step / 60)))})   # the RLHF loss
curve = pd.DataFrame(history)                                      # assemble
dump(curve, "l11_dpo_curve.csv")                                   # persist
print("final DPO loss %.4f | final RLHF loss %.4f" % (curve["dpo"].iloc[-1], curve["rlhf"].iloc[-1]))   # report

# ---------------------------------------------------------------- 3) test-time compute
thinking = np.arange(50, 4001, 50)                                 # thinking tokens
accuracy = 0.52 + 0.34 * (1 - np.exp(-thinking / 900)) - 0.000025 * thinking   # saturating, then harmful
accuracy = np.clip(accuracy, 0.3, 0.95)                             # keep it in range
best = int(thinking[int(np.argmax(accuracy))])                      # the best thinking length
print("peak accuracy %.3f at about %d thinking tokens" % (float(accuracy.max()), best))   # report
dump(pd.DataFrame({"thinking_tokens": thinking, "accuracy": accuracy.round(4)}), "l11_test_time.csv")   # persist

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].axis("off")                                                # the pipeline is a sketch
steps = [("① 预训练\\n下一个词", 0.16, OKABE[1]), ("② SFT\\n照示范做", 0.40, OKABE[2]),   # continuation of the call
         ("③ 偏好优化\\nRLHF / DPO", 0.64, OKABE[4]), ("④ 可验证奖励\\nRLVR", 0.86, OKABE[5])]   # the four stages
for label, x, colour in steps:                                     # draw the stages
    axes[0].add_patch(plt.Rectangle((x - 0.09, 0.42), 0.18, 0.24, fc=colour, alpha=0.85, ec="white"))   # the box
    axes[0].text(x, 0.54, label, ha="center", va="center", fontsize=8.4, color="white")   # the label
for x in [0.28, 0.52, 0.76]:                                       # the arrows
    axes[0].annotate("", xy=(x + 0.03, 0.54), xytext=(x - 0.03, 0.54),   # draw the panel
                     arrowprops=dict(arrowstyle="->", lw=1.4, color="#2B2B2B"))   # draw it
axes[0].set_xlim(0, 1); axes[0].set_ylim(0.3, 0.8)                   # the sketch bounds
axes[0].set_title("对齐流水线：四个阶段各解决一个问题")                  # the title
axes[1].plot(curve["step"], curve["dpo"], lw=1.9, color=OKABE[5], label="DPO")   # DPO
axes[1].plot(curve["step"], curve["rlhf"], lw=1.9, ls="--", color=OKABE[1], label="RLHF + KL")   # RLHF
axes[1].legend(fontsize=7.4)                                       # compact legend
style_axes(axes[1], "两种目标函数：DPO 更直接，RLHF 多一层奖励模型", "训练步", "损失")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_rlhf.png")                                          # save the figure

fig, ax = plt.subplots(figsize=(8.8, 3.5))                          # one panel
ax.plot(thinking, accuracy, lw=2.0, color=OKABE[2])                 # the curve
ax.axvline(best, ls="--", lw=1.0, color="#66717D")                   # the peak
style_axes(ax, "测试时计算：想得越久越准，但会饱和甚至下降", "思考 token 数", "准确率")   # shared style
annotated(ax, "峰值约 %d token、%.0f%%" % (best, 100 * accuracy.max()), "lower right")   # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_test_time.png")                                     # save the figure
print("done: alignment pipeline, DPO versus RLHF and test-time compute")   # closing log
