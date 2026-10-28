"""l11_10_hallucination.py -- hallucination detection by self-consistency and citation checks.

Every line carries an English comment. The script builds a small claim set whose truth is known,   # continuation of the call
produces answers under three guardrail strategies (no check, self-consistency, citation   # continuation of the call
verification), and draws the trade-off between missed hallucinations and false alarms.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l11 import (get_model, make_documents, tokenize,        # shared helpers
                         save, dump, style_axes, annotated, OKABE, RNG)   # continuation of the call

model = get_model()                                                # the local chat model
docs = make_documents(20)                                          # a small ground-truth library

# ---------------------------------------------------------------- 1) claims with known truth
claims = []                                                        # one row per claim
for i, row in docs.head(12).iterrows():                            # use the first documents as truth
    claims.append({"claim_id": "C%02d" % (i + 1),                    # the identifier
                   "true_doc": row["doc_id"],                        # the document that supports it
                   "topic": row["topic"]})                           # the topic
claims = pd.DataFrame(claims)                                      # assemble
print("claims:", len(claims))                                      # report

def propose(claim, temperature=0.9):                               # one model attempt
    """Return a cited document id proposed for a claim (with mistakes at higher temperature)."""   # short docstring
    right = claim["true_doc"]                                      # the correct citation
    if RNG.random() < 0.55 + 0.1 * temperature:                     # the model is right most of the time
        return right                                               # a correct citation
    others = docs["doc_id"].tolist()                               # all candidates
    return str(RNG.choice([d for d in others if d != right]))        # a plausible-looking wrong one

# ---------------------------------------------------------------- 2) three guardrail strategies
def no_check(claim):                                               # strategy 1
    """Return the single attempt and accept it."""                 # short docstring
    cited = propose(claim, temperature=1.0)                         # one sample
    return cited, True                                             # always accepted

def self_consistency(claim, k=5):                                  # strategy 2
    """Return the majority citation and accept only if it repeats."""   # short docstring
    votes = [propose(claim, temperature=1.0) for _ in range(k)]      # k samples
    counted = {}                                                   # the tally
    for v in votes:                                                # count the votes
        counted[v] = counted.get(v, 0) + 1                          # increment
    winner, count = max(counted.items(), key=lambda kv: kv[1])       # the mode
    return winner, count >= 3                                      # accept only a clear majority

def citation_check(claim, cited):                                  # strategy 3
    """Return True when the cited document actually mentions the topic."""   # short docstring
    rows = docs[docs["doc_id"] == cited]                            # the cited document
    if rows.empty:                                                 # unknown id
        return False                                               # reject
    return claim["topic"] in rows.iloc[0]["body"]                   # a light topical check

rows = []                                                          # one row per claim
for _, claim in claims.iterrows():                                 # loop over the claims
    cited1, _ = no_check(claim)                                    # strategy 1
    cited2, accepted2 = self_consistency(claim, 5)                  # strategy 2
    ok3 = citation_check(claim, propose(claim, temperature=0.3))     # strategy 3 (low temperature)
    rows.append({"claim_id": claim["claim_id"], "true_doc": claim["true_doc"],   # the record
                 "no_check_ok": int(cited1 == claim["true_doc"]),   # strategy 1 correctness
                 "selfcons_ok": int(accepted2 and cited2 == claim["true_doc"]),   # strategy 2
                 "citation_ok": int(ok3)})                          # strategy 3
out = pd.DataFrame(rows)                                           # assemble
dump(out, "l11_hallucination.csv")                                 # persist
summary = pd.DataFrame([                                                # assemble the frame
    {"strategy": "不检查", "correct_rate": round(float(out["no_check_ok"].mean()), 3),   # continuation of the call
     "miss_rate": round(1 - float(out["no_check_ok"].mean()), 3), "extra_cost": 1},   # baseline
    {"strategy": "自洽投票 k=5", "correct_rate": round(float(out["selfcons_ok"].mean()), 3),   # continuation of the call
     "miss_rate": round(1 - float(out["selfcons_ok"].mean()), 3), "extra_cost": 5},   # voting
    {"strategy": "引用核验", "correct_rate": round(float(out["citation_ok"].mean()), 3),   # continuation of the call
     "miss_rate": round(1 - float(out["citation_ok"].mean()), 3), "extra_cost": 1},   # citation
])                                                                 # the summary
dump(summary, "l11_guardrail.csv")                                 # persist
print(summary.to_string(index=False))                              # report

# ---------------------------------------------------------------- 3) the trade-off curve
# A threshold sweep on the self-consistency vote share: higher threshold = fewer false accepts.
thresholds = np.linspace(0.2, 0.9, 15)                              # the vote-share cut-offs
curve = []                                                         # one row per threshold
for t in thresholds:                                               # loop over the cut-offs
    accepts = correct = 0                                          # counters
    for _ in range(400):                                           # simulate the accept/reject decision
        true_positive = RNG.random() < 0.72                         # the model is right 72% of the time
        vote_share = RNG.uniform(0.8, 1.0) if true_positive else RNG.uniform(0.2, 0.8)   # the evidence
        if vote_share >= t:                                        # the gate accepts
            accepts += 1                                           # count the accept
            correct += int(true_positive)                           # count the correct accept
    missed = 1 - (correct / 400)                                    # hallucinations that slipped through
    false_alarm = 1 - (accepts / 400) if accepts else 1.0             # valid answers we refused
    curve.append({"threshold": round(float(t), 2), "miss_rate": round(missed, 3),   # continuation of the call
                  "reject_rate": round(false_alarm, 3)})           # the row
trade = pd.DataFrame(curve)                                        # assemble
dump(trade, "l11_guardrail_curve.csv")                             # persist

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].bar(summary["strategy"], summary["miss_rate"], color=[OKABE[5], OKABE[1], OKABE[2]], alpha=0.92)   # miss rate
axes[0].set_xticks(range(3), ["不检查", "自洽投票", "引用核验"], rotation=12, ha="right")   # labels
style_axes(axes[0], "漏检率：三种护栏的对比", "", "漏检率（越低越好）")      # shared style
for i, v in enumerate(summary["miss_rate"]):                        # annotate
    axes[0].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=8.6)   # the value
axes[1].plot(trade["threshold"], trade["miss_rate"], "-o", ms=4, color=OKABE[5], label="漏检率")   # miss
axes[1].plot(trade["threshold"], trade["reject_rate"], "-s", ms=4, color=OKABE[1], label="误拒率")   # reject
axes[1].legend(fontsize=7.4)                                       # compact legend
style_axes(axes[1], "阈值越高越安全，但会误拒正常回答", "投票一致性阈值", "比率")   # shared style
annotated(axes[1], "护栏永远是权衡，不是免费的准确率", "upper center")        # annotate
plt.tight_layout()                                                 # tight layout
save(fig, "fig_guardrail.png")                                     # save the figure
print("done: hallucination detection and guardrails")                 # closing log
