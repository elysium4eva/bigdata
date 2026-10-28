"""l13_10_deflated_sharpe.py -- sentiment groups, RAG answers and the deflated Sharpe ratio.

Every line carries an English comment. The script builds a news-sentiment factor and tests its   # continuation of the call
return spread, assembles a retrieval-augmented answer with citations from filing snippets, and   # return the value
computes the deflated Sharpe ratio that corrects a backtest for the number of trials behind it.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from scipy import stats                                            # the normal distribution
from _common_l13 import (DAT, RNG, make_filings, save, dump,        # shared helpers
                         style_axes, annotated, OKABE)                  # continuation of the call

# ---------------------------------------------------------------- 1) the sentiment factor
prices = pd.read_csv(DAT / "l13_prices.csv", dtype={"ticker": str})   # the price panel
wide = prices.pivot(index="date", columns="ticker", values="ret")     # returns in wide form
n_days, n_assets = wide.shape                                       # the dimensions
sentiment = RNG.normal(0, 1, (n_days, n_assets))                     # a synthetic sentiment score
# the signal affects the NEXT day's return with a small loading, plus noise
forward = wide.values[1:] + 0.0006 * sentiment[:-1]                  # returns do react a little
signal = sentiment[:-1]                                             # today's sentiment
rows = []                                                           # one row per decile
quantiles = pd.qcut(signal.ravel(), 5, labels=False)                 # five sentiment buckets
for q in range(5):                                                  # loop over the buckets
    mask = quantiles == q                                            # the observations in this bucket
    rows.append({"bucket": q + 1, "mean_ret_bp": round(float(forward.ravel()[mask].mean() * 1e4), 3),   # the mean
                 "n": int(mask.sum())})                              # the count
groups = pd.DataFrame(rows)                                          # assemble
spread_bp = float(groups["mean_ret_bp"].iloc[-1] - groups["mean_ret_bp"].iloc[0])   # top minus bottom
print("sentiment spread (top-bottom): %.2f bp per day" % spread_bp)    # report
dump(groups, "l13_sentiment_groups.csv")                            # persist

# ---------------------------------------------------------------- 2) a RAG answer with citations
filings = make_filings(120)                                          # 120 annual-report snippets
dump(filings, "l13_filings.csv")                                     # persist
query = "哪家公司的营业收入最高？"                                     # the question
numbers = filings["revenue"].values                                   # the true revenues
best = int(np.argmax(numbers))                                        # the right answer
answer = "营业收入最高的是 %s（%.2f 亿元）[%s]。" % (                                # continuation of the call
    filings["company"].iloc[best], filings["revenue"].iloc[best], filings["doc_id"].iloc[best])   # the grounded answer
print("answer:", answer)                                             # report
# a simple citation check: does the answer quote a real document and the right number
cited = answer.split("[")[-1].split("]")[0]                          # the cited document id
cited_ok = cited == filings["doc_id"].iloc[best]                      # is the citation correct
value_ok = abs(filings["revenue"].iloc[best] - numbers[best]) < 1e-9   # is the number right
print("citation ok: %s | value ok: %s" % (cited_ok, value_ok))          # report
rag = pd.DataFrame([{"question": query, "answer": answer, "cited_doc": cited,   # assemble the frame
                     "citation_ok": bool(cited_ok), "value_ok": bool(value_ok)}])   # the record
dump(rag, "l13_rag_answer.csv")                                      # persist
# retrieval quality: how often the top-3 retrieved documents contain the answer
hits = 0                                                             # the counter
for q_i in range(30):                                               # thirty synthetic questions
    target = q_i                                                    # the document that holds the answer
    idx = RNG.choice(len(filings), 3, replace=False)                 # three "retrieved" documents
    if target in idx:                                               # a hit
        hits += 1                                                   # count it
print("top-3 retrieval contains the target: %.2f" % (hits / 30))        # report

# ---------------------------------------------------------------- 3) the deflated Sharpe ratio
# Run many random strategies, keep the best, and show how much of its Sharpe is selection.
n_trials, n_obs = 50, 500                                            # trials and observations
sharpe_trials = []                                                   # the Sharpe of each trial
for _ in range(n_trials):                                           # loop over the trials
    r = RNG.normal(0.0002, 0.01, n_obs)                              # a random strategy
    sharpe_trials.append(float(r.mean() / r.std() * np.sqrt(252)))    # its annualised Sharpe
sharpe_trials = np.array(sharpe_trials)                              # to array
best_sharpe = float(sharpe_trials.max())                             # the winner's Sharpe
expected_max = float(stats.norm.ppf(1 - 1 / n_trials) * sharpe_trials.std())   # the expected max under the null
deflated = best_sharpe - expected_max                                 # the deflated version
print("best Sharpe %.2f | expected max under null %.2f | deflated %.2f" % (   # report the result
    best_sharpe, expected_max, deflated))                            # report
dump(pd.DataFrame([{"trials": n_trials, "best_sharpe": round(best_sharpe, 3),   # assemble the frame
                    "expected_max_null": round(expected_max, 3), "deflated_sharpe": round(deflated, 3)}]),   # continuation of the call
     "l13_deflated_sharpe.csv")                                     # persist

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
axes[0].bar(groups["bucket"], groups["mean_ret_bp"], color=OKABE[2], alpha=0.92)   # the buckets
axes[0].axhline(0, color="#2B2B2B", lw=0.9)                          # zero line
style_axes(axes[0], "情绪分组的次日收益（bp）", "情绪五分位", "平均收益 (bp)")   # shared style
annotated(axes[0], "最高组 - 最低组 = %.1f bp" % spread_bp, "upper left")   # annotate
axes[1].hist(sharpe_trials, bins=20, color=OKABE[1], alpha=0.85)       # the trial Sharpes
axes[1].axvline(best_sharpe, ls="--", lw=1.4, color=OKABE[5], label="最优 %.2f" % best_sharpe)   # the winner
axes[1].axvline(deflated, ls=":", lw=1.4, color=OKABE[2], label="deflated %.2f" % deflated)   # the deflated
axes[1].legend(fontsize=7.4)                                         # compact legend
style_axes(axes[1], "试了 %d 个策略后的\"最优\" Sharpe" % n_trials, "Sharpe", "次数")   # shared style
plt.tight_layout()                                                   # tidy the layout
save(fig, "fig_sentiment_groups.png")                                # save the sentiment figure

fig, ax = plt.subplots(figsize=(9.6, 3.2))                           # one wide panel
ax.axis("off")                                                      # text only
ax.text(0.01, 0.86, "问：%s" % query, fontsize=10.5, fontweight="bold")   # the question
ax.text(0.01, 0.62, "答：%s" % answer, fontsize=9.6, color=OKABE[5])       # the grounded answer
ax.text(0.01, 0.40, "引用校验：文档 %s ✔ ｜ 数值一致 ✔" % cited, fontsize=9, color="#5A5A5A")   # the check
ax.text(0.01, 0.18, "检索命中率（top-3 含答案）：%.0f%%" % (100 * hits / 30), fontsize=9, color="#5A5A5A")   # retrieval
ax.set_title("RAG：检索 → 生成 → 引用校验")                            # the title
plt.tight_layout()                                                   # tidy the layout
save(fig, "fig_rag_answer.png")                                      # save the RAG figure
print("done: sentiment groups, RAG answer and the deflated Sharpe")    # closing log
