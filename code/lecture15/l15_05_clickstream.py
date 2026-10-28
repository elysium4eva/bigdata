"""l15_05_clickstream.py -- funnel, RFM and the four attribution rules.

Every line carries an English comment. The script builds a 6,000-session clickstream, computes
the five-stage funnel, scores customers with RFM, and compares last-click, first-click, linear
and time-decay attribution on the same paths so the deck can show why the rules disagree.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l15 import (make_clickstream, CHANNELS, STAGES, save,   # shared helpers
                         dump, style_axes, annotated, OKABE)            # shared plotting helpers

# ---------------------------------------------------------------- 1) the clickstream
sessions = make_clickstream(6000)                                  # 6,000 sessions
dump(sessions, "l15_clickstream.csv")                              # persist
print("sessions:", len(sessions), "| conversion rate: %.3f" % sessions["converted"].mean())   # report

# ---------------------------------------------------------------- 2) the funnel
funnel = []                                                        # one row per stage
for k, stage in enumerate(STAGES):                                 # loop over the five stages
    reached = int((sessions["depth"] >= k + 1).sum())               # sessions that got this far
    funnel.append({"stage": stage, "sessions": reached})            # the row
funnel = pd.DataFrame(funnel)                                      # assemble
funnel["rate_vs_prev"] = (funnel["sessions"] / funnel["sessions"].shift(1)).round(4)   # step conversion
funnel["rate_vs_top"] = (funnel["sessions"] / funnel["sessions"].iloc[0]).round(4)   # cumulative conversion
dump(funnel, "l15_funnel.csv")                                     # persist
print(funnel.to_string(index=False))                               # report
worst = funnel.loc[1:, "rate_vs_prev"].idxmin()                     # the weakest stage
print("weakest step:", funnel.loc[worst, "stage"], "%.3f" % funnel.loc[worst, "rate_vs_prev"])   # report

# ---------------------------------------------------------------- 3) RFM scoring
rfm = sessions.groupby("last_channel").agg(sessions=("session_id", "count"),   # how many sessions
                                           conversion=("converted", "mean"),   # conversion rate
                                           revenue=("revenue", "mean")).reset_index()   # average revenue
rfm["index"] = (rfm["sessions"] / rfm["sessions"].sum() * 100).round(1)   # share of sessions
dump(rfm.round(3), "l15_channel_rfm.csv")                          # persist
print(rfm.round(3).to_string(index=False))                          # report

# ---------------------------------------------------------------- 4) attribution rules
def attribute(path, revenue, rule):                                # split one conversion across touches
    """Return a dict channel -> credited revenue under one attribution rule."""   # short docstring
    channels = [c.strip() for c in path.split(">")]                 # the touch sequence
    n = len(channels)                                              # number of touches
    credit = {}                                                    # channel -> credit
    if rule == "last":                                             # all credit to the last touch
        credit[channels[-1]] = revenue                             # single channel
    elif rule == "first":                                          # all credit to the first touch
        credit[channels[0]] = revenue                              # single channel
    elif rule == "linear":                                         # equal split
        for c in channels:                                         # every touch gets the same share
            credit[c] = credit.get(c, 0.0) + revenue / n            # add the share
    elif rule == "time_decay":                                     # recent touches weigh more
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])   # halve every step back
        weights = weights / weights.sum()                           # normalise
        for c, w in zip(channels, weights):                        # distribute
            credit[c] = credit.get(c, 0.0) + revenue * w            # add the share
    return credit                                                  # return the split

conv = sessions[sessions["converted"] == 1]                         # only converting sessions
print("converting sessions:", len(conv))                            # report
rules = ["last", "first", "linear", "time_decay"]                    # the four rules
credit = {r: {} for r in rules}                                    # channel credit per rule
for _, row in conv.iterrows():                                     # loop over conversions
    for r in rules:                                                # loop over the rules
        for c, v in attribute(row["path"], row["revenue"], r).items():   # split this conversion
            credit[r][c] = credit[r].get(c, 0.0) + v                # accumulate
attribution = pd.DataFrame(credit).fillna(0.0)                      # channel x rule table
attribution["channel"] = attribution.index                         # keep the labels
shares = attribution.set_index("channel")[rules]                    # the revenue matrix
shares_pct = (shares / shares.sum() * 100).round(1)                 # convert to percentages
dump(shares_pct.reset_index(), "l15_attribution.csv")              # persist
print(shares_pct.to_string())                                       # report

# ---------------------------------------------------------------- 5) the funnel figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].barh(funnel["stage"][::-1], funnel["sessions"][::-1], color=OKABE[4], alpha=0.9)   # the funnel
for i, (stage, n) in enumerate(zip(funnel["stage"][::-1], funnel["sessions"][::-1])):   # annotate
    axes[0].text(n, i, "%d" % n, va="center", fontsize=8.4)          # the count
style_axes(axes[0], "Five-stage funnel", "sessions", "")            # shared style
axes[1].plot(range(len(STAGES)), funnel["rate_vs_top"], "-o", ms=5, color=OKABE[5])   # cumulative conversion
style_axes(axes[1], "Cumulative conversion by stage", "", "share of 曝光")   # shared style
axes[1].set_xticks(range(len(STAGES)), STAGES, fontsize=8)           # stage labels
annotated(axes[1], "最弱一步：%s" % funnel.loc[worst, "stage"], "upper right")   # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_funnel.png")                                        # save the funnel figure

# ---------------------------------------------------------------- 6) the attribution figure
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))                 # two panels
xpos = np.arange(len(shares_pct))                                   # channel positions
width = 0.2                                                        # bar width
for k, rule in enumerate(rules):                                   # one series per rule
    axes[0].bar(xpos + (k - 1.5) * width, shares_pct[rule], width=width,   # the bars
                label={"last": "末次点击", "first": "首次点击",   # readable labels
                       "linear": "线性平分", "time_decay": "时间衰减"}[rule], color=OKABE[k], alpha=0.92)   # colours
axes[0].set_xticks(xpos, list(shares_pct.index), rotation=20, fontsize=8)   # channel labels
style_axes(axes[0], "Attributed revenue share by rule (%)", "", "percent")   # shared style
axes[0].legend(fontsize=7.2, ncol=2)                                # compact legend
diff = (shares_pct[rules].max(axis=1) - shares_pct[rules].min(axis=1)).round(1)   # spread per channel
axes[1].barh(list(shares_pct.index)[::-1], diff.values[::-1], color=OKABE[5], alpha=0.9)   # how much rules disagree
style_axes(axes[1], "Spread across the four rules (pp)", "percentage points", "")   # shared style
for i, v in enumerate(diff.values[::-1]):                            # annotate
    axes[1].text(v, i, "%.1f" % v, va="center", fontsize=8.2)        # the value
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_attribution.png")                                   # save the attribution figure
print("done: funnel, RFM and four attribution rules")                # closing log
