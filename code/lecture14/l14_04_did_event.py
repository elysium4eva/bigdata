"""l14_04_did_event.py -- difference-in-differences and an event study for a tariff shock.

Every line carries an English comment. The script simulates a tariff episode with staggered   # continuation of the call
adoption, estimates a two-way fixed-effects DID, plots the event-study coefficients to check   # continuation of the call
parallel trends, and contrasts the local (partial equilibrium) effect with the general   # continuation of the call
equilibrium number the deck discusses as PTI/MTI/GETI.                  # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # OLS formulas
from _common_l14 import (make_country_panel, make_geography,       # shared generators
                         make_trade_flows, save, dump, style_axes,      # shared plotting helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the panel and the shock
countries = make_country_panel(60, (2000, 2019))                    # country-year panel
geo = make_geography(60)                                           # geography
flows = make_trade_flows(countries, geo, (2000, 2019))              # dyad-year flows
# choose an importer and a set of exporters that get a tariff at year 2012
treated_importer = "US"                                            # the importer imposing the tariff
candidates = ["CN", "VN", "IN", "KR", "MX", "JP"]                   # possible treated exporters
treated = ["CN", "VN", "IN", "KR", "MX", "JP"]                      # the exporters actually hit
flows["treated_pair"] = ((flows["iso_d"] == treated_importer) & (flows["iso_o"].isin(treated))).astype(int)   # the treated cell
flows["post"] = (flows["year"] >= 2012).astype(int)                 # the post period
flows["did"] = flows["treated_pair"] * flows["post"]                 # the interaction
# impose the simulated tariff shock: a 35% fall in the treated dyads after 2012
shock_mask = flows["did"] == 1                                     # the cells that are hit
flows.loc[shock_mask, "trade"] = flows.loc[shock_mask, "trade"] * np.exp(-0.35)   # apply the shock
print("shock applied to %d dyad-years" % int(shock_mask.sum()))      # report
print("treated dyads:", int(flows["treated_pair"].sum()), "| post years:", int(flows["post"].sum()))   # report

# ---------------------------------------------------------------- 2) the DID regression
work = flows[flows["trade"] > 0].copy()                             # keep positive flows for the log model
work["ltrade"] = np.log(work["trade"])                              # log outcome
did = smf.ols("ltrade ~ did + C(iso_o) + C(iso_d) + C(year)", data=work).fit(   # estimate the model
    cov_type="cluster", cov_kwds={"groups": work["iso_o"]})          # two-way FE with clustered SEs
print("DID coefficient: %.3f (se %.3f) → %.1f%%" % (                    # the event-time loop
    did.params["did"], did.bse["did"], 100 * (np.exp(did.params["did"]) - 1)))   # report

# ---------------------------------------------------------------- 3) the event study
work["rel_year"] = np.where(work["treated_pair"] == 1, work["year"] - 2012, -99)   # event time for treated
rows = []                                                          # event-time coefficients
for k in range(-5, 7):                                             # five leads and six lags
    col = "ev_m%d" % abs(k) if k < 0 else "ev_p%d" % k               # column name
    work[col] = ((work["rel_year"] == k)).astype(int)                # event-time dummy
    if k == -1:                                                    # the reference period
        continue                                                   # skip the omitted category
    m = smf.ols("ltrade ~ %s + C(iso_o) + C(iso_d) + C(year)" % col, data=work).fit(   # estimate the model
        cov_type="cluster", cov_kwds={"groups": work["iso_o"]})      # one regression per event time
    rows.append({"event_time": k, "coef": float(m.params[col]),       # the coefficient
                 "se": float(m.bse[col])})                          # its standard error
    work.drop(columns=[col], inplace=True)                          # keep the frame small
event = pd.DataFrame(rows)                                          # assemble
dump(event.round(4), "l14_event_study.csv")                        # persist
print(event.round(3).to_string(index=False))                        # report

# ---------------------------------------------------------------- 4) three policy numbers
# A local (partial equilibrium) effect, a terms-of-trade effect and a global effect, all
# generated from the same DID coefficient with different pass-through assumptions.
did_coef = float(did.params["did"])                                 # the estimated effect
rows = []                                                          # three scenarios
for name, pass_through, ge_multiplier in [("PTI 局部均衡", 0.8, 1.0),   # partial equilibrium
                                          ("MTI 贸易条件", 0.6, 0.7),  # terms of trade
                                          ("GETI 全球一般均衡", 0.5, 0.45)]:   # global
    effect = did_coef * pass_through * ge_multiplier                 # the adjusted effect
    rows.append({"model": name, "effect_pct": round(100 * (np.exp(effect) - 1), 2),   # the number
                 "assumption": "pass-through %.1f, GE multiplier %.2f" % (pass_through, ge_multiplier)})   # label
three = pd.DataFrame(rows)                                          # assemble
dump(three, "l14_policy_three.csv")                                # persist
print(three.to_string(index=False))                                # report

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].errorbar(event["event_time"], event["coef"], yerr=1.96 * event["se"],   # draw the panel
                 fmt="-o", ms=4, color=OKABE[5], ecolor="#2B2B2B", capsize=2)   # coefficients with intervals
axes[0].axhline(0, color="#2B2B2B", lw=0.9)                         # zero line
axes[0].axvline(-0.5, ls="--", lw=1.0, color="#66717D")              # the treatment year
style_axes(axes[0], "事件研究：处理前的系数应接近 0", "相对年份", "系数")   # shared style
annotated(axes[0], "左侧不显著 = 平行趋势可信", "lower left")           # annotate
axes[1].bar(three["model"], three["effect_pct"], color=[OKABE[1], OKABE[2], OKABE[3]], alpha=0.92)   # three numbers
style_axes(axes[1], "同一冲击，三种口径", "", "贸易下降（%）")           # shared style
for i, v in enumerate(three["effect_pct"]):                         # annotate
    axes[1].text(i, v, "%.1f%%" % v, ha="center", va="bottom" if v >= 0 else "top", fontsize=9)   # value
axes[1].tick_params(axis="x", labelsize=7.6)                        # small labels
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_event_study.png")                                   # save the event-study figure

# ---------------------------------------------------------------- 6) the policy figure
fig, ax = plt.subplots(figsize=(8.6, 3.6))                          # one panel
xs = np.arange(len(three))                                          # positions
ax.bar(xs, three["effect_pct"], color=[OKABE[1], OKABE[2], OKABE[3]], alpha=0.92)   # the three numbers
ax.set_xticks(xs, three["model"], fontsize=8.4)                      # labels
for i, row in three.iterrows():                                     # annotate each bar
    ax.text(i, row["effect_pct"], "%.1f%%" % row["effect_pct"], ha="center",   # draw the panel
            va="bottom" if row["effect_pct"] >= 0 else "top", fontsize=9)   # the value
style_axes(ax, "PTI / MTI / GETI：口径不同，数字差一倍", "", "贸易下降（%）")   # shared style
annotated(ax, three["assumption"].iloc[0], "lower right")            # the assumption label
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_policy_three.png")                                  # save
print("done: DID, event study and the three policy numbers")          # closing log
