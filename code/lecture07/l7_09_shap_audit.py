"""l7_09_shap_audit.py -- Shapley values as an audit tool.

Every line carries an English comment. The script fits a gradient-boosted model on
the tabular sample, explains it with SHAP, draws the summary and one dependence plot,
and writes an audit table that lists the most influential features with their mean
absolute contribution.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # the SHAP summary figure
import shap                                                        # the explanation library
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.metrics import r2_score                               # model quality
from xgboost import XGBRegressor                                   # the model being explained
from _common_l7 import (DAT, encode_nominal, save, dump, dump_artifact,   # shared helpers
                        style_axes, annotated, OKABE)                   # shared plotting helpers

# ---------------------------------------------------------------- 1) the model
data = pd.read_csv(DAT / "l7_tabular.csv")                         # the tabular sample from lesson 7.7
features = ["f1", "f2", "f3", "f4", "f5", "f6"]                    # predictor names
y = data["target"].values                                          # the target
X_train_raw, X_test_raw, y_train, y_test = train_test_split(data[features], y, test_size=0.3, random_state=42)   # split
X_train, X_test, nom_map = encode_nominal(X_train_raw, X_test_raw)   # encode only the nominal column
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,   # the explainable model
                     colsample_bytree=0.9, random_state=42, n_jobs=-1).fit(X_train, y_train)   # fit it
print("test R2: %.4f" % r2_score(y_test, model.predict(X_test)))    # report the quality first

# ---------------------------------------------------------------- 2) Shapley values
explainer = shap.TreeExplainer(model)                              # exact tree explainer
shap_values = explainer.shap_values(X_test)                        # one contribution matrix
mean_abs = np.abs(shap_values).mean(axis=0)                        # mean absolute contribution per feature
order = np.argsort(mean_abs)[::-1]                                 # features from most to least influential
audit = pd.DataFrame({"feature": [features[i] for i in order],      # feature names in audit order
                      "mean_abs_shap": np.round(mean_abs[order], 4),   # mean absolute contribution
                      "share": np.round(mean_abs[order] / mean_abs.sum(), 4)})   # relative share
dump(audit, "l7_shap_audit.csv")                                   # persist the audit table
print(audit.to_string(index=False))                                # report the table

# ---------------------------------------------------------------- 3) a single-customer explanation
index = int(np.argmax(np.abs(shap_values.sum(axis=1))))            # the row with the largest total effect
single = pd.DataFrame({"feature": features,                        # feature names
                       "value": np.round(X_test[index], 3),        # the observed values
                       "shap": np.round(shap_values[index], 4)})   # the contribution of each feature
dump(single, "l7_shap_single.csv")                                 # persist the single explanation
markdown = ["# SHAP audit for Lecture 7", "",                       # markdown excerpt for the deck
            "Model: XGBoost on the synthetic tabular sample (1,400 rows, six features).", "",   # the model description
            "| feature | mean(abs(SHAP)) | share |", "| --- | --- | --- |"]   # the table header
for _, r in audit.iterrows():                                      # one row per feature
    markdown.append("| %s | %.4f | %.1f%% |" % (r["feature"], r["mean_abs_shap"], r["share"] * 100))   # row
markdown += ["", "The audit table is the artifact a compliance review asks for: it names the features",   # note line one
             "that drive the score and quantifies their relative influence.", ""]   # note line two
dump_artifact("\n".join(markdown), "l7_shap_audit.md")              # persist the excerpt

# ---------------------------------------------------------------- 4) figure: summary and dependence
fig = plt.figure(figsize=(13.4, 4.2))                              # one wide figure
ax1 = fig.add_subplot(1, 3, 1)                                     # first panel: the bar summary
ax1.barh([features[i] for i in order][::-1], mean_abs[order][::-1],   # mean absolute contributions
         color=OKABE[3], alpha=0.9)                                # horizontal bars
style_axes(ax1, "Mean |SHAP| by feature", "mean absolute contribution", "")   # shared style
ax2 = fig.add_subplot(1, 3, 2)                                     # second panel: the dependence plot
top = int(order[0])                                                # the most influential feature
a = X_test[:, top]                                                 # its observed values
ax2.scatter(a, shap_values[:, top], s=8, color=OKABE[5], alpha=0.55)   # value versus contribution
style_axes(ax2, "Dependence: %s" % features[top], "feature value", "SHAP value")   # shared style
ax3 = fig.add_subplot(1, 3, 3)                                     # third panel: the single customer
colours = [OKABE[5] if v > 0 else OKABE[1] for v in single["shap"]]   # red for positive effects
ax3.barh(single["feature"], single["shap"], color=colours, alpha=0.9)   # contribution bars
ax3.axvline(0, color="#66717D", lw=0.9)                            # zero reference
style_axes(ax3, "One customer: why this score", "SHAP value", "")   # shared style
annotated(ax3, "row %d" % index, "upper right")                    # annotate which row
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_shap.png")                                          # save the figure
print("done: SHAP audit table, single explanation and figure")      # closing log
