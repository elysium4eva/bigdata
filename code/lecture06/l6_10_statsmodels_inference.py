"""l6_10_statsmodels_inference.py -- the inference view: standard errors, t values and intervals.

Every line carries an English comment. The script refits the prostate regression and
the spam-style logistic model with statsmodels, and records the coefficient table
with standard errors, confidence intervals and p values. This is the view a business
report needs when it quotes an effect rather than a prediction.
"""
import numpy as np                                                 # numeric helpers for the odds ratios
import pandas as pd                                                # tabular data handling
import statsmodels.api as sm                                       # OLS and Logit with full inference output
from _common_l6 import DAT, dump, dump_artifact, write_json, add_intercept   # shared helpers

# ---------------------------------------------------------------- 1) ordinary least squares with inference
prostate = pd.read_csv(DAT / "l6_prostate.csv")                    # the sample from lesson 6.1
cols = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]   # predictors
X = add_intercept(prostate[cols].values)                           # augmented design matrix
y = prostate["lpsa"].values                                        # target
ols = sm.OLS(y, X).fit()                                           # fitted model with full statistics
names = ["intercept"] + cols                                       # coefficient names in order
ci = ols.conf_int()                                                # 95 per cent confidence intervals
table = pd.DataFrame({"term": names,                               # term name
                      "coef": np.round(ols.params, 4),             # estimated coefficient
                      "std_err": np.round(ols.bse, 4),             # standard error
                      "t_value": np.round(ols.tvalues, 3),         # t statistic
                      "p_value": np.round(ols.pvalues, 4),         # two-sided p value
                      "ci_low": np.round(ci[:, 0], 4),             # lower bound of the interval
                      "ci_high": np.round(ci[:, 1], 4)})           # upper bound of the interval
dump(table, "l6_inference_ols.csv")                                # persist the table
print(table.to_string(index=False))                                # report it in the console
print("R^2 %.4f | adj R^2 %.4f | F %.2f | n %d" % (ols.rsquared, ols.rsquared_adj,   # fit statistics
                                                  ols.fvalue, int(ols.nobs)))   # and the sample size

# ---------------------------------------------------------------- 2) logistic regression with inference
two = pd.read_csv(DAT / "l6_two_class.csv")                        # the two-class sample from lesson 6.5
Xl = add_intercept(two[["x"]].values)                              # augmented feature matrix
yl = two["y"].values                                               # binary labels
logit = sm.Logit(yl, Xl).fit(disp=False)                           # maximum likelihood logistic regression
ci_l = logit.conf_int()                                            # confidence intervals
logit_table = pd.DataFrame({"term": ["intercept", "x"],            # term names
                            "coef": np.round(logit.params, 4),     # log-odds coefficients
                            "std_err": np.round(logit.bse, 4),     # standard errors
                            "z_value": np.round(logit.tvalues, 3),   # z statistics
                            "p_value": np.round(logit.pvalues, 4),   # p values
                            "odds_ratio": np.round(np.exp(logit.params), 4),   # exponentiated coefficients
                            "ci_low": np.round(ci_l[:, 0], 4),     # interval for the coefficient
                            "ci_high": np.round(ci_l[:, 1], 4)})   # interval for the coefficient
dump(logit_table, "l6_inference_logit.csv")                        # persist the table
print(logit_table.to_string(index=False))                          # report it in the console
print("pseudo R^2 %.4f | log-likelihood %.4f" % (logit.prsquared, logit.llf))   # report the fit

# ---------------------------------------------------------------- 3) a short markdown excerpt and a run log
excerpt = ["# Inference view (statsmodels)",                       # title line
           "",                                                     # blank line
           "## OLS: log PSA on clinical measures",                 # section title
           "",                                                     # blank line
           "| term | coef | std err | t | p | 95% CI |",           # table header
           "| --- | --- | --- | --- | --- | --- |"]                # table separator
for _, r in table.iterrows():                                      # one row per coefficient
    excerpt.append("| %s | %.4f | %.4f | %.2f | %.4f | [%.3f, %.3f] |" % (   # # format one markdown table row
        r["term"], r["coef"], r["std_err"], r["t_value"], r["p_value"], r["ci_low"], r["ci_high"]))   # row
excerpt += ["", "## Logit: binary target", "",                     # second section
            "| term | coef | odds ratio | z | p |",                # header
            "| --- | --- | --- | --- | --- |"]                     # separator
for _, r in logit_table.iterrows():                                # one row per coefficient
    excerpt.append("| %s | %.4f | %.4f | %.3f | %.4f |" % (r["term"], r["coef"],   # log-odds and odds ratio
                                                           r["odds_ratio"], r["z_value"], r["p_value"]))   # statistics
excerpt += ["", "Note: coefficients are log-odds; exponentiated values are odds ratios.", ""]   # closing note
dump_artifact("\n".join(excerpt), "l6_inference_excerpt.md")        # persist the markdown excerpt
write_json({"ols_r2": round(float(ols.rsquared), 4),                # run log: OLS fit
            "logit_pseudo_r2": round(float(logit.prsquared), 4),    # run log: logistic fit
            "n_ols": int(ols.nobs),                                 # OLS sample size
            "n_logit": int(logit.nobs)}, "l6_run_log_10.json")      # run log artefact
print("done: inference tables and excerpt")                        # closing log
