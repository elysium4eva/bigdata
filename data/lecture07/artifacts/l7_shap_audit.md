# SHAP audit for Lecture 7

Model: XGBoost on the synthetic tabular sample (1,400 rows, six features).

| feature | mean(abs(SHAP)) | share |
| --- | --- | --- |
| f1 | 1.0403 | 37.7% |
| f2 | 0.5301 | 19.2% |
| f3 | 0.4089 | 14.8% |
| f4 | 0.3433 | 12.5% |
| f5 | 0.3042 | 11.0% |
| f6 | 0.1289 | 4.7% |

The audit table is the artifact a compliance review asks for: it names the features
that drive the score and quantifies their relative influence.
