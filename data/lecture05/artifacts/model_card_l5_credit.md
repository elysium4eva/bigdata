# Model card -- Lecture 5 credit default demo

## Intended use
Teaching example for an introduction to machine learning. **Not** a production
credit decision system: the data are synthetic and the model has not been
validated on any real portfolio.

## Model
Gradient boosting classifier (`HistGradientBoostingClassifier`, 320 iterations,
learning rate 0.07) inside a pipeline with one-hot encoding for `education`.

## Data
`l5_credit_default.csv`, 3,000 synthetic accounts, positive rate
0.267. Split 70/30 stratified (seed 42). No protected attributes are
included; `education` acts as a proxy and is therefore a fairness risk.

## Metrics (held-out test split)
- ROC-AUC: 0.8092
- Brier score: 0.1579
- Most important features (permutation, 10 repeats): pay_status_3, pay_status_2

## Limitations and ethical considerations
- Decision threshold must be chosen from business costs, not from 0.5.
- Feature importance describes the model, **not** causal effects.
- Monitor for distribution shift before any reuse; retrain and re-document.
