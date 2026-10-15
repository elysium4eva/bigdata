"""l5_03_overfitting_curves.py -- complexity, under/overfitting and learning curves.

Every line carries an English comment. The script builds a synthetic housing price
dataset with a non-linear relationship, then contrasts a too-simple model, a good
model and a too-flexible model, and finally plots learning curves for diagnosis.
"""
import numpy as np                                                 # numeric arrays for the feature construction
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # figures for the complexity and learning curves
from sklearn.linear_model import LinearRegression, Ridge           # the estimators compared at several complexities
from sklearn.metrics import mean_squared_error                     # error measure for the curves
from sklearn.model_selection import (train_test_split,             # hold-out split for the error curves
                                     learning_curve)               # learning curve helper
from sklearn.pipeline import Pipeline                              # keep polynomial expansion and model together
from sklearn.preprocessing import PolynomialFeatures, StandardScaler   # polynomial expansion and scaling
from _common_l5 import RNG, save, dump, style_axes, OKABE          # shared helpers

# ---------------------------------------------------------------- 1) synthetic housing data with curvature
N = 1200                                                           # number of properties
area = np.clip(RNG.normal(95, 32, N), 25, 260)                     # living area in square metres
rooms = np.clip(np.round(area / 32 + RNG.normal(0, 0.6, N)), 1, 9)   # number of rooms, correlated with area
age = np.clip(RNG.normal(14, 9, N), 0, 60)                         # building age in years
dist_cbd = np.clip(RNG.lognormal(1.2, 0.6, N), 0.4, 40)            # distance to the city centre in kilometres
floor = RNG.integers(1, 34, N)                                     # floor number
district = RNG.choice(["nanshan", "futian", "baoan", "longgang"], N,   # district as a categorical feature
                      p=[0.3, 0.22, 0.3, 0.18])                    # sampling probabilities per district
district_mult = pd.Series(district).map({"nanshan": 1.35,           # premium multiplier per district
                                         "futian": 1.22,            # central district
                                         "baoan": 0.95,             # western district
                                         "longgang": 0.8}).values   # eastern district
price = (0.9 * area ** 1.32                                         # strong non-linear effect of area
         + 18.0 * rooms                                             # each extra room adds value
         - 1.9 * age                                                # age depreciates the property
         - 2.6 * dist_cbd ** 1.1                                    # distance hurts with slight curvature
         + 0.55 * floor                                             # higher floors are slightly pricier
         + 28.0 * (area / 100) * district_mult                       # interaction between size and district
         + RNG.normal(0, 22, N))                                    # residual noise in thousands of CNY
housing = pd.DataFrame({"area": area.round(1), "rooms": rooms,     # assemble the feature columns
                        "age": age.round(1), "dist_cbd": dist_cbd.round(2),   # continuous predictors
                        "floor": floor, "district": district,       # floor and categorical district
                        "price_k": np.round(price, 1)})            # target in thousands of CNY
dump(housing, "l5_housing.csv")                                    # persist the sample for the later scripts
print("housing rows:", len(housing), "| price mean %.0f k" % housing["price_k"].mean())   # report the scale

# ---------------------------------------------------------------- 2) features and a hold-out split
X = pd.get_dummies(housing.drop(columns=["price_k"]), drop_first=True)   # one-hot encode the district column
y = housing["price_k"].values                                      # target vector in thousands of CNY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)   # 75/25 split

# ---------------------------------------------------------------- 3) sweep the polynomial degree
rows = []                                                          # one row per degree
for degree in range(1, 13):                                        # degrees one to twelve
    pipe = Pipeline(steps=[("poly", PolynomialFeatures(degree=degree, include_bias=False)),   # polynomial expansion
                           ("scale", StandardScaler()),            # standardise the expanded features
                           ("model", Ridge(alpha=1.0))])           # ridge keeps the high degrees stable
    pipe.fit(X_train, y_train)                                     # fit on the training split
    train_rmse = np.sqrt(mean_squared_error(y_train, pipe.predict(X_train)))   # in-sample error
    test_rmse = np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))   # out-of-sample error
    rows.append({"degree": degree,                                 # polynomial degree
                 "n_features": int(pipe.named_steps["poly"].n_output_features_),   # expanded feature count
                 "train_rmse": round(float(train_rmse), 2),        # training error in thousands of CNY
                 "test_rmse": round(float(test_rmse), 2)})         # test error in thousands of CNY
curve = pd.DataFrame(rows)                                         # assemble the complexity table
dump(curve, "l5_complexity_table.csv")                             # persist the table
best_degree = int(curve.loc[curve["test_rmse"].idxmin(), "degree"])   # degree with the lowest test error
print(curve.to_string(index=False))                                # print the full sweep
print("best degree by test RMSE:", best_degree)                    # report the selected complexity

# ---------------------------------------------------------------- 4) learning curves for two regimes
def curve_for(degree, ridge_alpha):                                # helper returning train/test curves
    """Return training sizes plus training and test errors for one configuration."""   # short docstring
    pipe = Pipeline(steps=[("poly", PolynomialFeatures(degree=degree, include_bias=False)),   # complexity knob
                           ("scale", StandardScaler()),            # scaling inside the pipeline
                           ("model", Ridge(alpha=ridge_alpha))])   # regularisation strength
    sizes, train_err, test_err = learning_curve(                   # compute the learning curve
        pipe, X_train, y_train, cv=5, shuffle=True, random_state=42,   # five folds, shuffled for stability
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="neg_root_mean_squared_error")   # RMSE in negative form
    return sizes, -train_err.mean(axis=1), -test_err.mean(axis=1)   # flip the sign back to positive RMSE

sizes_over, train_over, test_over = curve_for(degree=12, ridge_alpha=0.001)   # high variance regime
sizes_good, train_good, test_good = curve_for(degree=3, ridge_alpha=1.0)   # balanced regime
sizes_simple, train_simple, test_simple = curve_for(degree=1, ridge_alpha=1.0)   # high bias regime

# ---------------------------------------------------------------- 5) figure one: complexity and fit quality
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels for the complexity story
axes[0].plot(curve["degree"], curve["train_rmse"], "-o", ms=4, color=OKABE[2], label="training error")   # in-sample
axes[0].plot(curve["degree"], curve["test_rmse"], "-s", ms=4, color=OKABE[5], label="test error")   # out-of-sample
axes[0].axvline(best_degree, color="#66717D", ls="--", lw=1.0)     # mark the best degree
style_axes(axes[0], "Underfit -> balanced -> overfit", "Polynomial degree", "RMSE (thousand CNY)")   # shared style
axes[0].legend(fontsize=8.0)                                       # legend inside the panel
grid = np.linspace(X_train["area"].min(), X_train["area"].max(), 200)   # area grid for the fitted curves
median_row = X_train.median(numeric_only=True).to_dict()           # a typical row to hold other features fixed
for degree, colour, label in [(1, OKABE[0], "degree 1 (underfit)"),   # the too-simple model
                              (3, OKABE[2], "degree 3 (good)"),   # the balanced model
                              (12, OKABE[5], "degree 12 (overfit)")]:   # the too-flexible model
    grid_X = pd.DataFrame([{**median_row, "area": a} for a in grid])   # build a frame along the area axis
    grid_X = grid_X[X_train.columns]                               # keep the training column order
    pipe = Pipeline(steps=[("poly", PolynomialFeatures(degree=degree, include_bias=False)),   # matching complexity
                           ("scale", StandardScaler()),            # matching preprocessing
                           ("model", Ridge(alpha=0.001 if degree == 12 else 1.0))]).fit(X_train, y_train)   # matching model
    axes[1].plot(grid, pipe.predict(grid_X), lw=1.9, color=colour, label=label)   # draw the fitted relationship
axes[1].scatter(X_train["area"], y_train, s=7, color="#B9B3B6", alpha=0.45, label="training data")   # raw points
axes[1].set_ylim(0, np.percentile(y_train, 99.5))                  # crop the y axis for readability
style_axes(axes[1], "Same data, three hypotheses", "Area (sqm)", "Price (thousand CNY)")   # shared style
axes[1].legend(fontsize=7.6, loc="upper left")                     # compact legend
axes[2].bar(["degree 1", "degree 3", "degree 12"],                 # compare the three regimes side by side
            [float(curve.loc[curve["degree"] == 1, "test_rmse"].iloc[0]),   # test error of the simple model
             float(curve.loc[curve["degree"] == 3, "test_rmse"].iloc[0]),   # test error of the balanced model
             float(curve.loc[curve["degree"] == 12, "test_rmse"].iloc[0])],   # test error of the flexible model
            color=[OKABE[0], OKABE[2], OKABE[5]], alpha=0.88)       # one colour per regime
style_axes(axes[2], "Test error tells the truth", "", "Test RMSE (thousand CNY)")   # shared style
for i, v in enumerate([float(curve.loc[curve["degree"] == d, "test_rmse"].iloc[0]) for d in (1, 3, 12)]):   # label bars
    axes[2].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9.5)   # value above each bar
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_complexity_fit.png")                                # save the complexity figure

# ---------------------------------------------------------------- 6) figure two: learning curves as diagnostics
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5))                # three panels, one per regime
for ax, (sizes, tr, te, name, colour) in zip(                          # iterate over the three regimes
        axes, [(sizes_simple, train_simple, test_simple, "high bias (degree 1)", OKABE[0]),   # too simple
               (sizes_good, train_good, test_good, "balanced (degree 3)", OKABE[2]),   # about right
               (sizes_over, train_over, test_over, "high variance (degree 12)", OKABE[5])]):   # too flexible
    ax.plot(sizes, tr, "-o", ms=4, color=colour, label="training error")   # in-sample curve
    ax.plot(sizes, te, "-s", ms=4, color="#66717D", label="cross-validated error")   # out-of-sample curve
    gap = te[-1] - tr[-1]                                          # final gap between the two curves
    ax.fill_between(sizes, tr, te, color=colour, alpha=0.12)        # shade the gap that diagnostics look at
    ax.annotate(f"gap {gap:.0f}", xy=(sizes[-1], (tr[-1] + te[-1]) / 2),   # annotate the gap at the right edge
                xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=8.4, color="#1E2630")   # small label
    style_axes(ax, name, "Training rows", "RMSE (thousand CNY)")    # shared style per panel
    ax.legend(fontsize=7.6, loc="upper right")                      # legend inside each panel
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_learning_curve.png")                                # save the learning-curve figure
