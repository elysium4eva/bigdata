"""run_all.py -- rebuild every Lecture 6 sample, figure and artefact in order.

Every line carries an English comment. The order matters: lesson 6.1 creates the
prostate sample, 6.5 the two-class sample, 6.7 the three-class sample and 6.9 the
imbalanced sample that later scripts reuse.
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l6" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l6_01_linreg_normal_equation.py",                             # least squares three ways
    "l6_02_numerical_stability.py",                                # condition number and stable solvers
    "l6_03_mle_gaussian.py",                                       # Gaussian and Bernoulli likelihoods
    "l6_04_logsumexp.py",                                          # underflow and log-sum-exp
    "l6_05_linreg_vs_sigmoid.py",                                  # step, linear, sigmoid and gradient descent
    "l6_06_logistic_torch.py",                                     # PyTorch autograd against the analytic gradient
    "l6_07_softmax_regression.py",                                 # multi-class softmax regression
    "l6_08_regularization_cv.py",                                  # LASSO, Ridge and Elastic Net with CV
    "l6_09_metrics_roc.py",                                        # imbalance, thresholds, ROC/PR, calibration
    "l6_10_statsmodels_inference.py",                              # the inference view
    "l6_11_overfit_curves.py",                                     # approximation-overfit and bias-variance
    "l6_12_glossary.py",                                           # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 6 examples finished: samples in website/data/lecture06, "   # # closing summary line one
      "figures in website/figures/lecture06, artefacts in data/lecture06/artifacts")   # closing summary
