"""run_all.py -- rebuild every Lecture 7 sample, figure and artefact in order.

Every line carries an English comment. The order matters: lesson 7.1 creates the
two-class sample, 7.3 the k-NN sample, 7.5 the poverty sample, 7.7 the tabular
sample that 7.8 and 7.9 reuse.
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l7" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l7_01_svm_margin.py",                                         # maximum margin and the cost parameter
    "l7_02_svm_kernels.py",                                        # four kernels and a tuned RBF
    "l7_03_knn_k_curve.py",                                        # k-NN maps and the k curve
    "l7_04_curse_dimensionality.py",                               # distance concentration
    "l7_05_tree_poverty.py",                                       # CART on the poverty sample
    "l7_06_random_forest_oob.py",                                  # bagging, random forest and OOB
    "l7_07_gbdt_compare.py",                                       # XGBoost, LightGBM, CatBoost
    "l7_08_optuna_search.py",                                      # hyper-parameter search
    "l7_09_shap_audit.py",                                         # SHAP explanations and the audit table
    "l7_10_imbalance_cost.py",                                     # imbalance and cost-sensitive thresholds
    "l7_11_adaboost_scratch.py",                                   # AdaBoost from scratch
    "l7_12_glossary.py",                                           # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 7 examples finished: samples in website/data/lecture07, "   # closing summary line one
      "figures in website/figures/lecture07, artefacts in data/lecture07/artifacts")   # closing summary
