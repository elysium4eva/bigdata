"""run_all.py -- rebuild every Lecture 5 sample, figure and artefact in order.

Every line carries an English comment. Order matters: lesson 5.1 creates the credit
sample that later scripts reuse, 5.3 creates the housing sample, 5.9 the customer
sample and 5.11 trains the reinforcement learning agent from scratch.
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l5" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l5_01_pipeline_baseline.py",                                  # end-to-end pipeline and baseline
    "l5_02_cv_and_leakage.py",                                     # cross-validation and leakage protocols
    "l5_03_overfitting_curves.py",                                 # complexity and learning curves
    "l5_04_metrics_thresholds.py",                                 # metrics, thresholds and calibration
    "l5_05_regularized_regression.py",                             # OLS, ridge, lasso, elastic net
    "l5_06_knn_svm_tuning.py",                                     # KNN, SVM and grid search
    "l5_07_trees_ensembles.py",                                    # trees, forests and boosting
    "l5_08_pca_factors.py",                                        # principal components and factors
    "l5_09_clustering.py",                                         # k-means, hierarchical and DBSCAN
    "l5_10_mlp_numpy.py",                                          # neural network from scratch
    "l5_11_rl_gridworld.py",                                       # Q-learning on a grid world
    "l5_12_interpretability_modelcard.py",                         # explanation and model card
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 5 examples finished: samples in website/data/lecture05, "   # closing summary line one
      "figures in website/figures/lecture05, artefacts in data/lecture05/artifacts")   # closing summary line two
