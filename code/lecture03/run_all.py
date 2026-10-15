"""run_all.py —— run every Lecture 3 example in order and rebuild all samples and figures.

Order matters: the two extracts produced by l3_02 are used by l3_05, and the dirty order table from l3_01 is also an input to l3_05.
Running this script once reproduces every CSV and PNG used in the slides.
"""
import runpy                                                       # run sibling scripts as scripts
import sys                                                         # adjust the module search path
import time                                                        # time each example
from pathlib import Path                                           # locate the script directory reliably

HERE = Path(__file__).resolve().parent                             # directory holding the example scripts
sys.path.insert(0, str(HERE))                                      # make "import _common_l3" work
SCRIPTS = [                                                        # run order follows the lecture
    "l3_01_quality_profile.py",                                    # quality dashboard and scorecard
    "l3_02_join_concat.py",                                        # vertical concat and schema alignment
    "l3_03_join_types_validate.py",                                # join semantics and key validation
    "l3_04_reshape_toolbox.py",                                    # wide/long, pivot and encoding tools
    "l3_05_cleaning_pipeline.py",                                  # the five-step cleaning method
    "l3_06_outliers_iqr_z_iforest.py",                             # three detection rules, three treatments
    "l3_07_missing_mechanisms.py",                                 # MCAR/MAR/MNAR simulation
    "l3_08_imputation_compare.py",                                 # scoring five imputation strategies
    "l3_09_panel_balance.py",                                      # panel structure and interpolation
    "l3_10_transform_assert.py",                                   # transforms and the schema contract
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # mark the current script in the log
    runpy.run_path(str(HERE / name), run_name="__main__")           # run the script as __main__
    print(f"----- done {name} in {time.time() - started:.2f}s")     # print the elapsed time
print("\nAll Lecture 3 examples finished: samples in website/data/lecture03, figures in website/figures/lecture03")   # closing note
