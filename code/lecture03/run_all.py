"""run_all.py -- run every Lecture 3 example in order and rebuild all samples and figures.

The order matters: l3_02 writes the quarterly extracts that l3_05 depends on, and
l3_01 writes the dirty order table read by l3_05. Running this script once
reproduces every CSV and PNG used in the slides.
"""
import runpy                                                       # execute the sibling scripts in-process
import sys                                                         # adjust the import path for _common_l3
import time                                                        # measure how long each example takes
from pathlib import Path                                           # locate the scripts reliably

HERE = Path(__file__).resolve().parent                             # the folder holding the examples
sys.path.insert(0, str(HERE))                                      # make "import _common_l3" work
SCRIPTS = [                                                        # the run order is the lecture order
    "l3_01_quality_profile.py",                                    # quality dashboard and scorecard
    "l3_02_join_concat.py",                                        # append/concat and schema alignment
    "l3_03_join_types_validate.py",                                # join semantics and key validation
    "l3_04_reshape_toolbox.py",                                    # wide/long/pivot plus encoders
    "l3_05_cleaning_pipeline.py",                                  # the five cleaning steps
    "l3_06_outliers_iqr_z_iforest.py",                             # three detection rules, three treatments
    "l3_07_missing_mechanisms.py",                                 # MCAR/MAR/MNAR simulation
    "l3_08_imputation_compare.py",                                 # five imputation strategies scored
    "l3_09_panel_balance.py",                                      # panel structure and interpolation
    "l3_10_transform_assert.py",                                   # transformations and schema contract
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example exactly once
    started = time.time()                                          # start the stopwatch
    print(f"\n===== run {name} =====")                             # progress marker in the log
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script as a program
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the duration
print("\nAll Lecture 3 examples finished: samples in website/data/lecture03, figures in website/figures/lecture03")   # summary
