"""run_all.py -- rebuild every Lecture 12 sample, figure and artefact in order.

Every line carries an English comment. Each script rebuilds its own data from the shared seed, so   # continuation of the call
the whole lecture regenerates with one command.                         # continuation of the call
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l12" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l12_01_simpson.py",                                           # Simpson's paradox
    "l12_02_potential.py",                                         # potential outcomes and bias
    "l12_03_power.py",                                             # power and multiple testing
    "l12_04_psm.py",                                               # propensity score matching
    "l12_05_did.py",                                               # DID and the event study
    "l12_06_iv.py",                                                # 2SLS and weak instruments
    "l12_07_rd.py",                                                # regression discontinuity
    "l12_08_late.py",                                              # non-compliance and LATE
    "l12_09_dml.py",                                               # double machine learning
    "l12_10_causal_forest.py",                                     # causal forest and CATE
    "l12_11_glossary.py",                                          # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 12 examples finished: data in website/data/lecture12, "   # report the result
      "figures in website/figures/lecture12, artefacts in data/lecture12/artifacts")   # closing summary
