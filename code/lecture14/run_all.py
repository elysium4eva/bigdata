"""run_all.py -- rebuild every Lecture 14 sample, figure and artefact in order.

Every line carries an English comment. Each script rebuilds its own panel from the shared seed,   # continuation of the call
so the whole lecture regenerates with one command.                      # continuation of the call
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l14" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l14_01_data_sources.py",                                      # three layers and the firm match
    "l14_02_gravity_ols.py",                                       # gravity with OLS and fixed effects
    "l14_03_ppml.py",                                              # PPML and zero trade
    "l14_04_did_event.py",                                         # DID and the event study
    "l14_05_network.py",                                           # network metrics and communities
    "l14_06_gvc.py",                                               # WIOT and GVC participation
    "l14_07_gnn.py",                                               # message passing versus gravity
    "l14_08_resilience.py",                                        # AIS ports and resilience
    "l14_09_text_extract.py",                                      # field extraction from declarations
    "l14_10_gpr.py",                                               # geopolitical risk index
    "l14_11_glossary.py",                                          # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 14 examples finished: data in website/data/lecture14, "   # report the result
      "figures in website/figures/lecture14, artefacts in data/lecture14/artifacts")   # closing summary
