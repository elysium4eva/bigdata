"""run_all.py -- rebuild every Lecture 15 sample, figure and artefact in order.

Every line carries an English comment. The order matters: each script builds its own data, so
the whole lecture can be regenerated with one command.
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l15" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l15_01_survey_reliability.py",                                # reliability, validity, error chain
    "l15_02_conjoint.py",                                          # conjoint design and part-worths
    "l15_03_choice_logit.py",                                      # conditional logit and IIA
    "l15_04_hedonic.py",                                           # hedonic prices and response curves
    "l15_05_clickstream.py",                                       # funnel, RFM and attribution
    "l15_06_recommender.py",                                       # ALS and precision@k
    "l15_07_pricing.py",                                           # elasticity and optimal price
    "l15_08_churn.py",                                             # churn and expected-loss threshold
    "l15_09_clv.py",                                               # retention curves and CLV
    "l15_10_experiment.py",                                        # A/B test, power and CUPED
    "l15_11_glossary.py",                                          # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 15 examples finished: data in website/data/lecture15, "   # closing summary line one
      "figures in website/figures/lecture15, artefacts in data/lecture15/artifacts")   # closing summary
