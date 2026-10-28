"""run_all.py -- rebuild every Lecture 13 sample, figure and artefact in order.

Every line carries an English comment. Each script rebuilds its own panel from the shared seed,   # continuation of the call
so the whole lecture regenerates with one command.                      # continuation of the call
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l13" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l13_01_data.py",                                              # prices, yield curve, data quality
    "l13_02_arima.py",                                             # ACF/PACF, ARIMA, IV smile
    "l13_03_unit_root.py",                                         # ADF and the spurious regression
    "l13_04_garch.py",                                             # GARCH(1,1) by maximum likelihood
    "l13_05_var_coint.py",                                         # VAR and cointegration
    "l13_06_portfolio.py",                                         # efficient frontier and factors
    "l13_07_backtest.py",                                          # an honest backtester
    "l13_08_credit.py",                                            # default model and scorecard
    "l13_09_gnn_fraud.py",                                         # graph fraud detection
    "l13_10_deflated_sharpe.py",                                   # sentiment, RAG, deflated Sharpe
    "l13_11_glossary.py",                                          # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 13 examples finished: data in website/data/lecture13, "   # report the result
      "figures in website/figures/lecture13, artefacts in data/lecture13/artifacts")   # closing summary
