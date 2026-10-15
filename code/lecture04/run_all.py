"""run_all.py -- run every Lecture 4 example in order, rebuilding samples, figures and interactive HTML.

Order matters: l4_01 produces the survey sample, which l4_03 and l4_12 read;
l4_05 produces the city and category samples (used by the maps and accessibility), l4_09 produces the A/B data.
One run reproduces every CSV, PNG and HTML used in the slides.
"""
import runpy                                                       # execute sibling scripts as programs
import sys                                                         # adjust the module search path
import time                                                        # time each example
from pathlib import Path                                           # locate the script directory reliably

HERE = Path(__file__).resolve().parent                             # directory of this script
sys.path.insert(0, str(HERE))                                      # make "import _common_l4" work
SCRIPTS = [                                                        # the run order is the lecture order
    "l4_01_docs_metadata.py",                                      # documentation audit and data dictionary
    "l4_02_descriptive_stats.py",                                  # center, spread and shape
    "l4_03_correlation_crosstab.py",                               # correlation, crosstab and restricted range
    "l4_04_multivariate.py",                                       # multivariate: heatmap and small multiples
    "l4_05_chart_zoo.py",                                          # chart zoo
    "l4_06_plotly_interactive.py",                                 # interactive charts vs static
    "l4_07_dashboard.py",                                          # Python dashboard
    "l4_08_geo_maps.py",                                           # geography: choropleth and spatial autocorrelation
    "l4_09_misleading.py",                                         # misleading vs corrected
    "l4_10_uncertainty.py",                                        # presenting uncertainty
    "l4_11_accessibility.py",                                      # palettes and accessibility
    "l4_12_report_pipeline.py",                                    # report pipeline (tables + figures + captions)
    "l4_13_highdim.py",                                            # high-dimensional visualization (projections / parallel coordinates)
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run the examples one by one
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # log marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # run it as the main program
    print(f"----- done {name} in {time.time() - started:.2f}s")     # print the elapsed time
print("\nAll Lecture 4 examples finished: data/samples in website/data/lecture04, "   # closing note (artifact locations)
      "figures in website/figures/lecture04, interactive HTML in figures/lecture04/interactive")   # closing note
