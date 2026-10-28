"""run_all.py -- rebuild every Lecture 11 sample, figure and artefact in order.

Every line carries an English comment. Everything runs offline: the local chat model is built from   # continuation of the call
a small in-memory corpus, so no API key and no network access are needed.   # continuation of the call
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l11" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l11_01_tokenize.py",                                          # tokens, lineage, scaling laws
    "l11_02_temperature.py",                                       # probabilities and the context window
    "l11_03_dpo.py",                                               # alignment, DPO and test-time compute
    "l11_04_eval.py",                                              # a minimal evaluation harness
    "l11_05_prompt.py",                                            # templates, few-shot, long documents
    "l11_06_cot.py",                                               # chain-of-thought and voting
    "l11_07_tools.py",                                             # function calling and the agent loop
    "l11_08_rag.py",                                               # chunking, retrieval, citations
    "l11_09_hybrid.py",                                            # hybrid retrieval, cost, model choice
    "l11_10_hallucination.py",                                     # guardrails and the trade-off curve
    "l11_11_glossary.py",                                          # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 11 examples finished: data in website/data/lecture11, "   # report the result
      "figures in website/figures/lecture11, artefacts in data/lecture11/artifacts")   # closing summary
