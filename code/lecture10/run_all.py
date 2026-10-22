"""run_all.py -- rebuild every Lecture 10 corpus, figure and artefact in order.

Every line carries an English comment. The order matters: lesson 10.1 builds the corpora that
later scripts reuse, and 10.3 trains the word vectors used by 10.5.
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l10" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l10_01_preprocess.py",                                        # corpora, segmentation, coverage
    "l10_02_tfidf_baseline.py",                                    # bag of words and TF-IDF
    "l10_03_word2vec.py",                                          # word vectors from gensim
    "l10_04_cooccurrence.py",                                      # co-occurrence, SVD, GloVe idea
    "l10_05_sentiment.py",                                         # two sentiment representations
    "l10_06_ner_rules.py",                                         # rule-based NER and BIO tags
    "l10_07_topics_zh.py",                                         # LDA on the Chinese corpus
    "l10_08_llm_agreement.py",                                     # annotation agreement audit
    "l10_09_rag_citation.py",                                      # retrieval with citations
    "l10_10_domain_lexicon.py",                                    # domain lexicon and segmentation
    "l10_11_glossary.py",                                          # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 10 examples finished: corpora in website/data/lecture10, "   # closing summary line one
      "figures in website/figures/lecture10, artefacts in data/lecture10/artifacts")   # closing summary
