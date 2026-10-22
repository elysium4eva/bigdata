"""run_all.py -- rebuild every Lecture 8 sample, figure and artefact in order.

Every line carries an English comment. The order matters: lesson 8.1 creates the phone
sample, 8.2 the image patches, 8.3 the Fama-French panel and so on.
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l8" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l8_01_pca_basics.py",                                         # PCA directions, scree and scaling
    "l8_02_pca_images.py",                                         # PCA as image compression
    "l8_03_factor_fama.py",                                        # factor loadings and Fama-French
    "l8_04_perceptual_map.py",                                     # the perceptual map
    "l8_05_kmeans.py",                                             # k-means, Voronoi and the k choice
    "l8_06_hierarchical.py",                                       # dendrograms and the single-link effect
    "l8_07_dbscan.py",                                             # DBSCAN, HDBSCAN and noise
    "l8_08_gmm_em.py",                                             # Gaussian mixture and EM
    "l8_09_metrics.py",                                            # internal and external metrics
    "l8_10_anomaly_rules.py",                                      # anomaly detection and association rules
    "l8_11_embeddings.py",                                         # PCA, t-SNE and UMAP
    "l8_12_faiss_ann.py",                                          # approximate nearest neighbours
    "l8_13_topics.py",                                             # NMF and LDA topic models
    "l8_14_glossary.py",                                           # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 8 examples finished: samples in website/data/lecture08, "   # closing summary line one
      "figures in website/figures/lecture08, artefacts in data/lecture08/artifacts")   # closing summary
