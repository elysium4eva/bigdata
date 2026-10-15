"""run_all.py -- run every data type example in one go: sample data and figures."""  # module comment
import runpy                                                          # use runpy to execute other .py files as scripts
import sys                                                            # used to change the module search path
import time                                                           # used to time each script
from pathlib import Path                                              # path helper

HERE = Path(__file__).resolve().parent                                # folder of this script: website/code
sys.path.insert(0, str(HERE))                                         # make sure the child scripts can import _common
SCRIPTS = ["01_datatype_numeric.py",                                  # numeric data, continuous and discrete
           "02_datatype_categorical.py",                              # categorical data, nominal and ordinal
           "03_datatype_text.py",                                     # text data
           "04_datatype_image.py",                                    # image data
           "05_datatype_audio.py",                                    # audio data
           "06_datatype_video.py",                                    # video data
           "07_datatype_graph.py",                                    # graph and network data
           "08_datatype_spatiotemporal.py",                           # spatiotemporal data
           "09_datastructure_cross_time_panel.py",                    # cross section / time series / panel
           "10_datatype_structured_unstructured.py"]                  # structured / semi-structured / unstructured
for name in SCRIPTS:                                                  # run each script in turn, the lecture order
    t0 = time.time()                                                  # record the start time
    print("\n===== run", name, "=====")                               # print the script name to follow the progress
    runpy.run_path(str(HERE / name), run_name="__main__")             # run the script as the main program
    print("----- done %s in %.2fs" % (name, time.time() - t0))        # print the elapsed time
print("\n全部完成：样本在 website/data/，图片在 website/figures/")      # closing note with the output locations
