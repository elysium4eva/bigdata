"""run_all.py -- rebuild every Lecture 9 sample, figure and artefact in order.

Every line carries an English comment. The order matters: lesson 9.1 creates the image
stack, 9.3 the two-moons sample, and later scripts reuse both.
"""
import runpy                                                       # execute the sibling scripts
import sys                                                         # adjust the module search path
import time                                                        # report how long each example takes
from pathlib import Path                                           # locate the script folder reliably

HERE = Path(__file__).resolve().parent                             # folder that holds this file
sys.path.insert(0, str(HERE))                                      # make "import _common_l9" work
SCRIPTS = [                                                        # execution order follows the lecture
    "l9_01_knn_images.py",                                         # the k-NN baseline on images
    "l9_02_linear_softmax.py",                                     # linear classifier, softmax, loss
    "l9_03_mlp_numpy.py",                                          # MLP from scratch and a gradient check
    "l9_04_torch_autograd.py",                                     # torch autograd and learning rates
    "l9_05_optimizers.py",                                         # four optimisers compared
    "l9_06_cnn_pytorch.py",                                        # a small CNN on the shape images
    "l9_07_rnn_lstm.py",                                           # RNN and LSTM on a recall task
    "l9_08_attention.py",                                          # attention from scratch
    "l9_09_finetune_lora.py",                                      # probing, fine-tuning and LoRA
    "l9_10_training_engineering.py",                               # batch size, accumulation, memory
    "l9_11_glossary.py",                                           # the bilingual glossary
]                                                                  # end of the run list
for name in SCRIPTS:                                               # run each example as a main program
    started = time.time()                                          # start the timer
    print(f"\n===== run {name} =====")                             # progress marker
    runpy.run_path(str(HERE / name), run_name="__main__")           # execute the script
    print(f"----- done {name} in {time.time() - started:.2f}s")     # report the elapsed time
print("\nAll Lecture 9 examples finished: samples in website/data/lecture09, "   # closing summary line one
      "figures in website/figures/lecture09, artefacts in data/lecture09/artifacts")   # closing summary
