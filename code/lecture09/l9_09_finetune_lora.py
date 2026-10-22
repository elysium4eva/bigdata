"""l9_09_finetune_lora.py -- linear probing, full fine-tuning and LoRA on a tiny transformer.

Every line carries an English comment. The script builds a miniature BERT from a config (no
weights are downloaded), trains it on a synthetic token task in three ways, and compares the
number of trainable parameters with the resulting accuracy so the deck can show the
parameter-efficiency argument behind LoRA.
"""
import numpy as np                                                 # arrays for the synthetic task
import pandas as pd                                                # tabular results
import matplotlib.pyplot as plt                                    # the comparison figure
from _common_l9 import (torch_seed, param_count, save, dump,       # shared helpers
                        style_axes, annotated, OKABE)                   # shared plotting helpers

torch = torch_seed()                                               # seed and get torch
import torch.nn as nn                                              # network modules
import torch.optim as optim                                        # optimisers
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer   # tiny BERT pieces
from peft import LoraConfig, get_peft_model, TaskType              # parameter-efficient fine-tuning

# ---------------------------------------------------------------- 1) a tiny BERT built from a config
VOCAB, SEQ, CLASSES = 200, 24, 2                                   # vocabulary, sequence length, classes
config = BertConfig(vocab_size=VOCAB, hidden_size=64, num_hidden_layers=2,   # a very small transformer
                    num_attention_heads=2, intermediate_size=128, num_labels=CLASSES)   # two classes
print("config: hidden %d | layers %d | heads %d" % (                    # report the config
    config.hidden_size, config.num_hidden_layers, config.num_attention_heads))   # report

# ---------------------------------------------------------------- 2) a synthetic token classification task
def make_batch(n=256, seed=0):                                     # build a batch of token sequences
    """Return token ids and labels where the label follows a marker token."""   # short docstring
    rng = np.random.default_rng(seed)                              # a local generator
    ids = rng.integers(10, VOCAB - 1, (n, SEQ))                    # random token ids
    marker_pos = rng.integers(1, SEQ - 1, n)                       # where the marker sits
    labels = rng.integers(0, CLASSES, n)                           # the class to encode
    ids[np.arange(n), marker_pos] = 5 + labels                     # tokens 5 and 6 encode the class
    ids[np.arange(n), marker_pos + 1] = 5 + labels                 # repeat for robustness
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)   # return tensors

Xtr, ytr = make_batch(512, seed=1)                                 # training batch
Xte, yte = make_batch(256, seed=2)                                 # test batch
print("train", tuple(Xtr.shape), "| test", tuple(Xte.shape))        # report

# ---------------------------------------------------------------- 3) the three training modes
def build_model():                                                 # a fresh identical backbone
    """Return a new tiny BERT classifier."""                        # short docstring
    torch.manual_seed(0)                                           # same start for every mode
    return BertForSequenceClassification(config)                     # the classification model

def run(mode, epochs=40, lr=3e-3):                                 # train one mode
    """Train one configuration and return accuracy and parameter counts."""   # short docstring
    model = build_model()                                          # the model
    if mode == "linear probing":                                   # freeze the backbone
        for name, param in model.named_parameters():               # loop over the parameters
            if "classifier" not in name:                           # keep only the head trainable
                param.requires_grad = False                        # freeze it
    if mode == "LoRA":                                             # wrap with low-rank adapters
        lora_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=4, lora_alpha=16,   # rank four adapters
                              lora_dropout=0.0, target_modules=["query", "value"])   # attention projections
        model = get_peft_model(model, lora_cfg)                     # wrap the model
    trainable = param_count(model)                                 # trainable parameters
    total = sum(p.numel() for p in model.parameters())              # all parameters
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)   # only trainable ones
    criterion = nn.CrossEntropyLoss()                              # the loss
    for epoch in range(epochs):                                    # training loop
        model.train()                                              # training mode
        opt.zero_grad()                                            # clear the gradients
        loss = criterion(model(input_ids=Xtr).logits, ytr)          # forward and loss
        loss.backward()                                            # backward
        opt.step()                                                 # update
    model.eval()                                                   # evaluation mode
    with torch.no_grad():                                          # no gradients
        acc = float((model(input_ids=Xte).logits.argmax(dim=1) == yte).float().mean())   # test accuracy
    return {"mode": mode, "trainable_params": int(trainable), "total_params": int(total),   # the row
            "trainable_share": round(trainable / total, 5), "test_accuracy": round(acc, 4)}   # more fields

rows = [run(mode) for mode in ["linear probing", "full fine-tuning", "LoRA"]]   # three modes
table = pd.DataFrame(rows)                                         # assemble the table
dump(table, "l9_finetune_modes.csv")                               # persist it
print(table.to_string(index=False))                                # report

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))                # two panels
axes[0].bar(table["mode"], table["trainable_params"], color=[OKABE[1], OKABE[5], OKABE[2]], alpha=0.9)   # parameters
style_axes(axes[0], "Trainable parameters by mode", "", "parameters")   # shared style
for i, row in table.iterrows():                                    # annotate the bars
    axes[0].text(i, row["trainable_params"], "%.0f%%" % (row["trainable_share"] * 100),   # the share
                 ha="center", va="bottom", fontsize=8)             # value label
axes[1].bar(table["mode"], table["test_accuracy"], color=[OKABE[1], OKABE[5], OKABE[2]], alpha=0.9)   # accuracy
axes[1].set_ylim(0.0, 1.02)                                        # full scale
style_axes(axes[1], "Accuracy is often comparable", "", "test accuracy")   # shared style
for i, row in table.iterrows():                                    # annotate the bars
    axes[1].text(i, row["test_accuracy"], "%.3f" % row["test_accuracy"], ha="center", va="bottom", fontsize=8)   # value
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_lora_params.png")                                   # save the figure

# ---------------------------------------------------------------- 5) a GNN / RL sketch figure
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))                # two panels
rng = np.random.default_rng(0)                                     # for the graph layout
coords = rng.uniform(0, 1, (8, 2))                                 # node positions
for i in range(8):                                                 # draw the edges
    for j in range(i + 1, 8):                                      # every node pair
        if rng.random() < 0.28:                                    # keep some edges
            axes[0].plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#B9B3B6", lw=0.9)   # edge
axes[0].scatter(coords[:, 0], coords[:, 1], s=120, color=OKABE[4], zorder=3)   # nodes
for i in range(8):                                                 # label the nodes
    axes[0].text(coords[i, 0], coords[i, 1], str(i), ha="center", va="center", color="white", fontsize=8)   # label
axes[0].set_axis_off()                                             # no axes for the graph
axes[0].set_title("GNN: message passing on a graph")               # panel title
states = np.array([[0.6, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.2, 0.7]])   # three states
for s, colour in zip(states, [OKABE[5], OKABE[2], OKABE[1]]):      # one bar series per state
    axes[1].plot(range(3), s, "-o", ms=5, color=colour)            # the value curve
style_axes(axes[1], "RL: value estimates improve with interaction", "episode", "state value")   # shared style
annotated(axes[1], "Q-learning in L5", "upper left")               # link back to L5
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_gnn_rl.png")                                        # save the sketch figure
print("done: linear probing, full fine-tuning and LoRA")            # closing log
