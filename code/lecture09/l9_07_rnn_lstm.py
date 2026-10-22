"""l9_07_rnn_lstm.py -- a recurrent network and an LSTM on a delayed-recall task.

Every line carries an English comment. The script builds sequences where the model must
remember a marker several steps back, trains a hand-written RNN cell and a torch LSTM,
and draws the accuracy curves so the deck can show why gating helps long dependencies.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular results
import matplotlib.pyplot as plt                                    # the curves
from _common_l9 import (torch_seed, param_count, make_delay_task,   # shared helpers
                        save, dump, style_axes, annotated, OKABE)       # shared plotting helpers

torch = torch_seed()                                               # seed and get torch
import torch.nn as nn                                              # network modules
import torch.optim as optim                                        # optimisers

# ---------------------------------------------------------------- 1) the delayed-recall task
seqs, targets = make_delay_task(600, delay=6)                      # sequences with a marker to remember
max_len = max(len(s) for s in seqs)                                # pad to the longest sequence
X = np.zeros((len(seqs), max_len, 2), dtype="float32")             # padded tensor
mask = np.zeros((len(seqs), max_len), dtype="float32")             # valid-step mask
for i, s in enumerate(seqs):                                       # loop over the sequences
    X[i, :len(s)] = s                                              # copy the sequence
    mask[i, :len(s)] = 1.0                                         # mark the valid steps
Xt = torch.tensor(X)                                               # features
Mt = torch.tensor(mask)                                            # mask
yt = torch.tensor(targets, dtype=torch.long)                       # labels
print("sequences:", X.shape, "| positive rate: %.3f" % targets.mean())   # report

# ---------------------------------------------------------------- 2) two models: vanilla RNN and LSTM
class RNNClassifier(nn.Module):                                    # a vanilla recurrent classifier
    """A single-layer RNN followed by a linear head."""             # short docstring
    def __init__(self, hidden=24):                                 # constructor
        super().__init__()                                         # initialise the parent class
        self.rnn = nn.RNN(2, hidden, batch_first=True)             # the recurrent layer
        self.head = nn.Linear(hidden, 2)                           # the classifier
    def forward(self, x):                                          # forward pass
        """Return the class scores from the last hidden state."""   # short docstring
        out, _ = self.rnn(x)                                       # run the recurrence
        return self.head(out[:, -1])                               # take the last step

class LSTMClassifier(nn.Module):                                   # a gated recurrent classifier
    """A single-layer LSTM followed by a linear head."""            # short docstring
    def __init__(self, hidden=24):                                 # constructor
        super().__init__()                                         # initialise the parent class
        self.lstm = nn.LSTM(2, hidden, batch_first=True)           # the gated recurrent layer
        self.head = nn.Linear(hidden, 2)                           # the classifier
    def forward(self, x):                                          # forward pass
        """Return the class scores from the last hidden state."""   # short docstring
        out, _ = self.lstm(x)                                      # run the recurrence
        return self.head(out[:, -1])                               # take the last step

torch.manual_seed(0)                                               # same start for both models
rnn_model = RNNClassifier(24)                                      # the vanilla model
torch.manual_seed(0)                                               # same start
lstm_model = LSTMClassifier(24)                                    # the gated model
print("RNN params %d | LSTM params %d" % (param_count(rnn_model), param_count(lstm_model)))   # report

# ---------------------------------------------------------------- 3) train both
def train(model, epochs=60, lr=0.01):                              # a shared training routine
    """Train one model and return its accuracy and loss histories."""   # short docstring
    opt = optim.Adam(model.parameters(), lr=lr)                    # the optimiser
    criterion = nn.CrossEntropyLoss()                              # the loss
    accs, losses = [], []                                          # histories
    for epoch in range(epochs):                                    # training loop
        model.train()                                              # training mode
        opt.zero_grad()                                            # clear the gradients
        out = model(Xt)                                            # forward
        loss = criterion(out, yt)                                  # loss
        loss.backward()                                            # backward
        opt.step()                                                 # update
        acc = float((out.argmax(dim=1) == yt).float().mean())       # running accuracy
        accs.append(acc)                                           # record
        losses.append(float(loss.item()))                          # record
    return accs, losses                                            # return the histories

rnn_acc, rnn_loss = train(rnn_model)                               # train the vanilla RNN
lstm_acc, lstm_loss = train(lstm_model)                            # train the LSTM
print("RNN final accuracy %.3f | LSTM final accuracy %.3f" % (rnn_acc[-1], lstm_acc[-1]))   # report
summary = pd.DataFrame({"model": ["RNN", "LSTM"],                  # model names
                        "final_accuracy": [round(rnn_acc[-1], 4), round(lstm_acc[-1], 4)],   # accuracy
                        "final_loss": [round(rnn_loss[-1], 4), round(lstm_loss[-1], 4)]})   # loss
dump(summary, "l9_rnn_lstm.csv")                                   # persist
dump(pd.DataFrame({"epoch": np.arange(1, len(rnn_acc) + 1),         # the epoch index
                   "rnn_acc": np.round(rnn_acc, 4), "lstm_acc": np.round(lstm_acc, 4),   # accuracies
                   "rnn_loss": np.round(rnn_loss, 4), "lstm_loss": np.round(lstm_loss, 4)}),   # losses
     "l9_rnn_lstm_histories.csv")                                  # persist the histories

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))                # two panels
axes[0].plot(np.arange(1, len(rnn_acc) + 1), rnn_acc, lw=1.8, color=OKABE[5], label="RNN")   # RNN accuracy
axes[0].plot(np.arange(1, len(lstm_acc) + 1), lstm_acc, lw=1.8, color=OKABE[2], label="LSTM")   # LSTM accuracy
style_axes(axes[0], "Delayed recall: gating helps", "epoch", "accuracy")   # shared style
axes[0].legend(fontsize=7.4, loc="lower right")                     # compact legend
axes[1].plot(np.arange(1, len(rnn_loss) + 1), rnn_loss, lw=1.7, color=OKABE[5], label="RNN loss")   # RNN loss
axes[1].plot(np.arange(1, len(lstm_loss) + 1), lstm_loss, lw=1.7, color=OKABE[2], label="LSTM loss")   # LSTM loss
axes[1].set_yscale("log")                                          # losses span orders of magnitude
style_axes(axes[1], "Loss: the vanilla RNN struggles first", "epoch", "loss (log)")   # shared style
axes[1].legend(fontsize=7.4, loc="upper right")                     # compact legend
annotated(axes[1], "remember 6+ steps", "lower left")              # annotate the task
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_rnn_lstm.png")                                      # save the figure
print("done: RNN and LSTM on the delayed-recall task")              # closing log
