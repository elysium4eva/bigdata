"""l9_06_cnn_pytorch.py -- a small convolutional network on the synthetic shape images.

Every line carries an English comment. The script trains a LeNet-style CNN on the shape
images, draws the learned first-layer filters and an activation map, and reports the
confusion matrix so the deck can contrast the CNN with the k-NN baseline from lesson 9.1.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular results
import matplotlib.pyplot as plt                                    # the figure panels
import seaborn as sns                                              # heatmaps
from sklearn.metrics import confusion_matrix, accuracy_score       # evaluation
from sklearn.model_selection import train_test_split               # split
from _common_l9 import (torch_seed, param_count, DAT, save, dump,   # shared helpers
                        style_axes, annotated, OKABE)                   # shared plotting helpers

torch = torch_seed()                                               # seed and get torch
import torch.nn as nn                                              # network modules
import torch.optim as optim                                        # optimisers

# ---------------------------------------------------------------- 1) the data
data = np.load(DAT / "l9_shapes.npz")                              # the shape images
images = data["images"].astype("float32")                          # image stack
labels = data["labels"].astype(int)                                # labels
names = ["circle", "square", "triangle", "cross"]                   # class names
X_tr, X_te, y_tr, y_te = train_test_split(images, labels, test_size=0.3, random_state=42, stratify=labels)   # split
Xtr = torch.tensor(X_tr).unsqueeze(1)                              # add the channel dimension
Xte = torch.tensor(X_te).unsqueeze(1)                              # add the channel dimension
ytr = torch.tensor(y_tr, dtype=torch.long)                         # labels as tensors
yte = torch.tensor(y_te, dtype=torch.long)                         # labels as tensors
print("train/test:", Xtr.shape, Xte.shape)                         # report

# ---------------------------------------------------------------- 2) the CNN
class SmallCNN(nn.Module):                                         # a LeNet-style network
    """A two-convolution network for 16x16 grey-scale images."""    # short docstring
    def __init__(self):                                            # constructor
        super().__init__()                                         # initialise the parent class
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 -> 8
                                      nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))   # 8 -> 4
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 4 * 4, 32), nn.ReLU(),   # dense head
                                        nn.Linear(32, 4))          # four classes
    def forward(self, x):                                          # forward pass
        """Return class scores for a batch of images."""           # short docstring
        return self.classifier(self.features(x))                   # convolutions then the head

model = SmallCNN()                                                 # build the model
print("trainable parameters:", param_count(model))                  # report the size
print("a dense layer on 256 pixels would need: %d" % (256 * 64 + 64))   # comparison number
criterion = nn.CrossEntropyLoss()                                  # the loss
opt = optim.Adam(model.parameters(), lr=0.01)                      # the optimiser

# ---------------------------------------------------------------- 3) train
history = []                                                       # loss per epoch
for epoch in range(120):                                           # training loop
    model.train()                                                  # training mode
    opt.zero_grad()                                                # clear the gradients
    out = model(Xtr)                                               # forward
    loss = criterion(out, ytr)                                     # loss
    loss.backward()                                                # backward
    opt.step()                                                     # update
    history.append(float(loss.item()))                             # record
print("final train loss: %.4f" % history[-1])                       # report
model.eval()                                                       # evaluation mode
with torch.no_grad():                                              # no gradients
    pred_te = model(Xte).argmax(dim=1).numpy()                     # test predictions
    pred_tr = model(Xtr).argmax(dim=1).numpy()                     # train predictions
test_acc = float(accuracy_score(y_te, pred_te))                    # test accuracy
train_acc = float(accuracy_score(y_tr, pred_tr))                   # train accuracy
print("train accuracy %.3f | test accuracy %.3f" % (train_acc, test_acc))   # report
cm = confusion_matrix(y_te, pred_te)                               # confusion matrix
dump(pd.DataFrame(cm, index=names, columns=names), "l9_cnn_confusion.csv")   # persist it
dump(pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "loss": np.round(history, 6)}),   # loss curve
     "l9_cnn_history.csv")                                         # persist

# ---------------------------------------------------------------- 4) the figures
filters = model.features[0].weight.detach().numpy()                 # first-layer filters
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6))                # four panels
for i in range(4):                                                 # show four filters
    kernel = filters[i, 0]                                         # one filter
    axes[0].imshow(kernel, cmap="RdPu")                            # draw it
    axes[0].set_title("four of eight 3x3 filters", fontsize=9)      # panel title
    axes[0].axis("off")                                            # hide the axes
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_cnn_filters.png")                                   # save the filter figure

fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8))                # three panels
with torch.no_grad():                                              # compute the feature map
    feat = model.features[0](Xte[:1])[0, 0].numpy()                # the first channel's activation
axes[0].imshow(Xte[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)      # the input image
axes[0].set_title("input %s" % names[int(y_te[0])])                # panel title
axes[0].axis("off")                                                # hide the axes
im = axes[1].imshow(feat, cmap="RdPu")                             # the feature map
axes[1].set_title("conv feature map")                              # panel title
axes[1].axis("off")                                                # hide the axes
plt.colorbar(im, ax=axes[1], fraction=0.046)                       # colour bar
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", cbar=False, ax=axes[2],   # the confusion matrix
            xticklabels=names, yticklabels=names)                  # labels
axes[2].set_title("confusion matrix (test acc %.3f)" % test_acc)    # panel title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_cnn_training.png")                                  # save the training figure
print("done: a small CNN on the shape images")                      # closing log
