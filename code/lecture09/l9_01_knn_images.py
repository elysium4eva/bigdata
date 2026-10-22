"""l9_01_knn_images.py -- the nearest-neighbour baseline on a synthetic image set.

Every line carries an English comment. The script saves the synthetic shape images, runs
a nearest-neighbour and a k-NN classifier, sweeps k, and draws the images and their
nearest neighbours so the deck can show both the baseline accuracy and the failure mode.
"""
import numpy as np                                                 # arrays for the images
import pandas as pd                                                # tabular results
import matplotlib.pyplot as plt                                    # the figure panels
from sklearn.neighbors import KNeighborsClassifier                 # the reference implementation
from sklearn.model_selection import train_test_split               # the split
from sklearn.metrics import accuracy_score                         # the metric
from _common_l9 import (make_shapes, RNG, DAT, save, dump,         # shared helpers
                        style_axes, annotated, OKABE)                   # shared plotting helpers

# ---------------------------------------------------------------- 1) the synthetic image set
images, labels, names = make_shapes(60, 16)                        # four classes of 16x16 images
np.savez(DAT / "l9_shapes.npz", images=images, labels=labels)       # persist the image stack
print("[data]", DAT / "l9_shapes.npz")                              # log the artefact
print("images:", images.shape, "| classes:", len(names))            # report the shapes
X = images.reshape(len(images), -1)                                # flatten every image into a vector
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42, stratify=labels)   # split
print("train/test:", X_train.shape[0], X_test.shape[0])             # report the split

# ---------------------------------------------------------------- 2) nearest neighbour with two distances
rows = []                                                          # results per configuration
for metric in ["euclidean", "manhattan"]:                          # two distance metrics
    for k in [1, 3, 5, 10, 20]:                                    # a few neighbourhood sizes
        model = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X_train, y_train)   # fit
        acc = accuracy_score(y_test, model.predict(X_test))         # test accuracy
        rows.append({"metric": metric, "k": k, "test_accuracy": round(float(acc), 4)})   # record
        print("%-10s k=%2d | accuracy %.3f" % (metric, k, acc))      # report
sweep = pd.DataFrame(rows)                                         # assemble the table
dump(sweep, "l9_knn_sweep.csv")                                    # persist it
best = sweep.loc[sweep["test_accuracy"].idxmax()]                  # the best configuration
print("best:", dict(best))                                         # report the winner

# ---------------------------------------------------------------- 3) nearest neighbours of a test image
model = KNeighborsClassifier(n_neighbors=5, metric="euclidean").fit(X_train, y_train)   # the reference model
distances, indices = model.kneighbors(X_test[:1])                  # the five nearest training images
print("nearest labels:", y_train[indices[0]].tolist())             # their labels
print("query label:", int(y_test[0]))                              # the true label of the query

# ---------------------------------------------------------------- 4) a figure with the data and the neighbours
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6))                # four panels
for i, name in enumerate(names):                                   # one example per class
    ax = axes[min(i, 2)]                                           # the first three classes share panels
    if i < 3:                                                      # draw the class examples
        ax.imshow(images[labels == i][0], cmap="gray", vmin=0, vmax=1)   # the first image of the class
        ax.set_title(name)                                         # panel title
        ax.axis("off")                                             # hide the axes
axes[3].plot(sweep[sweep["metric"] == "euclidean"]["k"], sweep[sweep["metric"] == "euclidean"]["test_accuracy"],   # euclidean curve
             "-o", ms=4, color=OKABE[2], label="euclidean")        # legend label
axes[3].plot(sweep[sweep["metric"] == "manhattan"]["k"], sweep[sweep["metric"] == "manhattan"]["test_accuracy"],   # manhattan curve
             "-s", ms=4, color=OKABE[5], label="manhattan")        # legend label
style_axes(axes[3], "k-NN accuracy versus k", "k", "test accuracy")   # shared style
axes[3].legend(fontsize=7.4, loc="lower right")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_knn_images.png")                                    # save the examples figure

# ---------------------------------------------------------------- 5) a figure: the five neighbours
fig, axes = plt.subplots(1, 6, figsize=(13.6, 2.6))                # query plus five neighbours
axes[0].imshow(X_test[0].reshape(16, 16), cmap="gray", vmin=0, vmax=1)   # the query image
axes[0].set_title("query (%s)" % names[int(y_test[0])], fontsize=9)   # its label
axes[0].axis("off")                                                # hide the axes
for j in range(5):                                                 # the five neighbours
    ax = axes[j + 1]                                               # the panel
    ax.imshow(X_train[indices[0][j]].reshape(16, 16), cmap="gray", vmin=0, vmax=1)   # neighbour image
    ax.set_title("%s\n%.1f" % (names[int(y_train[indices[0][j]])], distances[0][j]), fontsize=8)   # label and distance
    ax.axis("off")                                                 # hide the axes
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_knn_neighbours.png")                                # save the neighbour figure

# ---------------------------------------------------------------- 6) the five challenges figure
fig, axes = plt.subplots(1, 5, figsize=(13.6, 2.9))                # five panels, one per challenge
base = images[labels == 0][0]                                      # a circle to transform
variants = {"original": base,                                      # the original
            "shift": np.roll(base, 3, axis=1),                     # viewpoint-like shift
            "darker": np.clip(base * 0.45, 0, 1),                  # illumination change
            "stretched": np.repeat(base[::2], 2, axis=0),          # deformation
            "occluded": np.where(np.arange(16)[None, :] < 8, 0.0, base)}   # occlusion by half
for ax, (name, img) in zip(axes, variants.items()):                # loop over the variants
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)                    # show the variant
    ax.set_title(name, fontsize=9)                                 # panel title
    ax.axis("off")                                                 # hide the axes
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_challenges.png")                                    # save the challenge figure
print("done: k-NN baseline, neighbours and the five challenges")     # closing log
