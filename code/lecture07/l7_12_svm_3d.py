"""l7_12_svm_3d.py -- a three-dimensional support vector machine, drawn.

Every line carries an English comment. The script builds a two-class problem in R^3, fits a linear
soft-margin SVM, and draws the separating plane together with the two margin planes so that the
geometry behind w, b, the margin 2/||w|| and the support vectors becomes visible.
"""
import numpy as np                                                 # arrays
import matplotlib                                                  # backend before pyplot
matplotlib.use("Agg")                                              # headless backend
import matplotlib.pyplot as plt                                    # figures
from mpl_toolkits.mplot3d import Axes3D                            # 3D projection (side effect import)
from sklearn.svm import SVC                                        # the estimator
from pathlib import Path                                           # paths

RNG = np.random.default_rng(20261113)                              # a fixed seed
ROOT = Path(__file__).resolve().parents[2]                         # website/
FIG = ROOT / "figures" / "lecture07"                               # figure folder
DAT = ROOT / "data" / "lecture07"                                  # data folder
for d in (FIG, DAT):                                               # ensure both exist
    d.mkdir(parents=True, exist_ok=True)                           # create when missing

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK stack
plt.rcParams["axes.unicode_minus"] = False                         # real minus signs
plt.rcParams["savefig.dpi"] = 200                                  # slide resolution
plt.rcParams["savefig.bbox"] = "tight"                             # trim white space
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# ---------------------------------------------------------------- 1) a two-class problem in R^3
n = 240                                                            # 240 points
centres = np.array([[0.0, 0.0, 0.0], [2.2, 2.0, 1.6]])              # two cluster centres
labels = RNG.integers(0, 2, n)                                     # balanced labels
X = centres[labels] + RNG.normal(0, 0.85, (n, 3))                   # isotropic noise around each centre
# make the problem not perfectly separable: push four points across the gap
flip = RNG.choice(n, 6, replace=False)                             # six noisy points
X[flip] += RNG.normal(0, 1.6, (6, 3))                              # blur the boundary
y = np.where(labels == 1, 1, -1)                                   # SVM wants labels in {-1, +1}
frame = np.column_stack([X, y])                                    # the table
np.savetxt(DAT / "l7_svm_3d.csv", frame, delimiter=",", header="x1,x2,x3,y", comments="", fmt="%.4f")   # persist

# ---------------------------------------------------------------- 2) fit a soft-margin linear SVM
C = 1.0                                                            # the penalty parameter
clf = SVC(kernel="linear", C=C).fit(X, y)                           # fit in the original 3-D space
w = clf.coef_[0]                                                   # the normal vector
b = float(clf.intercept_[0])                                       # the offset
margin = 2.0 / np.linalg.norm(w)                                   # the geometric margin width
sv = clf.support_vectors_                                          # the support vectors
print("w = %s | b = %.3f | 1/||w|| = %.3f | margin 2/||w|| = %.3f | #SV = %d" % (
    np.round(w, 3), b, 1 / np.linalg.norm(w), margin, len(sv)))      # report
print("accuracy on the training set: %.3f" % clf.score(X, y))        # report
np.savetxt(DAT / "l7_svm_3d_plane.csv", np.append(w, b).reshape(1, -1), delimiter=",",
           header="w1,w2,w3,b", comments="", fmt="%.4f")            # persist the plane

# ---------------------------------------------------------------- 3) the figure
fig = plt.figure(figsize=(12.6, 4.6))                               # one row, two 3-D views
views = [(22, 18), (35, -60)]                                       # two camera angles
for k, (elev, azim) in enumerate(views, start=1):                    # loop over the views
    ax = fig.add_subplot(1, 2, k, projection="3d")                   # a 3-D axes
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], s=14, color=OKABE[5], alpha=0.75, label="正类")
    ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], s=14, color=OKABE[1], alpha=0.75, label="负类")
    ax.scatter(sv[:, 0], sv[:, 1], sv[:, 2], s=60, facecolors="none", edgecolors=OKABE[7],
               linewidths=1.3, label="支持向量")                      # circle the support vectors
    # draw the three parallel planes: w'x + b = 0 and +-1
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 12),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), 12))   # a grid in x1 and x2
    for level, colour, alpha in ((0.0, OKABE[4], 0.35), (1.0, OKABE[2], 0.22), (-1.0, OKABE[2], 0.22)):
        zz = (-w[0] * xx - w[1] * yy - b + level) / w[2]              # solve for x3
        ax.plot_surface(xx, yy, zz, color=colour, alpha=alpha, linewidth=0)   # the plane
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")   # axis labels
    ax.view_init(elev=elev, azim=azim)                              # camera
    ax.set_title("w·x+b=0 与 ±1 两个间隔面；间隔宽 2/||w|| = %.2f" % margin,
                 fontsize=10)                                       # the panel title
    if k == 1:                                                      # legend only once
        ax.legend(fontsize=7.2, loc="upper left")                   # compact legend
fig.suptitle("三维空间里的支持向量机：一个平面、两个间隔面、若干个支持向量", fontsize=13)   # the title
fig.savefig(FIG / "fig_svm_3d.png", facecolor="white")               # save
plt.close(fig)                                                     # release memory
print("[figure]", FIG / "fig_svm_3d.png")                            # log it
print("done: 3-D SVM with the margin planes")                        # closing log
